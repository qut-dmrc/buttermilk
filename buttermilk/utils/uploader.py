import asyncio
import atexit
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from pydantic import BaseModel
from tenacity import RetryError

from buttermilk import bm
from buttermilk._core.log import logger
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.types import BaseRecord
from buttermilk.storage import Storage


class AsyncDataUploader:

    def __init__(
        self,
        storage: Storage,
        *,
        buffer_size: int = 10,
        flush_interval: int = 30,
        max_flush_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 30.0,
    ):
        self.storage: Storage = bm.get_storage(storage) if not isinstance(storage, Storage) else storage

        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_flush_retries = max_flush_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait

        self.queue: asyncio.Queue = asyncio.Queue()
        self.buffer: list[Any] = []
        self.last_flush = time.time()
        self._shutdown: asyncio.Event = asyncio.Event()

        self.backup_dir = Path(mkdtemp())
        self.worker_task = None

        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    async def add(self, item: Any):
        """Add item (preferably a BaseRecord) to upload queue."""
        # Lazily start worker task
        if self.worker_task is None:
            worker_coroutine = self._worker()
            self.worker_task = asyncio.shield(asyncio.create_task(worker_coroutine))

        # Backup a serializable representation, but enqueue the original
        await self._backup_item(item)
        await self.queue.put(item)

    async def process(self, inputs: dict[str, Any]):
        """Process method to make AsyncDataUploader work as a Processor in pipelines.

        Adds the record to the upload queue and passes it through unchanged.

        Args:
            inputs: Input dictionary containing 'record' to upload

        Yields:
            The same inputs dict (pass-through behavior)
        """
        # Prefer BaseRecord if present
        record: Any = inputs.get("record") if isinstance(inputs, dict) else None
        to_enqueue: Any = record if record is not None else inputs
        await self.add(to_enqueue)
        yield inputs  # Pass through unchanged

    async def _worker(self):
        """Background worker that processes the queue."""
        while not (self._shutdown.is_set() and self.queue.empty()):
            try:
                # Get item with timeout
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    self.buffer.append(item)
                except TimeoutError:
                    pass

                # Check if we should flush
                should_flush = len(self.buffer) >= self.buffer_size or time.time() - self.last_flush >= self.flush_interval

                if should_flush and self.buffer:
                    await self._flush()

            except Exception as e:
                current = asyncio.current_task()
                logger.exception(
                    f"Worker error: {e}",
                    phase="worker",
                    error=str(e),
                    type=type(e).__name__,
                    buffer_len=len(self.buffer),
                    queue_size=self.queue.qsize(),
                    storage=type(self.storage).__name__,
                    task=(current.get_name() if current else None),
                )
                await asyncio.sleep(1)
        logger.info("Data uploader loop finished.")

    async def _save_with_retry(self, data: list[Any]) -> None:
        """Save data with retry logic using RetryWrapper."""
        # Create a wrapper for the storage save operation
        retry_wrapper = RetryWrapper(
            client=self.storage,
            max_retries=self.max_flush_retries,
            min_wait_seconds=self.retry_min_wait,
            max_wait_seconds=self.retry_max_wait,
        )

        # Create an async wrapper around the sync save method
        async def async_save():
            return self.storage.save(data)

        # Execute with retry logic
        await retry_wrapper._execute_with_retry(async_save)

    async def _flush(self):
        """Upload buffered items"""
        if not self.buffer:
            return

        try:
            await self._save_with_retry(self.buffer)

            self.last_flush = time.time()
            self.buffer = []
            await self._clear_backup()
        except RetryError as e:
            # All retries exhausted - dump to emergency file
            logger.error(
                f"All flush retries exhausted after {self.max_flush_retries} attempts. Dumping to emergency file.",
                error=str(e),
                phase="flush_retry_exhausted",
                buffer_len=len(self.buffer),
                storage=type(self.storage).__name__,
                max_retries=self.max_flush_retries,
            )

            # Emergency dump to disk
            emergency_file = bm.save(self.buffer, extension=".json")
            logger.error(f"Emergency data saved to: {emergency_file}")

            # Clear buffer to prevent infinite retry loop
            self.buffer = []
            self.last_flush = time.time()

        except Exception as e:
            # Unexpected error not covered by retry logic
            logger.exception(
                f"Unexpected flush error: {e}",
                error=str(e),
                type=type(e).__name__,
                args=e.args,
                traceback=e.__traceback__,
                phase="flush",
                buffer_len=len(self.buffer),
                storage=type(self.storage).__name__,
            )
            # Keep items in buffer for retry

    async def _backup_item(self, item):
        """Write item to backup file."""
        backup_file = self.backup_dir / f"backup_{datetime.now().isoformat()}.json"
        try:
            if isinstance(item, BaseModel):
                payload = item.model_dump(mode="json")
            elif isinstance(item, BaseRecord):
                # BaseRecord is a BaseModel; included above, but keep explicit branch for clarity
                payload = item.model_dump(mode="json")
            elif isinstance(item, dict):
                payload = item
            else:
                payload = {"value": str(item)}
            backup_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write backup file: {e}")

    async def _clear_backup(self):
        """Clear backup files after successful upload."""
        for f in self.backup_dir.glob("backup_*.json"):
            f.unlink()

    def shutdown(self, *_args):
        """Graceful shutdown ensuring all data is flushed."""
        self._shutdown.set()

        # Handle synchronously to avoid event loop issues
        if self.buffer:
            try:
                self.storage.save(self.buffer)
            except Exception as e:
                logger.error(f"Error during final sync flush: {e}. Falling back to emergency save.")
                bm.save(self.buffer, extension=".json")

                # Clean backup files synchronously
                for f in self.backup_dir.glob("backup_*.json"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
