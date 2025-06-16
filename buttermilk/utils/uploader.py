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

from buttermilk._core.log import logger


class AsyncDataUploader:
    def __init__(
        self,
        buffer_size: int = 10,
        flush_interval: int = 30,
    ):
        self.dataset_name = None  # Will be determined from flow context

        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

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
    
    def configure_storage(self, dataset_name: str) -> None:
        """Configure storage destination. Should be called by orchestrator/flow, not agent.
        
        Args:
            dataset_name: Dataset name for BigQuery storage
        """
        self.dataset_name = dataset_name

    async def add(self, item: Any):
        """Add item to upload queue."""
        # Lazily start worker task
        if self.worker_task is None:
            worker_coroutine = self._worker()
            self.worker_task = asyncio.shield(asyncio.create_task(worker_coroutine))
            
        if isinstance(item, BaseModel):
            # Convert to serialisable types
            item = item.model_dump(mode="json")
        await self._backup_item(item)
        await self.queue.put(item)

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
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
        logger.info("Data uploader loop finished.")

    async def _flush(self):
        """Upload buffered items"""
        if not self.buffer:
            return

        try:
            from buttermilk._core.dmrc import get_bm
            bm = get_bm()
            
            # Determine storage from flow context or use default
            # The flow/orchestrator should configure storage, not the agent
            if self.dataset_name:
                storage = bm.get_bigquery_storage(self.dataset_name)
                storage.save(self.buffer)
            else:
                # Fallback: use BM's session-level save for agent traces
                bm.save(self.buffer, extension=".json")
                
            self.last_flush = time.time()
            self.buffer = []
            await self._clear_backup()
        except Exception as e:
            logger.error(f"Flush error: {e}")
            # Keep items in buffer for retry

    async def _backup_item(self, item):
        """Write item to backup file."""
        backup_file = self.backup_dir / f"backup_{datetime.now().isoformat()}.json"
        backup_file.write_text(json.dumps(item))

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
                from buttermilk._core.dmrc import get_bm
                bm = get_bm()
                
                if self.dataset_name:
                    storage = bm.get_bigquery_storage(self.dataset_name)
                    storage.save(self.buffer)
                else:
                    # Fallback: use BM's session-level save
                    bm.save(self.buffer, extension=".json")
            except Exception as e:
                logger.error(f"Error during final sync flush: {e}. Falling back to emergency save.")
                from buttermilk._core.dmrc import get_bm
                bm = get_bm()
                bm.save(self.buffer, extension=".json")

                # Clean backup files synchronously
                for f in self.backup_dir.glob("backup_*.json"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
