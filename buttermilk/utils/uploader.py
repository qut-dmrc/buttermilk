import asyncio
import atexit
import json
import signal
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from pydantic import BaseModel

from buttermilk._core.config import SaveInfo
from buttermilk.bm import (  # Buttermilk global instance and logger
    get_bm,  # Buttermilk global instance and logger
    logger,
)

bm = get_bm()
from buttermilk.utils.save import upload_rows, upload_rows_async


class AsyncDataUploader:
    def __init__(
        self,
        save_dest: SaveInfo,
        buffer_size: int = 10,
        flush_interval: int = 30,
    ):
        # Validate that save_dest is a proper SaveInfo instance
        if not isinstance(save_dest, SaveInfo):
            # If it's a dict (from Hydra), try to create a SaveInfo object
            if isinstance(save_dest, Mapping):
                try:
                    save_dest = SaveInfo(**save_dest)
                except Exception as e:
                    raise TypeError(f"Failed to convert save_dest dict to SaveInfo: {e}")
            else:
                raise TypeError(f"save_dest must be a SaveInfo object, got {type(save_dest)}")

        self.save_dest = save_dest

        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.queue: asyncio.Queue = asyncio.Queue()
        self.buffer: list[Any] = []
        self.last_flush = time.time()
        self._shutdown: asyncio.Event = asyncio.Event()

        self.backup_dir = Path(mkdtemp())

        # Create the worker task and shield it
        worker_coroutine = self._worker()
        self.worker_task = asyncio.shield(asyncio.create_task(worker_coroutine))

        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    async def add(self, item: Any):
        """Add item to upload queue."""
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
            await upload_rows_async(self.buffer, save_dest=self.save_dest)
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

    def shutdown(self, *args):
        """Graceful shutdown ensuring all data is flushed."""
        self._shutdown.set()

        # Handle synchronously to avoid event loop issues
        if self.buffer:
            try:
                upload_rows(self.buffer, save_dest=self.save_dest)
            except Exception as e:
                logger.error(f"Error during final sync flush: {e}. Falling back to emergency save.")
                bm.save(self.buffer)

                # Clean backup files synchronously
                for f in self.backup_dir.glob("backup_*.json"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
