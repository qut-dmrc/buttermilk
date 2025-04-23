import asyncio
import atexit
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from cloudpathlib import CloudPath
from promptflow.tracing import trace
from pydantic import BaseModel, ConfigDict, Field, model_validator

from buttermilk._core.job import Job
from buttermilk.bm import logger
from buttermilk.utils.save import upload_rows


class AsyncDataUploader:
    def __init__(
        self,
        upload_fn,
        buffer_size: int = 1000,
        flush_interval: int = 60,
    ):
        self.upload_fn = upload_fn
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.queue: asyncio.Queue = asyncio.Queue()
        self.buffer: list[Any] = []
        self.last_flush = time.time()
        self._shutdown = False

        self.backup_dir = Path(mkdtemp())

        # Start background worker
        self.worker_task = asyncio.create_task(self._worker())

        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    async def add(self, item: Any):
        """Add item to upload queue."""
        await self.queue.put(item)
        await self._backup_item(item)

    async def _worker(self):
        """Background worker that processes the queue."""
        while not self._shutdown or not self.queue.empty():
            try:
                # Get item with timeout
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    self.buffer.append(item)
                except TimeoutError:
                    pass

                # Check if we should flush
                should_flush = (
                    len(self.buffer) >= self.buffer_size
                    or time.time() - self.last_flush >= self.flush_interval
                )

                if should_flush and self.buffer:
                    await self._flush()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def _flush(self):
        """Upload buffered items"""
        if not self.buffer:
            return

        try:
            await self.upload_fn(self.buffer)
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
        self._shutdown = True
        asyncio.run(self._flush())
