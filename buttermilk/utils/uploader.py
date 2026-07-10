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
from buttermilk._core.constants import cache, get_base_cache_dir
from buttermilk._core.log import logger
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.types import BaseRecord
from buttermilk.storage import Storage
from buttermilk.utils.utils import scrub_serializable

# Budget (seconds) for draining the upload queue during finalization before we warn
# the researcher loudly. Aligned with the default timeout of BM.graceful_shutdown()
# (see buttermilk/_core/bm_init.py): once graceful shutdown's own budget elapses it
# force-cancels background tasks, so a drain that exceeds this risks the very data loss
# this uploader is designed to prevent. We keep draining past the warning rather than
# drop records; the warning tells the researcher how to avoid the delay next time.
SHUTDOWN_FLUSH_WARN_SECONDS = 10.0


class AsyncDataUploader:
    """Asynchronous background uploader for batching records to storage.

    Warning:
        When used with highly concurrent scrapers (e.g. TMDB) that generate data
        very rapidly, do NOT set `buffer_size` too small. A very small buffer_size
        with high-throughput extraction can cause the upload queue to fall too far behind,
        risking data loss if the shutdown phase exceeds the 10-second graceful timeout.
        For high-throughput scrapers, use a larger buffer_size (e.g., 1000-5000).
    """

    def __init__(
        self,
        storage: Storage,
        *,
        buffer_size: int = 10,
        flush_interval: int = 30,
        max_flush_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 30.0,
        use_timestamp_suffix: bool | None = None,
        output_col: str = "uri",
        shutdown_warn_seconds: float = SHUTDOWN_FLUSH_WARN_SECONDS,
    ):
        self.storage: Storage = bm.get_storage(storage) if not isinstance(storage, Storage) else storage

        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_flush_retries = max_flush_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        # Seconds to wait for the queue to drain on finalization before warning the
        # researcher. Exposed as a parameter so tests can drive it to a small value.
        self.shutdown_warn_seconds = shutdown_warn_seconds

        # Default behavior: use timestamp suffix if file exists, unless explicitly overridden
        if use_timestamp_suffix is not None:
            # Explicit value provided (True or False) - use it
            self.use_timestamp_suffix = use_timestamp_suffix
        else:
            # No explicit value - auto-detect based on file existence
            self.use_timestamp_suffix = hasattr(self.storage, "exists") and self.storage.exists()

        self.original_storage = self.storage  # Keep reference to original

        self.queue: asyncio.Queue = asyncio.Queue()
        self.buffer: list[Any] = []
        self.last_flush = time.time()
        self._shutdown: asyncio.Event = asyncio.Event()

        # Use a persistent backup directory under ~/.cache/buttermilk/backup/ instead of
        # an ephemeral temp directory. This ensures backups survive process restarts
        # and can be recovered if uploads fail during shutdown.
        self.backup_dir = self._get_persistent_backup_dir()
        self.worker_task = None

        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    def _get_persistent_backup_dir(self) -> Path:
        """Get a persistent backup directory for trace recovery.

        Uses the centralized cache infrastructure from constants.py:
        get_base_cache_dir() / cache.BACKUP / {session_id}

        Falls back to a temp directory if the persistent location cannot be created.

        Returns:
            Path to the backup directory (created if necessary)
        """
        try:
            # Try to use bm.session_info.session_id for a session-specific backup dir
            session_id = getattr(bm, "session_info", None)
            if session_id and hasattr(session_id, "session_id"):
                session_id = session_id.session_id
            else:
                session_id = "unknown"

            backup_base = get_base_cache_dir() / cache.BACKUP / session_id
            backup_base.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Using persistent backup directory: {backup_base}")
            return backup_base
        except Exception as e:
            # Fall back to temp directory if we can't create the persistent one
            logger.warning(f"Could not create persistent backup dir, using temp: {e}")
            return Path(mkdtemp())

    async def add(self, item: Any):
        """Add item (preferably a BaseRecord) to upload queue."""
        # Lazily start worker task
        if self.worker_task is None:
            worker_coroutine = self._worker()
            self.worker_task = asyncio.shield(asyncio.create_task(worker_coroutine))

        # Backup a serializable representation, but enqueue the original
        await self._backup_item(item)
        await self.queue.put(item)

    async def process(
        self,
        context,
        processor_stage: str = "save",
        **kwargs,  # Accept extra args like parent_trace_id from pipeline
    ):
        """Process method to make AsyncDataUploader work as a Processor in pipelines.

        Adds the record to the upload queue and passes it through unchanged.

        Args:
            context: ProcessingContext containing the record, or BaseRecord directly
            processor_stage: Name of this processing stage (unused)
            **kwargs: Additional arguments from pipeline (ignored)

        Yields:
            The same record (pass-through behavior)
        """
        # Extract record from ProcessingContext if needed
        record = context.record if hasattr(context, "record") else context
        await self.add(record)
        yield record  # Pass through unchanged

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

        # Ensure any remaining items in buffer are flushed before worker exits
        if self.buffer:
            try:
                await self._flush()
            except Exception as e:
                logger.error(f"Worker final flush error: {e}")

        logger.info("Data uploader loop finished.")

    async def _save_with_retry(self, data: list[Any], target_storage=None) -> None:
        """Save data with retry logic using RetryWrapper."""
        storage_to_use = target_storage if target_storage is not None else self.storage

        # Create a wrapper for the storage save operation
        retry_wrapper = RetryWrapper(
            client=storage_to_use,
            max_retries=self.max_flush_retries,
            min_wait_seconds=self.retry_min_wait,
            max_wait_seconds=self.retry_max_wait,
        )

        # Create an async wrapper around the sync save method
        async def async_save():
            return storage_to_use.save(data)

        # Execute with retry logic
        await retry_wrapper._execute_with_retry(async_save)

    def _create_timestamped_storage(self):
        """Create a storage instance with timestamped filename."""
        if not self.use_timestamp_suffix:
            return self.storage

        # Only works with FileStorage for now
        if not hasattr(self.storage, "path"):
            return self.storage

        # Create timestamp suffix
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Get original path and add timestamp before extension
        original_path = str(self.storage.path)
        if "." in original_path:
            name, ext = original_path.rsplit(".", 1)
            timestamped_path = f"{name}-{timestamp}.{ext}"
        else:
            timestamped_path = f"{original_path}-{timestamp}"

        # Create new storage with timestamped path
        new_config = self.storage.config.model_copy()
        new_config.path = timestamped_path

        # Import here to avoid circular imports
        from buttermilk.storage.file import FileStorage

        return FileStorage(new_config)

    async def _flush(self):
        """Upload buffered items"""
        if not self.buffer:
            return

        buffer_size = len(self.buffer)

        try:
            # Use timestamped storage if configured
            target_storage = self._create_timestamped_storage()
            await self._save_with_retry(self.buffer, target_storage)

            self.last_flush = time.time()
            self.buffer = []
            await self._clear_backup()
            logger.debug(
                f"{buffer_size} traces flushed to storage and buffer cleared.",
                buffer_size=buffer_size,
            )
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
            logger.error(
                f"Emergency data saved to: {emergency_file}",
                emergency_file=emergency_file,
                buffer_size=buffer_size,
            )

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
                buffer_size=buffer_size,
                storage=type(self.storage).__name__,
            )
            # Keep items in buffer for retry

    async def _backup_item(self, item):
        """Write item to backup file."""
        backup_file = self.backup_dir / f"backup_{datetime.now().isoformat()}.json"
        try:
            if isinstance(item, BaseModel):
                payload = scrub_serializable(item.model_dump())
            elif isinstance(item, BaseRecord):
                # BaseRecord is a BaseModel; included above, but keep explicit branch for clarity
                payload = scrub_serializable(item.model_dump())
            elif isinstance(item, dict):
                payload = scrub_serializable(item)
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
                # Use timestamped storage if configured
                target_storage = self._create_timestamped_storage()
                target_storage.save(self.buffer)
                self.buffer = []  # Clear buffer to prevent double-flush by worker
            except Exception as e:
                logger.error(f"Error during final sync flush: {e}. Falling back to emergency save.")
                bm.save(self.buffer, extension=".json")

                # Clean backup files synchronously
                for f in self.backup_dir.glob("backup_*.json"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

    async def finalize_processing(self) -> bool:
        """Finalize processing by flushing any remaining data.

        This method is called by the pipeline at the end of processing
        to ensure all data is safely written before completion.

        Returns:
            bool: True if finalization succeeded, False otherwise
        """
        try:
            # Trigger shutdown to tell worker to stop after queue is empty
            self._shutdown.set()

            # Wait for the worker to finish processing the queue.
            if self.worker_task is not None and not self.worker_task.done():
                logger.info("Waiting for AsyncDataUploader worker to finish draining queue...")

                # Wait up to the drain budget, then warn loudly but KEEP waiting so no
                # records are dropped. We use asyncio.wait (not asyncio.wait_for): wait_for
                # cancels its target on timeout, and worker_task is an asyncio.shield wrapper
                # (see add()) — cancelling that wrapper would leave us awaiting a cancelled
                # future. asyncio.wait just reports done/pending and never cancels.
                _done, pending = await asyncio.wait({self.worker_task}, timeout=self.shutdown_warn_seconds)
                if pending:
                    logger.warning(
                        f"AsyncDataUploader has been flushing for over {self.shutdown_warn_seconds:.0f}s "
                        f"({self.queue.qsize()} queued + {len(self.buffer)} buffered records still pending). "
                        f"Shutdown is blocked until the upload finishes so no records are lost. "
                        f"To avoid this delay, increase `buffer_size` (currently {self.buffer_size}) so records "
                        f"upload in fewer, larger batches.",
                    )
                    await self.worker_task  # keep waiting; do NOT drop records

            # Flush any remaining items in the buffer (should be handled by worker, but as a fallback)
            if self.buffer:
                await self._flush()

            return True
        except Exception as e:
            logger.error(f"Error during AsyncDataUploader finalization: {e}")
            return False
