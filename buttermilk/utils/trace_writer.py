"""Trace writer for ExecutionTrace storage.

This module provides a singleton TraceWriter that uses AsyncDataUploader
to efficiently batch and store ExecutionTrace objects to BigQuery.
"""

from typing import Optional

from buttermilk import get_bm, logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk.storage import Storage
from buttermilk.utils.uploader import AsyncDataUploader


class TraceWriter:
    """Singleton trace writer for ExecutionTrace storage.

    Uses AsyncDataUploader to batch traces and write them to BigQuery
    based on the configuration in conf/storage/traces.yaml.
    """

    _instance: Optional['TraceWriter'] = None
    _initialized: bool = False

    def __new__(cls) -> 'TraceWriter':
        """Ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the trace writer with storage configuration."""
        if self._initialized:
            return

        try:
            # Get the global BM instance
            bm = get_bm()

            # Look for traces storage configuration
            if hasattr(bm, 'config') and 'storage' in bm.config:
                storage_configs = bm.config.get('storage', {})

                # Check for traces storage config
                traces_config = storage_configs.get('traces')
                if traces_config:
                    # Create storage instance
                    storage = Storage(traces_config)

                    # Create uploader with reasonable defaults
                    self.uploader = AsyncDataUploader(
                        storage,
                        buffer_size=100,  # Batch 100 traces before writing
                        flush_interval=30  # Or flush every 30 seconds
                    )
                    logger.info("TraceWriter initialized with traces storage")
                else:
                    self.uploader = None
                    logger.warning("No traces storage configuration found")
            else:
                self.uploader = None
                logger.warning("No storage configuration available")

        except Exception as e:
            logger.error(f"Failed to initialize TraceWriter: {e}")
            self.uploader = None

        self._initialized = True

    async def add(self, trace: ExecutionTrace) -> None:
        """Add a trace to the upload queue.

        Args:
            trace: The ExecutionTrace to store
        """
        if self.uploader:
            try:
                await self.uploader.add(trace)
                logger.debug(f"Trace {trace.call_id} queued for upload")
            except Exception as e:
                logger.error(f"Failed to queue trace {trace.call_id}: {e}")
        else:
            logger.debug("TraceWriter not configured, trace not stored")

    async def flush(self) -> None:
        """Force flush any pending traces."""
        if self.uploader:
            try:
                await self.uploader._flush()
                logger.debug("Traces flushed to storage")
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")


# Global singleton instance
trace_writer = TraceWriter()


def get_trace_writer() -> TraceWriter:
    """Get the global TraceWriter instance.

    Returns:
        The singleton TraceWriter instance
    """
    return trace_writer