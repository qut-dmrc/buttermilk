"""Trace writer for ExecutionTrace storage.

This module provides a singleton TraceWriter that uses AsyncDataUploader
to efficiently batch and store ExecutionTrace objects to BigQuery.
"""

from typing import Optional

from buttermilk import bm, logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk.storage import Storage
from buttermilk.utils.uploader import AsyncDataUploader


class TraceWriter:
    """Singleton trace writer for ExecutionTrace storage.

    Uses AsyncDataUploader to batch traces and write them to BigQuery
    based on the configuration in conf/storage/traces.yaml.
    """

    _instance: Optional["TraceWriter"] = None
    _initialized: bool = False

    def __new__(cls) -> "TraceWriter":
        """Ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the trace writer (actual config loading is deferred)."""
        # Don't initialize storage immediately - do it lazily when first needed
        pass

    def _ensure_initialized(self):
        """Lazy initialization - only initialize when first trace is added."""
        if self._initialized:
            return

        try:
            # Debug logging for configuration investigation
            logger.debug(
                "TraceWriter lazy initialization",
                has_bm_cfg=hasattr(bm, "cfg"),
                bm_cfg_type=type(getattr(bm, "cfg", None)).__name__,
                bm_cfg_keys=list(getattr(bm, "cfg", {}).keys()) if hasattr(bm, "cfg") else None,
                has_storage_in_cfg="storage" in getattr(bm, "cfg", {})
            )

            # Look for traces storage configuration
            if hasattr(bm, "cfg") and "storage" in bm.cfg:
                storage_configs = bm.cfg.get("storage", {})

                logger.debug(
                    "Storage configuration debug",
                    storage_config_keys=list(storage_configs.keys()),
                    has_traces_config="traces" in storage_configs
                )

                # Check for traces storage config
                traces_config = storage_configs.get("traces")
                if traces_config:
                    logger.debug("Found traces config", traces_config=traces_config)
                    # Create storage instance
                    storage = bm.get_storage(traces_config)

                    # Create uploader with reasonable defaults
                    self.uploader = AsyncDataUploader(
                        storage,
                        buffer_size=100,  # Batch 100 traces before writing
                        flush_interval=30  # Or flush every 30 seconds
                    )
                    logger.info("TraceWriter initialized with traces storage")
                else:
                    self.uploader = None
                    logger.warning("No traces storage configuration found", storage_keys=list(storage_configs.keys()))
            else:
                self.uploader = None
                logger.warning(
                    "No storage configuration available",
                    has_bm_cfg=hasattr(bm, "cfg"),
                    cfg_type=type(getattr(bm, "cfg", None)).__name__
                )

        except Exception as e:
            logger.error(f"Failed to initialize TraceWriter: {e}")
            self.uploader = None

        self._initialized = True

    async def add(self, trace: ExecutionTrace) -> None:
        """Add a trace to the upload queue.

        Args:
            trace: The ExecutionTrace to store
        """
        self._ensure_initialized()
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
        self._ensure_initialized()
        if self.uploader:
            try:
                await self.uploader._flush()
                logger.debug("Traces flushed to storage")
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")


# Global singleton instance (lazy initialization)
trace_writer = None


def get_trace_writer() -> TraceWriter:
    """Get the global TraceWriter instance.

    Creates the singleton on first access to ensure BM is initialized.

    Returns:
        The singleton TraceWriter instance
    """
    global trace_writer
    if trace_writer is None:
        trace_writer = TraceWriter()
    return trace_writer
