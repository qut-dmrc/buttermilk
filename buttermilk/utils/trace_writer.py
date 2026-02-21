"""Trace writer for ExecutionTrace storage.

This module provides a singleton TraceWriter that uses AsyncDataUploader
to efficiently batch and store ExecutionTrace objects to BigQuery.

Storage Configuration
---------------------
TraceWriter requires a storage config at `bm.cfg.storage.traces`. This is loaded
from the centralized storage configuration in `conf/storage/traces.yaml` or
defined directly in `conf/config.yaml` under `storage.traces`.

Example config (conf/storage/traces.yaml):
    type: bigquery
    project_id: my-project
    dataset_id: testing
    table_id: traces
    schema_path: traces.schema.json

The config is accessed via `bm.cfg.storage.traces` at initialization time.
If no traces storage is configured, TraceWriter will raise ConfigurationError.
"""

from typing import Optional

from buttermilk import bm, logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.exceptions import StorageConfigError
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
        """Lazy initialization - only initialize when first trace is added.

        Raises:
            StorageConfigError: If bm.cfg.storage.traces is not configured.
        """
        if self._initialized:
            return

        # Debug logging for configuration investigation
        cfg = getattr(bm, "cfg", None)
        logger.debug(
            "TraceWriter lazy initialization",
            has_bm_cfg=cfg is not None,
            bm_cfg_type=type(cfg).__name__,
        )

        # Look for traces storage configuration
        if cfg is None or not hasattr(cfg, "storage"):
            raise StorageConfigError(
<<<<<<< HEAD
                "No storage configuration available in bm.cfg. Ensure Hydra configuration is loaded with storage.traces defined."
=======
                "No storage configuration available in bm.cfg. "
                "Ensure Hydra configuration is loaded with storage.traces defined."
>>>>>>> origin/stable
            )

        storage_configs = cfg.storage
        logger.debug(
            "Storage configuration debug",
<<<<<<< HEAD
            storage_config_keys=list(storage_configs.keys()) if isinstance(storage_configs, dict) else None,
            has_traces_config="traces" in storage_configs if isinstance(storage_configs, dict) else hasattr(storage_configs, "traces"),
        )

        # Check for traces storage config
        traces_config = storage_configs.get("traces") if isinstance(storage_configs, dict) else getattr(storage_configs, "traces", None)
        if not traces_config:
            storage_keys = list(storage_configs.keys()) if isinstance(storage_configs, dict) else dir(storage_configs)
=======
            storage_config_keys=list(storage_configs.keys())
            if isinstance(storage_configs, dict)
            else None,
            has_traces_config="traces" in storage_configs
            if isinstance(storage_configs, dict)
            else hasattr(storage_configs, "traces"),
        )

        # Check for traces storage config
        traces_config = (
            storage_configs.get("traces")
            if isinstance(storage_configs, dict)
            else getattr(storage_configs, "traces", None)
        )
        if not traces_config:
            storage_keys = (
                list(storage_configs.keys())
                if isinstance(storage_configs, dict)
                else dir(storage_configs)
            )
>>>>>>> origin/stable
            raise StorageConfigError(
                f"No 'traces' storage configuration found in bm.cfg.storage. "
                f"Configure storage.traces in conf/config.yaml or conf/storage/traces.yaml. "
                f"Available storage configs: {storage_keys}"
            )

        logger.debug("Found traces config", traces_config=traces_config)
        # Create storage instance
        storage = bm.get_storage(traces_config)

        # Create uploader with reasonable defaults
        self.uploader = AsyncDataUploader(
            storage,
            buffer_size=100,  # Batch 100 traces before writing
            flush_interval=30,  # Or flush every 30 seconds
        )
        logger.info("TraceWriter initialized with traces storage")
        self._initialized = True

    async def add(self, trace: ExecutionTrace) -> None:
        """Add a trace to the upload queue.

        Args:
            trace: The ExecutionTrace to store

        Raises:
            StorageConfigError: If traces storage is not configured (on first call).
        """
        self._ensure_initialized()
        await self.uploader.add(trace)
        logger.debug(f"Trace {trace.call_id} queued for upload")

    async def flush(self) -> None:
        """Force flush any pending traces.

        Raises:
            StorageConfigError: If traces storage is not configured (on first call).
        """
        self._ensure_initialized()
        await self.uploader._flush()


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
