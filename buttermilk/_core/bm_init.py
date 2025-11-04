"""Buttermilk initialization and core resource management.

This module provides the main `BM` class, often used as a singleton instance
(conventionally named `bm`), which serves as the central access point for
all Buttermilk resources. These resources include Language Model (LLM) clients,
cloud provider connections (like Google Cloud Storage, BigQuery), secret
management, query execution, and overall configuration management for a
Buttermilk execution session.

The `BM` class handles the initialization of these components based on provided
configuration (typically loaded via Hydra) and offers a unified interface for
accessing them throughout the application. It also manages session-specific
information like run IDs and save directories.

Key functionalities:
-   Centralized access to configured LLM clients (`bm.llms`).
-   Management of cloud provider connections (`bm.gcs`, `bm.bq`).
-   Access to secrets via a configured secret provider (`bm.secret_manager`).
-   Execution of SQL queries (`bm.query_runner`).
-   Setup and management of logging, including optional cloud logging.
-   Handling of session information (`bm.session_info`) and standardized saving of artifacts.
-   Integration with Weave for tracing (`bm.get_weave_client()`).
"""

from __future__ import annotations  # Enable postponed annotations for type hinting

import asyncio
import datetime
import os  # For path expansion
import platform  # For system information like node name
from pathlib import Path
from tempfile import mkdtemp  # For creating temporary directories
from typing import Any

import psutil  # For system utilities like getting username
import pydantic  # Pydantic core
import shortuuid  # For generating short, unique IDs
from cloudpathlib import AnyPath, CloudPath  # For handling local and cloud paths
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator  # Pydantic components

from buttermilk._core.log import logger  # Centralized logger instance
from buttermilk._core.storage_config import BaseStorageConfig, StorageConfig  # Unified storage config
from buttermilk.utils import save  # Utility for saving data


def _make_session_id() -> str:
    """Generates a unique session ID for the current session.

    The ID is constructed using the current UTC timestamp, a short UUID,
    the machine's node name, and the current username. Each session gets
    its own unique identifier.

    Returns:
        str: A unique string identifier for this session.

    """
    node_name = platform.uname().node
    username = psutil.Process().username()
    # Strip domain from username if present (e.g., "DOMAIN\user" -> "user")
    username = str.split(username, "\\")[-1]

    # Format timestamp for use in filenames (simplified ISO 8601)
    session_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    session_id = f"session-{session_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    return session_id


class SessionInfo(BaseModel):
    """Simplified session information for observability and tracking.

    SessionInfo serves as the primary observability unit with a clean, simple design.
    Each session has a unique identifier and optional batch grouping for related tasks.

    This approach eliminates artificial complexity while providing proper session
    isolation and optional task grouping when needed.

    Note: project_name and job are stored at the root config level by default. We still
    store them here.

    Attributes:
        session_id (str): Unique identifier for this session.
        batch_id (str | None): Optional batch identifier for grouping related sessions.
        platform (str): Platform where the session is running.
        name (str): User-defined name for the current session or project.
        job (str): User-defined name for the specific job or task.
        ip (str | None): IP address of the machine running the session.
        node_name (str): Network name of the machine.
        save_dir (str | None): Primary directory for saving session outputs.
        flow_api (str | None): URL or identifier for a flow API, if applicable.

        # Observability fields
        status (str): Current session status.
        started_at (datetime | None): When the session started execution.
        completed_at (datetime | None): When the session completed.
        error_message (str | None): Error message if session failed.

        # Metrics
        records_processed (int): Number of records processed.
        outputs_generated (int): Number of outputs generated.

        # Configuration tracking
        agent_configs (dict): Agent configurations used in this session.
        flow_config (dict): Flow configuration for this session.

    """

    # Core identification
    session_id: str = Field(default_factory=_make_session_id, description="Unique identifier for this session.")
    batch_id: str | None = Field(default=None, description="Optional batch identifier for grouping related sessions.")

    # Basic session info
    platform: str = Field(default="local", description="Platform where the session is running.")
    project_name: str = Field(..., description="Project name for this session.")
    job: str = Field(..., description="User-defined name for the specific job or task.")

    # System information
    ip: str | None = Field(default=None, description="IP address of the machine, fetched asynchronously.")
    node_name: str = Field(default_factory=lambda: platform.uname().node, description="Network name of the machine.")
    save_dir: str | None = Field(default=None, description="Primary directory for saving session outputs.")
    cache_dir: str = Field(
        default_factory=lambda: os.path.expandvars(os.path.expanduser("~/.cache/buttermilk")), description="Directory for caching session data."
    )
    sessions_dir: str = Field(default="data/sessions", description="Directory for storing session data files.")
    flow_api: str | None = Field(default=None, description="URL or identifier for a flow API, if applicable.")

    # Enhanced observability fields
    status: str = Field(default="initializing", description="Current session status.")
    started_at: datetime.datetime | None = Field(default=None, description="When the session started execution.")
    completed_at: datetime.datetime | None = Field(default=None, description="When the session completed.")
    error_message: str | None = Field(default=None, description="Error message if session failed.")

    # Metrics
    records_processed: int = Field(default=0, description="Number of records processed.")
    outputs_generated: int = Field(default=0, description="Number of outputs generated.")

    # Configuration tracking
    agent_configs: dict[str, Any] = Field(default_factory=dict, description="Agent configurations used.")
    flow_config: dict[str, Any] = Field(default_factory=dict, description="Flow configuration for this session.")
    flow_hash: str | None = Field(default=None, description="Hash of flow configuration for A/B testing.")
    template_paths: list[str] = Field(default_factory=list, description="Paths to search for templates.")
    llm_wrapper: str = Field(
        default="autogen", description="Global LLM wrapper selection (autogen or litellm). Per-model use_litellm overrides this."
    )

    _get_ip_task: asyncio.Task[Any] | None = PrivateAttr(default=None)  # type: ignore

    def update_status(self, status: str, error_message: str | None = None) -> None:
        """Update the session status and timestamps.

        Args:
            status: New status for the session.
            error_message: Error message if status indicates failure.
        """
        old_status = self.status
        self.status = status

        current_time = datetime.datetime.now(datetime.UTC)

        # Update timestamps based on status
        if status in ["active", "running"] and self.started_at is None:
            self.started_at = current_time
        elif status in ["completed", "failed", "error", "terminated"] and self.completed_at is None:
            self.completed_at = current_time

        # Set error message if provided
        if error_message:
            self.error_message = error_message

        logger.info("Session status updated", session_id=self.session_id, old_status=old_status, new_status=status, batch_id=self.batch_id)

    def increment_records_processed(self, count: int = 1) -> None:
        """Increment the count of records processed.

        Args:
            count: Number of records to add to the count.
        """
        self.records_processed += count

    def increment_outputs_generated(self, count: int = 1) -> None:
        """Increment the count of outputs generated.

        Args:
            count: Number of outputs to add to the count.
        """
        self.outputs_generated += count

    def set_batch_context(self, batch_id: str | None = None) -> None:
        """Set the batch context identifier.

        Args:
            batch_id: ID of the batch this session belongs to (if any).
        """
        if batch_id is not None:
            self.batch_id = batch_id

    def get_session_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the session.

        Returns:
            Dict containing session summary information.
        """
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            duration = (datetime.datetime.now(datetime.UTC) - self.started_at).total_seconds()

        return {
            "session_id": self.session_id,
            "batch_id": self.batch_id,
            "project_name": self.project_name,
            "job": self.job,
            "status": self.status,
            "platform": self.platform,
            "timing": {
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "duration_seconds": duration,
            },
            "metrics": {
                "records_processed": self.records_processed,
                "outputs_generated": self.outputs_generated,
            },
            "system": {
                "node_name": self.node_name,
                "ip": self.ip,
                "save_dir": self.save_dir,
            },
            "error_message": self.error_message,
        }

    @field_validator("cache_dir", mode="after")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        """Expand user home directory and environment variables in cache_dir path."""
        return os.path.expandvars(os.path.expanduser(v))

    def get_cache_subdir(self, subdir: str, create: bool = True) -> Path:
        """Get a cache subdirectory path.

        This is the single method for accessing cache subdirectories.
        Use cache constants from buttermilk._core.constants for consistency.

        Args:
            subdir: Subdirectory name (use cache.CHROMADB, cache.ZOTERO, etc.)
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path: The cache subdirectory path

        Example:
            from buttermilk._core.constants import cache
            chromadb_path = bm.session_info.get_cache_subdir(cache.CHROMADB)
        """
        cache_path = Path(self.cache_dir) / subdir
        if create:
            cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_chromadb_cache_dir(self) -> Path:
        """Get the ChromaDB-specific cache directory.

        DEPRECATED: Use get_cache_subdir(cache.CHROMADB) instead.
        This method is kept for backward compatibility.

        Returns:
            Path: The ChromaDB cache directory within the session cache.
        """
        from buttermilk._core.constants import cache

        return self.get_cache_subdir(cache.CHROMADB)

    @staticmethod
    def generate_cache_key(path_or_identifier: str) -> str:
        """Generate a consistent cache key from any path or identifier.

        This is the single source of truth for cache key generation to ensure
        consistency across all caching operations (ChromaDB, embeddings, records, etc.).

        Args:
            path_or_identifier: Any path (e.g., "gs://bucket/path") or identifier to convert to cache key

        Returns:
            str: Cache key suitable for use as directory/file name
        """
        # Handle protocol separators first to avoid double underscores
        result = path_or_identifier.replace("://", "_")
        # Then handle remaining special characters
        result = result.replace("/", "_").replace(":", "_").replace(".", "_")
        return result

    class Config:
        """Pydantic model configuration for SessionInfo."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),  # Use ISO format for datetime
        }


class BM(BaseModel):
    """Session-scoped Buttermilk instance with simplified infrastructure sharing.

    Each session gets its own BM instance with session-specific state while sharing
    infrastructure resources (clouds, secrets, LLMs) through dependency injection.
    This eliminates complex hierarchy while providing proper session isolation.

    Attributes:
        session_info (SessionInfo): Session-specific information and metrics.
        save_dir_base (str): Base directory for this session's outputs.
        datasets (dict[str, BaseStorageConfig]): Session-specific dataset overrides.

    """

    # Session information
    session_info: SessionInfo = Field(..., description="Session information including session ID, batch ID, job name, etc.")

    # Session-specific configuration
    datasets: dict[str, BaseStorageConfig] = Field(
        default_factory=dict,
        description="Session-specific dataset configuration overrides.",
    )
    save_dir_base: str = Field(
        default_factory=mkdtemp,  # Creates a new temporary directory by default
        validate_default=True,
        description="Base directory for saving session-specific outputs.",
    )

    # Shared infrastructure - injected during creation
    _cloud_manager: Any = PrivateAttr(default=None)  # Will be injected
    _secret_manager: Any = PrivateAttr(default=None)  # Will be injected
    _llms_instance: Any = PrivateAttr(default=None)  # Will be injected
    _query_runner: Any = PrivateAttr(default=None)  # Will be injected
    _logger_cfg: Any = PrivateAttr(default=None)  # Will be injected from ExecutionContext
    _config: Any = PrivateAttr(default=None)  # Will store the full Hydra config

    # Session-specific state
    _initialization_complete: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _initialization_error: Exception | None = PrivateAttr(default=None)

    # OTEL context management
    _otel_baggage_token: Any = PrivateAttr(default=None)  # Token for cleaning up OTEL baggage

    # Allow attaching test doubles/mocks to instances (e.g., real_bm.get_storage = Mock(...))
    # This relaxes Pydantic's attribute setting restrictions for testing convenience.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @pydantic.field_validator("save_dir_base", mode="before")
    @classmethod
    def get_save_dir(cls, save_dir_base: Any) -> str:
        """Validates and normalizes the `save_dir_base` path.

        Converts `Path` or `CloudPath` objects to their string representations
        (POSIX path or URI).

        Args:
            save_dir_base: The input value for `save_dir_base`.

        Returns:
            str: The validated and normalized string representation of the path.

        Raises:
            ValueError: If `save_dir_base` is not a string, `Path`, or `CloudPath`.

        """
        if isinstance(save_dir_base, str):
            return save_dir_base
        if isinstance(save_dir_base, Path):
            return save_dir_base.as_posix()
        if isinstance(save_dir_base, CloudPath):
            return save_dir_base.as_uri()
        raise ValueError(
            f"save_dir_base must be a string, Path, or CloudPath, got {type(save_dir_base)}",
        )

    @pydantic.model_validator(mode="before")  # Changed to model_validator for Pydantic v2
    @classmethod
    def _remove_target(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Removes the `_target_` attribute commonly added by Hydra from input values.

        This is a pre-validation step to clean up configuration data before
        it's parsed by Pydantic.

        Args:
            values: The dictionary of raw input values for the model.

        Returns:
            dict[str, Any]: The `values` dictionary with `_target_` removed, if present.

        """
        values.pop("_target_", None)  # Remove if exists, do nothing otherwise
        return values

    def __init__(
        self,
        logger_cfg: Any = None,
        cloud_manager: Any = None,
        secret_manager: Any = None,
        llms_instance: Any = None,
        query_runner: Any = None,
        **data: Any,
    ) -> None:
        """Initializes the BM instance with minimal field assignment.

        This is a lightweight constructor that only sets fields. All I/O operations
        and setup logic happen in the async _async_init() method.

        Args:
            logger_cfg: Logger configuration for cloud logging (optional).
            cloud_manager: Shared cloud manager instance (optional).
            secret_manager: Shared secret manager instance (optional).
            llms_instance: Shared LLMs instance (optional).
            query_runner: Shared query runner instance (optional).
            **data: Keyword arguments representing the BM-specific configuration fields.

        """
        super().__init__(**data)

        # Inject shared infrastructure - just field assignment
        self._logger_cfg = logger_cfg
        self._cloud_manager = cloud_manager
        self._secret_manager = secret_manager
        self._llms_instance = llms_instance
        self._query_runner = query_runner
        self._initialization_error = None

    async def _async_init(self) -> None:
        """Async initialization of session-specific setup tasks.

        This method performs all I/O operations and setup logic that should happen
        asynchronously after the BM instance is created.

        IMPORTANT: This initializes the complete session context including:
        - Python contextvars (session_id_var, batch_id_var)
        - OTEL baggage (for span attribute propagation)
        - OTEL root span context (detached from any parent)

        Each new BM instance creates a fresh, isolated observability context.
        """
        try:
            # UNIFIED SESSION CONTEXT SETUP
            # This must happen FIRST to establish observability context
            # for all subsequent operations (logging, tracing, etc.)
            self._setup_session_context()

            # Set up session-specific logging context
            self._setup_session_logging()

            # Finalize save directory
            self._finalize_save_dir()

            # Save initial session config
            self._save_initial_config()

            # Start async IP fetch task if event loop is running
            self.start_fetch_ip_task()

            self._initialization_complete.set()
            logger.info(
                "Session initialized successfully",
                session_id=self.session_info.session_id,
                batch_id=self.session_info.batch_id,
                save_dir=self.session_info.save_dir,
            )
        except Exception as e:
            logger.error(f"Error during session initialization: {e}")
            self._initialization_error = e
            self._initialization_complete.set()

    def _setup_session_context(self) -> None:
        """Establish unified session context for observability.

        This method initializes ALL session context in one place:
        1. Python contextvars (session_id_var, batch_id_var) for logging/app logic
        2. OTEL baggage (buttermilk.session.id, etc.) for span propagation
        3. Detaches from any parent OTEL context to ensure root span creation

        This ensures that each BM session creates an independent observability context,
        preventing trace nesting when multiple jobs run in the same worker process.
        """
        from buttermilk._core.context import set_logging_context
        from buttermilk.utils.otel import attach_session_baggage

        session_id = self.session_info.session_id
        batch_id = self.session_info.batch_id
        project_name = self.session_info.project_name

        # 1. Set Python contextvars (for logging and application logic)
        set_logging_context(
            session_id=session_id,
            batch_id=batch_id,
        )

        # 2. Attach OTEL baggage (propagates to all child spans and logs)
        # Store the baggage token so it can be cleaned up later
        self._otel_baggage_token = attach_session_baggage(
            session_id=session_id,
            extra={
                "buttermilk.project": project_name,
                "buttermilk.batch.id": batch_id,
            },
        )

        logger.debug(
            "Session context established (contextvars + OTEL baggage)",
            session_id=session_id,
            batch_id=batch_id,
            project_name=project_name,
        )

    def _setup_session_logging(self) -> None:
        """Sets up simplified session-specific logging context."""
        from buttermilk._core.context import set_logging_context

        # Set simplified logging context for this session
        set_logging_context(
            session_id=self.session_info.session_id,
            batch_id=self.session_info.batch_id,
            agent_id=None,  # Will be set by agents when needed
        )

        # Set up cloud logging if configured and cloud manager is available
        if self._logger_cfg and self._cloud_manager:
            try:
                from buttermilk._core.log import setup_cloud_logging

                setup_cloud_logging(self._logger_cfg, self._cloud_manager, self.session_info)
                logger.info("Cloud logging configured for session", session_id=self.session_info.session_id, logger_type=self._logger_cfg.type)
            except Exception as e:
                logger.warning("Failed to setup cloud logging for session", session_id=self.session_info.session_id, error=str(e))

        # Set up structlog context variables for automatic injection into all log messages
        # This ensures all subsequent log messages include session context
        import structlog

        structlog.contextvars.clear_contextvars()  # Clear any previous context
        structlog.contextvars.bind_contextvars(
            session_id=self.session_info.session_id,
            batch_id=self.session_info.batch_id,
            platform=self.session_info.platform,
            project_name=self.session_info.project_name,
            job=self.session_info.job,
        )

        logger.info(
            "Session logging context established",
            session_id=self.session_info.session_id,
            batch_id=self.session_info.batch_id,
            platform=self.session_info.platform,
            project_name=self.session_info.project_name,
            job=self.session_info.job,
        )

    def _finalize_save_dir(self) -> None:
        """Construct the final save_dir path for this session.

        Constructs the full save directory path and stores it in session_info.save_dir.
        """
        # Construct full save directory path using session_id for uniqueness
        save_dir_path = AnyPath(self.save_dir_base) / self.session_info.project_name / self.session_info.job / self.session_info.session_id
        self.session_info.save_dir = str(save_dir_path)
        logger.debug(f"Finalized session save_dir: {self.session_info.save_dir}")

    async def ensure_initialized(self) -> None:
        """Ensure that session initialization is complete.

        Raises:
            RuntimeError: If initialization failed with an error
        """
        # Ensure session initialization is complete
        await self._initialization_complete.wait()
        if self._initialization_error:
            raise RuntimeError(f"Session initialization failed: {self._initialization_error}") from self._initialization_error
        logger.debug("Session initialization verified complete")

    async def cleanup(self) -> None:
        """Clean up session resources including OTEL baggage.

        Call this when the session is complete to ensure proper cleanup of:
        - OTEL baggage context
        - Any other session-specific resources
        """
        from buttermilk.utils.otel import detach_session_baggage

        # Detach OTEL baggage if it was attached
        if self._otel_baggage_token is not None:
            detach_session_baggage(self._otel_baggage_token)
            self._otel_baggage_token = None
            logger.debug("Cleaned up OTEL baggage for session", session_id=self.session_info.session_id)

    def _save_initial_config(self) -> None:
        """Save the initial BM configuration to disk including the full .cfg object.

        Saves three top-level keys:
        - cfg: The complete Hydra configuration (infrastructure, agents, pipelines, etc.)
        - bm: BM instance state (includes session_info nested within)
        - session_info: Direct reference to session_info for convenience (also in bm.session_info)

        Note: session_info appears both at top level and nested in bm for ease of access.
        """

        # Convert the entire .cfg object to a plain dict for saving
        cfg_dict = None
        if self._config is not None:
            cfg_dict = self._config.model_dump()

        # Data to save: full config object
        # Note: In typical usage, cfg contains the Hydra configuration tree,
        # which is separate from the BM instance state and session_info
        config_data_to_save = {
            "cfg": cfg_dict,
            "bm": self.model_dump(exclude_none=True),
            "session_info": self.session_info.model_dump(exclude_none=True),
        }

        self.save(
            data=config_data_to_save,
            basename="initial_bm_config",
            extension=".json",
        )
        logger.debug("Initial BM config saved successfully")

    # Permit overriding/attaching attributes (e.g., monkeypatching methods) in tests
    def __setattr__(self, name: str, value: Any) -> None:  # type: ignore[override]
        try:
            return super().__setattr__(name, value)
        except ValueError:
            # Fallback to plain setattr for non-field attributes (e.g., method monkeypatch)
            object.__setattr__(self, name, value)

    @property
    def cloud_manager(self) -> Any:
        """Provides access to the CloudManager instance."""
        if self._cloud_manager is None:
            raise RuntimeError("CloudManager not available. Ensure infrastructure is properly injected.")
        return self._cloud_manager

    @property
    def secret_manager(self) -> Any:
        """Provides access to the SecretsManager instance."""
        if self._secret_manager is None:
            raise RuntimeError("SecretsManager not available. Ensure infrastructure is properly injected.")
        return self._secret_manager

    @property
    def llms(self) -> Any:
        """Provides access to the LLMs manager instance."""
        if self._llms_instance is None:
            raise RuntimeError("LLMs instance not available. Ensure infrastructure is properly injected.")
        return self._llms_instance

    @property
    def query_runner(self) -> Any:
        """Provides access to the QueryRunner instance."""
        if self._query_runner is None:
            raise RuntimeError("QueryRunner not available. Ensure infrastructure is properly injected.")
        return self._query_runner

    @property
    def gcp_credentials(self) -> Any:
        """Provides access to GCP credentials."""
        return self.cloud_manager.gcp_credentials

    def get_gcp_access_token(self) -> str:
        """Get a valid GCP access token."""
        return self.cloud_manager.get_access_token()

    @property
    def gcs(self) -> Any:
        """Provides access to the GCS client."""
        return self.cloud_manager.gcs

    @property
    def bq(self) -> Any:
        """Provides access to the BigQuery client."""
        return self.cloud_manager.bq

    @property
    def genai(self) -> Any:
        """Provides access to the GenAI client."""
        return self.cloud_manager.genai

    @property
    def pubsub(self) -> Any:
        """Provides access to complete Pub/Sub configuration including project_id."""
        if self._cloud_manager is None:
            raise RuntimeError("CloudManager not available. Ensure infrastructure is properly injected.")

        gcp_config = self._cloud_manager.gcp_cloud_cfg
        if not gcp_config:
            raise RuntimeError("No GCP cloud configuration found for Pub/Sub access.")

        if not gcp_config.pubsub:
            raise RuntimeError("No Pub/Sub configuration found in GCP cloud config. Ensure pubsub is configured in your cloud configuration.")

        # Return the actual PubSubServiceConfig object
        return gcp_config.pubsub

    async def get_weave_client(self) -> None:
        """Legacy method - weave has been removed.

        This method previously provided access to the Weave client via ExecutionContext.
        After weave removal, it always returns None.

        Returns:
            None: Weave is no longer used
        """
        logger.debug("get_weave_client called but weave has been removed, returning None")

    @property
    def credentials(self) -> dict[str, str]:
        """Provides access to shared system credentials.

        Returns an empty dict if no secret manager is configured, allowing the system
        to function in local-only mode without cloud credentials.
        """
        # Check if secret manager is available
        if self._secret_manager is None:
            logger.debug("No secret manager configured, returning only environment variables as credentials.")
            return os.environ.copy()  # Return environment variables as fallback

        try:
            return self.secret_manager.get_secret(cfg_key="credentials_secret")
        except Exception as e:
            logger.warning(f"Failed to fetch credentials from secret manager: {e}. Returning only environment variables as credentials.")
            return os.environ.copy()  # Return environment variables as fallback

    @property
    def cfg(self) -> Any:
        """Provides access to the instantiated Hydra configuration."""
        return self._config

    @property
    def logger(self) -> Any:
        """Returns a contextualized logger with session information."""
        from buttermilk import logger as base_logger

        return base_logger.bind(session_id=self.session_info.session_id, project=self.session_info.project_name, job=self.session_info.job)

    def start_fetch_ip_task(self) -> None:
        """Starts an asynchronous task to fetch the machine's external IP address.

        The IP address is stored in `self.session_info.ip` upon completion. This task is
        initiated if an event loop is running and the task hasn't been started already.
        """
        from buttermilk.utils import get_ip  # Utility function to get IP

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Start task only if it hasn't been started or is already done
                if not hasattr(self, "_get_ip_task") or self._get_ip_task is None or self._get_ip_task.done():

                    async def _fetch_and_set_ip() -> None:
                        ip = await get_ip()
                        self.session_info.ip = ip
                        logger.debug(f"Fetched IP address: {ip}")

                    self._get_ip_task = asyncio.create_task(_fetch_and_set_ip())
            # else: No event loop running, cannot start async task
        except RuntimeError:  # No current event loop
            logger.debug("No running event loop, skipping async IP fetch task.")

    def save(
        self,
        data: Any,
        save_dir: str | AnyPath | None = None,
        extension: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Saves provided data to a file with standardized naming and location.

        The actual saving logic is delegated to `buttermilk.utils.save.save`.
        If `save_dir` is not provided, `self.save_dir` (the session's default
        save directory) is used. If that's also not set, a temporary directory
        is created.

        Args:
            data: The data to be saved (e.g., dict, list, string).
            save_dir: Optional directory to save the file in. Defaults to
                `self.save_dir`.
            extension: Optional file extension (e.g., ".json", ".txt").
                Defaults to ".json".
            **kwargs: Additional keyword arguments to pass to the underlying
                `buttermilk.utils.save.save` function.

        Returns:
            str | None: The full path to the saved file as a string, or `None`
            if the save operation failed.

        """
        effective_save_dir_str: str
        if save_dir:
            effective_save_dir_str = str(save_dir)
        elif self.session_info.save_dir:
            effective_save_dir_str = self.session_info.save_dir
        else:
            # Fallback to a temporary directory if no save_dir is configured
            effective_save_dir_str = mkdtemp()
            logger.warning(f"No save_dir specified or configured in BM; using temporary directory: {effective_save_dir_str}")

        # Ensure extension starts with a dot if provided, otherwise default to .json
        effective_extension = extension or ".json"
        if not effective_extension.startswith("."):
            effective_extension = "." + effective_extension

        try:
            # Call the utility save function
            saved_file_path = save.save(
                data=data,
                save_dir=AnyPath(effective_save_dir_str),  # Convert to AnyPath for utility
                extension=effective_extension,
                **kwargs,
            )
            logger.debug(
                f"Successfully saved data to: {saved_file_path}",
                uri=str(saved_file_path),  # Ensure URI is a string
            )
            return str(saved_file_path)  # Return path as string
        except Exception as e:
            logger.error(f"Failed to save data to '{effective_save_dir_str}' with extension '{effective_extension}': {e!s}")
            return None  # Indicate save failure

    def run_query(  # noqa: PLR0913
        self,
        sql: str,
        destination: str | None = None,
        overwrite: bool = False,
        do_not_return_results: bool = False,
        save_to_gcs: bool = False,
        return_df: bool = True,
    ) -> Any:  # Return type can be pd.DataFrame, None, or other based on params
        """Runs a BigQuery SQL query using the configured `QueryRunner`.

        This method is a convenience wrapper that delegates to
        `self.query_runner.run_query`. Refer to `QueryRunner.run_query`
        for detailed documentation of parameters and behavior. The `save_dir`
        for any local saves will be `self.save_dir`.

        Args:
            sql (str): The SQL query to execute.
            destination (str | None): Optional BigQuery table ID (project.dataset.table)
                to save query results to. Defaults to None.
            overwrite (bool): If True, overwrites the destination table if it exists.
                Defaults to False.
            do_not_return_results (bool): If True, does not attempt to fetch results
                into memory (e.g., if results are large and saved to a table).
                Defaults to False.
            save_to_gcs (bool): If True, saves results to Google Cloud Storage
                instead of a BigQuery table (destination might specify GCS path).
                Defaults to False.
            return_df (bool): If True (and `do_not_return_results` is False),
                returns results as a Pandas DataFrame. Defaults to True.

        Returns:
            Any: Typically a Pandas DataFrame if `return_df` is True and results
            are fetched. Can be None or other types depending on parameters.

        """
        return self.query_runner.run_query(
            sql=sql,
            destination=destination,
            overwrite=overwrite,
            do_not_return_results=do_not_return_results,
            save_to_gcs=save_to_gcs,
            save_dir=self.session_info.save_dir,  # Pass BM's default save directory
            return_df=return_df,
        )

    def get_storage(self, config: StorageConfig | dict | DictConfig | None = None) -> Any:
        """Factory method to create unified storage instances.

        Creates the appropriate storage class based on the configuration type,
        using this BM instance for client access and default configurations.

        Note: For ChromaDB with remote storage, you must call ensure_cache_initialized()
        before accessing the collection. Consider using get_storage_async() for auto-initialization.

        Args:
            config: Storage configuration (StorageConfig object, dict, or None)

        Returns:
            Storage instance (BigQueryStorage, FileStorage, etc.)

        Raises:
            ValueError: If storage type is not supported

        """
        # Ensure config is a StorageConfig object
        if config is None:
            raise ValueError("Storage configuration is required")

        # Use the storage factory to create the appropriate storage instance
        from buttermilk._core.storage_config import StorageFactory  # noqa import here to avoid loop

        return StorageFactory.create_storage(config)

    async def get_storage_async(self, config: BaseStorageConfig | dict | None = None) -> Any:
        """Async factory method that creates and auto-initializes storage instances.

        For ChromaDB with remote storage (gs://, s3://, etc.), this automatically calls
        ensure_cache_initialized() so the storage is ready for immediate use.

        Args:
            config: Storage configuration (BaseStorageConfig object, dict, or None)

        Returns:
            Fully initialized storage instance ready for use

        Raises:
            ValueError: If storage type is not supported

        Example:
            # Auto-initialized ChromaDB (recommended)
            vectorstore = await bm.get_storage_async(cfg.storage.osb_vector)
            count = vectorstore.collection.count()  # ✅ Works immediately

            # Compare with manual approach:
            vectorstore = bm.get_storage(cfg.storage.osb_vector)
            await vectorstore.ensure_cache_initialized()  # Extra step
            count = vectorstore.collection.count()  # ✅ Works after init

        """
        # Create storage instance using sync method
        storage = self.get_storage(config)

        # Auto-initialize if it's ChromaDB with remote storage
        if hasattr(storage, "ensure_cache_initialized"):
            # Check if it's remote storage requiring initialization
            if hasattr(storage, "persist_directory") and storage.persist_directory:
                if storage.persist_directory.startswith(("gs://", "gcs://", "s3://", "azure://")):
                    logger.info(f"🔄 Auto-initializing remote storage: {storage.persist_directory}")
                    await storage.ensure_cache_initialized()
                    logger.info("✅ Storage ready for use")

        return storage

    def get_bigquery_storage(self, dataset_name: str, **kwargs: Any) -> Any:
        """Convenience method to create BigQuery storage with dataset name.

        Args:
            dataset_name: The dataset name for the storage
            **kwargs: Additional configuration overrides. Must include dataset_id and table_id.

        Returns:
            BigQueryStorage instance

        Raises:
            ValueError: If required BigQuery table components are missing

        """
        config_data = {
            "type": "bigquery",
            "dataset_name": dataset_name,
            **kwargs,
        }
        config = StorageConfig(**config_data)
        return self.get_storage(config)

    async def graceful_shutdown(self, timeout: float = 10.0) -> None:
        """Perform graceful shutdown of all async operations.

        This method waits for background tasks to complete, flushes buffers,
        and ensures traces/logs are uploaded before the process exits.

        Args:
            timeout: Maximum time to wait for shutdown operations (seconds)
        """
        logger.info(f"Starting graceful shutdown (timeout={timeout}s)...")

        # Collect all pending tasks
        all_tasks = []

        # Get all running tasks except the current one
        current_task = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task != current_task and not task.done():
                all_tasks.append(task)

        if all_tasks:
            logger.info(f"Waiting for {len(all_tasks)} background tasks to complete...")

            # Wait for tasks with timeout
            try:
                await asyncio.wait_for(asyncio.gather(*all_tasks, return_exceptions=True), timeout=timeout)
                logger.info("All background tasks completed successfully")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for tasks after {timeout}s")
                # Cancel remaining tasks
                for task in all_tasks:
                    if not task.done():
                        task.cancel()
                # Wait briefly for cancellation to complete
                await asyncio.gather(*all_tasks, return_exceptions=True)

        # Additional delay for trace/log uploads that might not be tracked as tasks
        logger.info("Waiting for traces and logs to upload...")
        await asyncio.sleep(2.0)

        logger.info("Graceful shutdown complete")


# Factory functions for creating session-scoped BM instances


async def create_session_bm_async(  # noqa: PLR0913
    project_name: str,
    job: str,
    batch_id: str | None = None,
    platform: str = "local",
    save_dir_base: str | None = None,
    template_paths: list[str] | None = None,
    cloud_manager: Any = None,
    secret_manager: Any = None,
    llms_instance: Any = None,
    query_runner: Any = None,
    logger_cfg: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> BM:
    """Create a new session-scoped BM instance with async initialization.

    This is the primary async factory method for creating BM instances.

    Args:
        name: User-defined name for the current session or project.
        job: User-defined name for the specific job or task.
        batch_id: Optional batch identifier for grouping related sessions.
        platform: Platform where the session is running.
        save_dir_base: Base directory for session outputs.
        template_paths: Optional list of paths to search for templates.
        cloud_manager: Shared cloud manager instance (optional).
        secret_manager: Shared secret manager instance (optional).
        llms_instance: Shared LLMs instance (optional).
        query_runner: Shared query runner instance (optional).
        logger_cfg: Logger configuration for cloud logging (optional).
        config: Full Hydra configuration to store on BM instance (optional).
        **kwargs: Additional arguments for SessionInfo.

    Returns:
        BM: A fully initialized session-scoped BM instance.
    """
    # Create session info
    session_info_data = {
        "project_name": project_name,
        "job": job,
        "platform": platform,
        "batch_id": batch_id,
        "template_paths": template_paths or [],
        **kwargs,
    }

    # Create SessionInfo instance to get auto-generated session_id
    session_info = SessionInfo(**session_info_data)

    # Use provided query_runner or create one if cloud_manager is available
    if query_runner is None and cloud_manager is not None:
        from buttermilk._core.query import QueryRunner

        query_runner = QueryRunner(bq_client=cloud_manager.bq)

    # Create BM instance with all dependencies passed to constructor
    bm_data = {
        "session_info": session_info,
        "logger_cfg": logger_cfg,
        "cloud_manager": cloud_manager,
        "secret_manager": secret_manager,
        "llms_instance": llms_instance,
        "query_runner": query_runner,
    }

    if save_dir_base is not None:
        bm_data["save_dir_base"] = save_dir_base

    bm = BM(**bm_data)

    # Store config on BM instance if provided
    # Note: Config is stored as-is, not instantiated, because some objects
    # (like pipeline) may have circular dependencies on BM existing first.
    # Users should call hydra.utils.instantiate() on specific parts when needed.
    if config is not None:
        bm._config = config

    # Perform async initialization
    await bm._async_init()

    return bm
