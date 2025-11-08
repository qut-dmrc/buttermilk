"""Execution Context for managing process-level infrastructure.

This module provides the ExecutionContext class, which manages shared infrastructure
across multiple sessions within the same process. This includes:
- Cloud provider connections (GCS, BigQuery)
- Secret management
- Logging infrastructure
- LLM connections
- Process-level configuration

The ExecutionContext replaces the global singleton pattern for infrastructure
while allowing session-specific BM instances to be created.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import platform
from pathlib import Path
from typing import Any

import psutil
import shortuuid
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk._core.cloud import CloudManager
from buttermilk._core.cloud_config import CloudProvider
from buttermilk._core.config import LoggerConfig, Tracing
from buttermilk._core.constants import (
    CONFIG_CACHE_FILENAME,
    MODELS_CFG_KEY,
    SHARED_CREDENTIALS_KEY,
    cache,
    get_base_cache_dir,
)
from buttermilk._core.keys import SecretsManager
from buttermilk._core.llms import LLMs
from buttermilk._core.log import logger, setup_console_logging, setup_file_logging
from buttermilk._core.query import QueryRunner
from buttermilk._core.storage_config import BaseStorageConfig
from buttermilk.utils.utils import load_json_flexi

# Global variable to store the execution context ID
_global_execution_context_id = ""


def _make_execution_context_id() -> str:
    """Generates a unique execution context ID for the current process.

    The ID is constructed using the current UTC timestamp, a short UUID,
    the machine's node name, and the current username. This represents
    the infrastructure context that spans multiple sessions.

    Returns:
        str: A unique string identifier for this execution context.
    """
    global _global_execution_context_id
    if _global_execution_context_id:
        return _global_execution_context_id

    node_name = platform.uname().node
    username = psutil.Process().username()
    username = str.split(username, "\\")[-1]  # Strip domain if present

    # Format timestamp for use in filenames (simplified ISO 8601)
    context_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    execution_context_id = (
        f"exec-{context_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    )
    _global_execution_context_id = execution_context_id
    return execution_context_id


class ExecutionContext(BaseModel):
    """Process-level infrastructure context shared across sessions.

    ExecutionContext manages the shared infrastructure that was previously
    handled by the global BM singleton. This includes cloud connections,
    secrets, LLMs, and logging setup. Multiple sessions can share the same
    ExecutionContext while maintaining their own session-specific state.

    Attributes:
        execution_context_id (str): Unique identifier for this execution context.
        clouds (list[CloudProvider]): List of cloud provider configurations.
        logging (LoggerConfig | None): Global logging configuration.
        tracing (dict[str, Tracing] | None): Global tracing configurations.
        datasets (dict[str, BaseStorageConfig]): Shared dataset configurations.
    """

    execution_context_id: str = Field(
        default_factory=_make_execution_context_id,
        description="Unique identifier for this execution context.",
    )

    # Project management
    project_name: str | None = Field(
        default=None,
        description="Project name shared across all sessions in this execution context.",
    )

    # Infrastructure configuration
    clouds: list[CloudProvider] = Field(
        default_factory=list, description="List of cloud provider configurations."
    )
    logging: LoggerConfig | None = Field(
        default=None, description="Configuration for cloud-based logging."
    )
    tracing: dict[str, Tracing] | None = Field(
        default_factory=dict, description="Configuration for tracing systems."
    )
    datasets: dict[str, BaseStorageConfig] = Field(
        default_factory=dict, description="Shared dataset configurations."
    )
    default_llm_wrapper: str = Field(
        default="autogen",
        description="Default LLM wrapper type (autogen or litellm). Passed to LLMs instance.",
    )

    # Private attributes for lazy-loaded infrastructure
    _cloud_manager: CloudManager | None = PrivateAttr(default=None)
    _secret_manager: SecretsManager | None = PrivateAttr(default=None)
    _llms_instance: LLMs | None = PrivateAttr(default=None)
    _query_runner: QueryRunner | None = PrivateAttr(default=None)
    _credentials_cached: dict[str, str] | None = PrivateAttr(default=None)
    _initialization_complete: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _initialization_error: Exception | None = PrivateAttr(default=None)
    _tracing_instrumented: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _tracing_providers_initialized: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any) -> None:
        """Initialize the ExecutionContext with minimal field assignment.

        This is a lightweight constructor. All I/O and setup logic happens
        in the async _async_init() method.
        """
        super().__init__(**data)
        self._initialization_error = None

    async def _async_init(self) -> None:
        """Async initialization of ExecutionContext infrastructure.

        This method performs all I/O operations and setup logic that should happen
        asynchronously after the ExecutionContext instance is created.
        """
        # Set up logging early
        self._setup_logging()

        # Set GCP environment variables immediately
        self._setup_gcp_environment()

        # Initialize infrastructure asynchronously
        await self._async_background_init()

        # Check for secrets cloud using service-aware pattern
        secrets_cloud = self._find_cloud_with_service("secrets")
        secret_provider_type = secrets_cloud.type if secrets_cloud else None

        logger.info(
            "Initialized ExecutionContext",
            execution_context_id=self.execution_context_id,
            cloud_providers=len(self.clouds),
            secret_provider=secret_provider_type,
        )

    def _setup_logging(self) -> None:
        """Set up modern logging for the execution context."""
        verbose = getattr(self.logging, "verbose", False) if self.logging else False
        enable_console = (
            getattr(self.logging, "console", True) if self.logging else True
        )
        setup_console_logging(verbose=verbose, enable_console=enable_console)

        # Set up structured JSON file logging with project name
        log_files = setup_file_logging(
            execution_context_id=self.execution_context_id,
            verbose=verbose,
            project_name=self.project_name,
        )
        for log_file in log_files:
            logger.info("ExecutionContext logging enabled", log_file=log_file)

        # Log initialization message
        logger.info(
            "ExecutionContext logging initialized",
            execution_context_id=self.execution_context_id,
            project_name=self.project_name,
        )

    def _setup_gcp_environment(self) -> None:
        """Set up GCP environment variables for early access."""
        if not self.clouds:
            return

        gcp_cloud_cfg = next(
            (c for c in self.clouds if c and hasattr(c, "type") and c.type == "gcp"),
            None,
        )

        if gcp_cloud_cfg:
            project_id = getattr(gcp_cloud_cfg, "project_id", None)
            quota_project_id = getattr(gcp_cloud_cfg, "quota_project_id", project_id)

            if project_id:
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            if quota_project_id:
                os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = quota_project_id

            logger.debug(
                f"Set GCP environment: GOOGLE_CLOUD_PROJECT={project_id}, "
                f"GOOGLE_CLOUD_QUOTA_PROJECT={quota_project_id}"
            )

    async def _async_background_init(self) -> None:
        """Async initialization of infrastructure components."""
        try:
            # Initialize cloud manager
            if self.clouds:
                logger.debug("Performing cloud authentication...")
                _ = self.cloud_manager

            # Initialize secret manager if secrets cloud is available
            if self._find_cloud_with_service("secrets"):
                logger.debug("Initializing secret manager...")
                _ = self.secret_manager

            logger.info(
                "ExecutionContext initialization completed",
                execution_context_id=self.execution_context_id,
            )
            self._initialization_complete.set()
        except Exception as e:
            logger.error("Error during ExecutionContext initialization", error=str(e))
            self._initialization_error = e
            self._initialization_complete.set()

    async def ensure_initialized(self) -> None:
        """Ensure that ExecutionContext initialization is complete."""
        await self._initialization_complete.wait()
        if self._initialization_error:
            raise RuntimeError(
                f"ExecutionContext initialization failed: {self._initialization_error}"
            ) from self._initialization_error

        # Set up async components like tracing
        await self._setup_tracing()

        logger.debug("ExecutionContext initialization verified complete")

    def validate_and_set_project(self, project: str | None) -> str:
        """Validate and set the project name for this execution context.

        Args:
            project: Project name to validate and set. If None and no project is set,
                    raises an error. If None and project is already set, returns existing.

        Returns:
            The validated project name.

        Raises:
            RuntimeError: If project validation fails or required project is missing.
        """
        if self.project_name is None:
            # First session - project is required
            if project is None:
                raise RuntimeError(
                    "project parameter is required for the first session in an execution context. Example: init(job='my_job', project='my_project')"
                )
            self.project_name = project
            logger.debug("Set project name for execution context", project=project)
            return project
        else:
            # Subsequent sessions - validate consistency
            if project is not None and project != self.project_name:
                raise RuntimeError(
                    f"Project name mismatch: execution context is using project '{self.project_name}', "
                    f"but session specified project '{project}'. All sessions in the same execution "
                    f"context must use the same project. Either omit the project parameter to use "
                    f"'{self.project_name}', or start a new process for project '{project}'."
                )
            # Return the existing project (whether user specified it or not)
            return self.project_name

    @property
    def cloud_manager(self) -> CloudManager:
        """Provides access to the CloudManager instance."""
        if self._cloud_manager is None:
            self._cloud_manager = CloudManager(clouds=self.clouds)
            self._ensure_cloud_authentication()
        return self._cloud_manager

    def _ensure_cloud_authentication(self) -> None:
        """Ensure cloud providers are authenticated."""
        if self._cloud_manager:
            logger.debug("Performing lazy cloud authentication...")
            self._cloud_manager.login_clouds()

            # Note: Cloud logging is now set up at the BM session level
            # This ensures proper session context and avoids mock session objects

            logger.debug("Cloud authentication completed")

    def _find_cloud_with_service(self, service: str) -> Any | None:
        """Find the first cloud provider that has a specific service configured.

        Args:
            service: Service name to look for (e.g., "secrets", "logging", "pubsub", "tracing")

        Returns:
            Cloud provider configuration with the service, or None if not found.
        """
        for cloud in self.clouds:
            if hasattr(cloud, "has_service") and cloud.has_service(service):
                return cloud
        return None

    @property
    def secret_manager(self) -> SecretsManager:
        """Provides access to the SecretsManager instance."""
        if self._secret_manager is None:
            # Use service-aware cloud provider pattern
            secrets_cloud = self._find_cloud_with_service("secrets")
            if not secrets_cloud:
                raise RuntimeError("No cloud provider with secrets service configured.")

            from buttermilk._core.keys import SecretsManager

            # Get secrets configuration from cloud provider
            secrets_config = secrets_cloud.get_client_config("secretmanager")
            self._secret_manager = SecretsManager(**secrets_config)
            logger.debug(
                f"SecretsManager initialized with {secrets_cloud.type} provider"
            )
        return self._secret_manager

    @property
    def llms(self) -> LLMs:
        """Provides access to the LLMs manager instance."""
        if self._llms_instance is None:
            connections_data: dict[str, Any] | None = None
            # Use centralized cache directory
            cache_dir = get_base_cache_dir() / cache.MODELS
            cache_path = cache_dir / CONFIG_CACHE_FILENAME

            # Try to load from local cache first
            if cache_path.exists() and cache_path.is_file():
                try:
                    connections_data = load_json_flexi(
                        cache_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(connections_data, dict):
                        logger.warning(
                            f"LLM connections cache at {cache_path} is not a dict. Will try secrets."
                        )
                        connections_data = None
                    else:
                        logger.debug(
                            "Loaded LLM connections from cache",
                            cache_path=str(cache_path),
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to load LLM connections from cache, will try secrets",
                        error=str(e),
                    )
                    connections_data = None

            # If not loaded from cache, try secret manager (if available)
            if connections_data is None:
                secrets_cloud = self._find_cloud_with_service("secrets")
                if not secrets_cloud:
                    logger.warning(
                        "No secret manager configured and no cached LLM connections found. "
                        "LLM functionality will be limited. To enable LLMs, either: "
                        f"1) Configure a cloud provider with secrets service, or "
                        f"2) Provide a cached models config at {cache_path}"
                    )
                    connections_data = {}
                else:
                    try:
                        connections_data = self.secret_manager.get_secret(
                            cfg_key=MODELS_CFG_KEY
                        )
                        if not isinstance(connections_data, dict):
                            raise TypeError(
                                f"LLM connections from secrets is not a dict, got {type(connections_data)}."
                            )
                        logger.debug(
                            "Loaded LLM connections from secret manager",
                            key=MODELS_CFG_KEY,
                        )

                        # Cache the connections data
                        self._write_cache_sync(connections_data, cache_path)
                    except Exception as e:
                        logger.error(
                            "Failed to load LLM connections from secret manager",
                            error=str(e),
                            secret_key=MODELS_CFG_KEY,
                        )
                        logger.warning(
                            "Proceeding with empty LLM connections. LLM functionality will not be available."
                        )
                        connections_data = {}

            self._llms_instance = LLMs(
                connections=connections_data, default_wrapper=self.default_llm_wrapper
            )
        return self._llms_instance

    def _write_cache_sync(
        self, connections_data: dict[str, Any], cache_path: Path
    ) -> None:
        """Synchronous cache writing helper."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        logger.debug("Caching LLM connections", cache_path=str(cache_path))
        cache_path.write_text(json.dumps(connections_data), encoding="utf-8")

    @property
    def query_runner(self) -> QueryRunner:
        """Provides access to the QueryRunner instance."""
        if self._query_runner is None:
            self._query_runner = QueryRunner(bq_client=self.bq)
        return self._query_runner

    @property
    def gcp_credentials(self) -> Any:
        """Provides access to GCP credentials."""
        return self.cloud_manager.gcp_credentials

    def get_gcp_access_token(self) -> str:
        """Get a valid GCP access token, refreshing if needed."""
        return self.cloud_manager.get_access_token()

    @property
    def gcs(self) -> Any:
        """Provides access to the Google Cloud Storage client."""
        return self.cloud_manager.gcs

    @property
    def bq(self) -> Any:
        """Provides access to the Google BigQuery client."""
        return self.cloud_manager.bq

    @property
    def genai(self) -> Any:
        """Provides access to the Google GenAI client."""
        return self.cloud_manager.genai

    @property
    def credentials(self) -> dict[str, str]:
        """Retrieves shared system credentials from the secret manager.

        Returns an empty dict if no secret manager is configured, allowing the system
        to function in local-only mode without cloud credentials.
        """
        if self._credentials_cached is None:
            # Check if secret manager is available
            secrets_cloud = self._find_cloud_with_service("secrets")
            if not secrets_cloud:
                logger.debug(
                    "No secret manager configured, returning empty credentials dict"
                )
                self._credentials_cached = {}
                return self._credentials_cached

            logger.debug("Fetching shared credentials from secret manager...")
            try:
                creds = self.secret_manager.get_secret(cfg_key=SHARED_CREDENTIALS_KEY)
                if not isinstance(creds, dict):
                    raise TypeError(
                        f"Expected shared credentials to be a dict, got {type(creds)}"
                    )
                self._credentials_cached = creds
            except Exception as e:
                logger.warning(
                    f"Failed to fetch credentials from secret manager: {e}. Using empty credentials dict."
                )
                self._credentials_cached = {}
        return self._credentials_cached

    async def _setup_tracing(self) -> None:
        """Set up tracing based on configuration.

        All tracing setup is deferred to avoid circular dependencies during
        ExecutionContext initialization. Tracing will be initialized on-demand
        when first accessed.
        """
        # Log configured tracing providers but defer actual initialization
        enabled_providers = []
        # Weave support has been removed
        if self.tracing.get("traceloop") and self.tracing["traceloop"].enabled:
            enabled_providers.append("traceloop")
        if self.tracing.get("otel") and self.tracing["otel"].enabled:
            enabled_providers.append("otel")

        if enabled_providers:
            logger.debug(
                "Tracing providers configured (deferred initialization)",
                providers=enabled_providers,
            )
        else:
            logger.debug("No tracing providers enabled")

        self._tracing_instrumented.set()

    async def get_weave_client(self) -> None:
        """Legacy method - weave has been removed.

        This method previously provided access to the Weave client.
        After weave removal, it always returns None.

        Returns:
            None: Weave is no longer used
        """
        logger.debug(
            "get_weave_client called but weave has been removed, returning None"
        )

    async def _ensure_tracing_initialized(self) -> None:
        """Ensure all tracing providers are initialized on-demand."""
        if not self._tracing_instrumented.is_set():
            await self._setup_tracing()

        # Now perform actual tracing initialization for all enabled providers
        await self._initialize_all_tracing_providers()

    async def _initialize_all_tracing_providers(self) -> None:
        """Initialize all configured tracing providers."""
        # Skip if already initialized to prevent duplicate setup
        if self._tracing_providers_initialized:
            return

        # Weave support has been removed

        # Initialize Traceloop if enabled
        if self.tracing.get("traceloop") and self.tracing["traceloop"].enabled:
            await self._initialize_traceloop()

        # Initialize OTEL if enabled (now safe since BM singleton should be available)
        if self.tracing.get("otel") and self.tracing["otel"].enabled:
            await self._initialize_otel()

        # Mark as initialized
        self._tracing_providers_initialized = True
        logger.debug("All tracing providers initialization completed")

    async def _initialize_weave(self) -> None:
        """Legacy method - weave has been removed.

        This method previously initialized Weave tracing. After weave removal,
        it does nothing and logs a debug message.
        """
        logger.debug(
            "_initialize_weave called but weave has been removed, doing nothing"
        )

    async def _initialize_traceloop(self) -> None:
        """Initialize Traceloop tracing."""
        traceloop_config = self.tracing["traceloop"]
        api_key = getattr(traceloop_config, "api_key", None)

        if not api_key:
            raise RuntimeError(
                "Traceloop tracing enabled but api_key not configured. Add api_key to infrastructure.tracing.traceloop in config."
            )

        try:
            from traceloop.sdk import Traceloop

            Traceloop.init(app_name="buttermilk", api_key=api_key)
            logger.info(
                "Traceloop initialized successfully",
                execution_context_id=self.execution_context_id,
            )
        except Exception as e:
            logger.error("Failed to initialize Traceloop tracing", error=str(e))
            raise RuntimeError(f"Traceloop tracing initialization failed: {e}") from e

    async def _initialize_otel(self) -> None:
        """Initialize OTEL tracing using ExecutionContext's infrastructure."""
        try:
            from buttermilk.utils.otel import setup_tracing_otel_with_execution_context

            setup_tracing_otel_with_execution_context(self.tracing["otel"], self)
            logger.info(
                "OTEL Tracing has been set up successfully",
                execution_context_id=self.execution_context_id,
            )
        except Exception as e:
            logger.error("Failed to initialize OTEL tracing", error=str(e))
            raise RuntimeError(f"OTEL tracing initialization failed: {e}") from e


# Global execution context instance
_global_execution_context: ExecutionContext | None = None
_execution_context_initialized: bool = False


def get_execution_context() -> ExecutionContext:
    """Get the global ExecutionContext instance."""
    global _global_execution_context
    if _global_execution_context is None:
        raise RuntimeError(
            "ExecutionContext not initialized. Call set_execution_context() first."
        )
    return _global_execution_context


def set_execution_context(context: ExecutionContext) -> None:
    """Set the global ExecutionContext instance."""
    global _global_execution_context, _execution_context_initialized
    _global_execution_context = context
    _execution_context_initialized = True


async def create_execution_context_async(**kwargs) -> ExecutionContext:
    """Create and set a new ExecutionContext with async initialization.

    This is the primary async factory method for creating ExecutionContext instances.

    Raises:
        RuntimeError: If an ExecutionContext has already been initialized.
                     This prevents accidental reinitialization that would
                     break logging configuration and lose execution context state.
    """
    global _execution_context_initialized

    if _execution_context_initialized:
        raise RuntimeError(
            "ExecutionContext has already been initialized. "
            "Creating multiple ExecutionContext instances will break logging configuration, "
            "reset verbose logging settings, and cause loss of execution context state. "
            "Use get_execution_context() to access the existing context, or "
            "get_or_create_execution_context() for safe initialization."
        )

    context = ExecutionContext(**kwargs)
    await context._async_init()
    set_execution_context(context)
    _execution_context_initialized = True
    return context


def create_execution_context(**kwargs) -> ExecutionContext:
    """Sync wrapper for create_execution_context_async - DEPRECATED.

    This is a lightweight sync wrapper that exists for backward compatibility.
    New code should use create_execution_context_async() directly.

    Raises:
        RuntimeError: If an ExecutionContext has already been initialized.
                     This prevents accidental reinitialization that would
                     break logging configuration and lose execution context state.
    """
    return asyncio.run(create_execution_context_async(**kwargs))


async def get_or_create_execution_context_async(**kwargs) -> ExecutionContext:
    """Get existing ExecutionContext or create a new one if none exists (async).

    This is the safe async way to initialize ExecutionContext that won't break
    if called multiple times.

    Args:
        **kwargs: Arguments passed to ExecutionContext constructor if creating new

    Returns:
        ExecutionContext: The existing or newly created ExecutionContext
    """
    global _execution_context_initialized

    if _execution_context_initialized:
        return get_execution_context()

    return await create_execution_context_async(**kwargs)


def get_or_create_execution_context(**kwargs) -> ExecutionContext:
    """Sync wrapper for get_or_create_execution_context_async - DEPRECATED.

    This is a lightweight sync wrapper that exists for backward compatibility.
    New code should use get_or_create_execution_context_async() directly.

    Args:
        **kwargs: Arguments passed to ExecutionContext constructor if creating new

    Returns:
        ExecutionContext: The existing or newly created ExecutionContext
    """
    return asyncio.run(get_or_create_execution_context_async(**kwargs))


# Factory methods for creating ExecutionContext from typed config


async def from_config_async(
    infrastructure,
    project_name: str | None = None,
    default_llm_wrapper: str = "autogen",
):
    """Create ExecutionContext from typed infrastructure config.

    This is the recommended factory method that takes the typed
    InfrastructureConfig from ButtermilkConfig.

    Args:
        infrastructure: Typed InfrastructureConfig from ButtermilkConfig
        project_name: Optional project name
        default_llm_wrapper: Default LLM wrapper type (autogen or litellm). Defaults to "autogen" for backward compatibility.

    Returns:
        Initialized ExecutionContext ready for creating sessions

    Example:
        >>> from buttermilk._core.config_bootstrap import load_typed_config_async
        >>> from buttermilk._core.execution_context import from_config_async
        >>>
        >>> typed_cfg = await load_typed_config_async()
        >>> ctx = await from_config_async(
        ...     typed_cfg.infrastructure,
        ...     project_name="my_project",
        ...     default_llm_wrapper=typed_cfg.session.llm_wrapper
        ... )
    """
    # Convert TracingConfig to dict format if needed
    tracing_dict = infrastructure.tracing
    if hasattr(tracing_dict, "model_dump"):
        # It's a Pydantic model, convert to dict
        tracing_dict = tracing_dict.model_dump()

    # Convert LoggerConfig to dict format if needed
    logging_config = infrastructure.logging
    if logging_config and hasattr(logging_config, "model_dump"):
        # It's a Pydantic model, convert to dict
        logging_config = logging_config.model_dump()

    # Extract components from typed config
    context = await get_or_create_execution_context_async(
        project_name=project_name,
        clouds=infrastructure.clouds,
        logging=logging_config,
        tracing=tracing_dict,
        datasets=infrastructure.datasets,
        default_llm_wrapper=default_llm_wrapper,
    )

    await context.ensure_initialized()
    return context


async def create_session_from_context_async(
    execution_context: ExecutionContext,
    session,
    storage_configs: dict | None = None,
    full_config=None,
):
    """Create a new session-scoped BM instance from ExecutionContext.

    This allows creating multiple sessions within the same ExecutionContext,
    perfect for API servers where each request gets its own session.

    Args:
        execution_context: The ExecutionContext to use for infrastructure
        session: Session configuration (SessionInfo from ButtermilkConfig)
        storage_configs: Optional storage configs from ButtermilkConfig.storage
        full_config: Optional full ButtermilkConfig to store on BM._config

    Returns:
        Fully initialized BM instance with infrastructure from this context

    Example:
        >>> # Application startup (once)
        >>> ctx = await from_config_async(typed_cfg.infrastructure)
        >>>
        >>> # Per request (many times, shared infrastructure)
        >>> bm1 = await create_session_from_context_async(ctx, typed_cfg.session)
        >>> bm2 = await create_session_from_context_async(ctx, typed_cfg.session)
    """
    from buttermilk._core.bm_init import create_session_bm_async

    # Validate project consistency
    validated_project = execution_context.validate_and_set_project(session.project_name)

    # Create BM with injected infrastructure
    bm = await create_session_bm_async(
        project_name=validated_project,
        job=session.job,
        batch_id=session.batch_id,
        platform=session.platform,
        template_paths=session.template_paths,
        cloud_manager=execution_context.cloud_manager
        if execution_context.clouds
        else None,
        secret_manager=execution_context.secret_manager
        if execution_context._find_cloud_with_service("secrets")
        else None,
        llms_instance=execution_context.llms,
        query_runner=execution_context.query_runner
        if execution_context.clouds
        else None,
        logger_cfg=execution_context.logging,
        config=full_config,  # Store full typed config
    )

    # Attach storage configs if provided
    if storage_configs:
        bm.datasets = storage_configs

    await bm.ensure_initialized()
    return bm
