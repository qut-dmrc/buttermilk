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
-   Handling of session information (`bm.run_info`) and standardized saving of artifacts.
-   Integration with Weave for tracing (`bm.weave`).
"""

from __future__ import annotations  # Enable postponed annotations for type hinting

import asyncio
import datetime
import logging
import platform  # For system information like node name
from pathlib import Path
from tempfile import mkdtemp  # For creating temporary directories
from typing import Any

import psutil  # For system utilities like getting username
import pydantic  # Pydantic core
import shortuuid  # For generating short, unique IDs
from cloudpathlib import AnyPath, CloudPath  # For handling local and cloud paths
from pydantic import BaseModel, Field, PrivateAttr  # Pydantic components
from rich import print  # For rich console output

from buttermilk._core.cloud import CloudManager  # Manages cloud provider connections
from buttermilk._core.config import CloudProviderCfg, LoggerConfig, Tracing  # Config models
from buttermilk._core.keys import SecretsManager  # Manages secrets
from buttermilk._core.storage_config import BaseStorageConfig  # Storage config models

try:
    from buttermilk._core.llms import LLMs  # Manages LLM clients
except ImportError:
    LLMs = None
from buttermilk._core.log import ContextFilter, logger  # Centralized logger instance

try:
    from buttermilk._core.query import QueryRunner  # For running SQL queries
except ImportError:
    QueryRunner = None
from buttermilk._core.utils.lazy_loading import cached_property  # Utility for lazy loading

try:
    from buttermilk.utils import save  # Utility for saving data
except ImportError:
    save = None
from buttermilk._core.storage_config import StorageConfig, StorageFactory  # Unified storage config

# Constants for configuration keys
CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
"""Path to the cache file for LLM model configurations."""
_MODELS_CFG_KEY = "models_secret"
"""Key used to retrieve LLM model configurations from the secret manager."""
_SHARED_CREDENTIALS_KEY = "credentials_secret"
"""Key used to retrieve shared system credentials from the secret manager."""

# Global variable to store the run ID, ensuring it's generated once per execution.
_global_run_id = ""


def _make_run_id() -> str:
    """Generates a unique run ID for the current execution session.

    The ID is constructed using the current UTC timestamp, a short UUID,
    the machine's node name, and the current username. This aims to create
    a globally unique and informative identifier for each run.
    If a global run ID has already been generated for the current session,
    it returns the existing one.

    Returns:
        str: A unique string identifier for this execution run.

    """
    global _global_run_id
    if _global_run_id:  # Return existing ID if already generated
        return _global_run_id

    node_name = platform.uname().node
    username = psutil.Process().username()
    # Strip domain from username if present (e.g., "DOMAIN\user" -> "user")
    username = str.split(username, "\\")[-1]

    # Format timestamp for use in filenames (simplified ISO 8601)
    run_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    _global_run_id = run_id  # Cache the generated ID globally
    return run_id


class SessionInfo(BaseModel):
    """Pydantic model holding information about the current execution session.

    This data is often used for logging, organizing outputs, and tracking runs.

    Attributes:
        platform (str): The platform where the session is running (e.g., "local",
            "gcp_vm", "azure_container"). Defaults to "local".
        name (str): A user-defined name for the current session or project.
        job (str): A user-defined name for the specific job or task being run.
        run_id (str): A unique identifier for this specific execution run.
            Defaults to a value generated by `_make_run_id()`.
        ip (str | None): The IP address of the machine running the session.
            Fetched asynchronously and may be None initially.
        node_name (str): The network name of the machine. Defaults to
            `platform.uname().node`.
        save_dir (str | None): The primary directory path where outputs for this
            session should be saved.
        flow_api (str | None): URL or identifier for a flow API, if applicable.
        _get_ip_task (asyncio.Task | None): Private attribute for the asyncio task
            that fetches the IP address.
        _ip (str | None): Private attribute storing the fetched IP address.
        Config (class): Pydantic model configuration.
            - `arbitrary_types_allowed`: True.
            - `json_encoders`: Custom JSON encoder for `datetime.datetime`.

    """

    platform: str = Field(description="Platform where the session is running (e.g., 'local', 'gcp').")
    name: str = Field(..., description="User-defined name for the current session or project.")
    job: str = Field(..., description="User-defined name for the specific job or task.")
    run_id: str = Field(default_factory=_make_run_id, description="Unique identifier for this execution run.")
    ip: str | None = Field(default=None, description="IP address of the machine, fetched asynchronously.")
    node_name: str = Field(default_factory=lambda: platform.uname().node, description="Network name of the machine.")
    save_dir: str | None = Field(default=None, description="Primary directory for saving session outputs.")
    flow_api: str | None = Field(default=None, description="URL or identifier for a flow API, if applicable.")

    _get_ip_task: asyncio.Task[Any] | None = PrivateAttr(default=None)  # type: ignore
    _ip: str | None = PrivateAttr(default=None)

    class Config:
        """Pydantic model configuration for SessionInfo."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),  # Use ISO format for datetime
        }


class BM(SessionInfo):
    """Central singleton-like class for Buttermilk, providing access to all resources.

    `BM` (often instantiated as `bm`) serves as the primary gateway to Buttermilk's
    major components, including cloud clients (GCS, BigQuery), LLM connections,
    configuration settings, and secret management. It handles the initialization
    of these components based on a provided configuration (typically loaded by Hydra)
    and offers a unified, simplified interface for accessing them from anywhere
    in the application code.

    It inherits from `SessionInfo` to also carry context about the current
    execution session.

    Typical Usage:
    ```python
    from buttermilk._core.dmrc import get_bm # Function to get/create the BM instance

    bm = get_bm() # Get the initialized BM instance

    # Access cloud storage
    bm.gcs.upload_from_filename(...)

    # Interact with an LLM
    response = bm.llms.my_chat_model.create(messages=[...])

    # Access secrets
    api_key = bm.secret_manager.get_secret("my_api_key_name")
    ```

    Attributes:
        connections (list[str]): List of connection names (e.g., for LLMs, databases).
            (Purpose might need further clarification based on usage).
        secret_provider (CloudProviderCfg | None): Configuration for the secret
            provider (e.g., GCP Secret Manager, Azure Key Vault).
        logger_cfg (CloudProviderCfg | None): Configuration for cloud-based logging
            (e.g., GCP Logging).
        pubsub (CloudProviderCfg | None): Configuration for a Pub/Sub system, if used.
        clouds (list[CloudProviderCfg]): List of configurations for different cloud
            providers to be initialized (e.g., GCP, Azure).
        tracing (Tracing | None): Configuration for tracing (e.g., Langfuse, Weave).
        datasets (dict[str, StorageConfig]): A dictionary of predefined data source
            configurations accessible via the `BM` instance.
        save_dir_base (str): The base directory under which session-specific save
            directories will be created. Defaults to a new temporary directory.
        _cloud_manager (CloudManager | None): Private attribute for the `CloudManager` instance.
        _secret_manager (SecretsManager | None): Private attribute for the `SecretsManager` instance.
        _llms_instance (LLMs | None): Private attribute for the `LLMs` manager instance.
        _query_runner (QueryRunner | None): Private attribute for the `QueryRunner` instance.
        _credentials_cached (dict[str, str] | None): Private cache for shared system credentials.

    """

    connections: list[str] = Field(
        default_factory=list,
        description="List of connection names (purpose may vary depending on context, e.g., active LLM connections).",
    )
    secret_provider: CloudProviderCfg | None = Field(
        default=None,
        description="Configuration for the secret provider (e.g., GCP Secret Manager, Azure Key Vault).",
    )
    logger_cfg: LoggerConfig | None = Field(
        default=None,
        description="Configuration for cloud-based logging (e.g., GCP Logging).",
    )
    pubsub: CloudProviderCfg | None = Field(
        default=None,
        description="Configuration for a Publish/Subscribe system, if used.",
    )
    clouds: list[CloudProviderCfg] = Field(
        default_factory=list,
        description="List of configurations for different cloud providers to initialize (e.g., GCP, Azure).",
    )
    tracing: Tracing | None = Field(
        default_factory=Tracing,  # Default to Tracing() which might have enabled=False
        description="Configuration for tracing system integration (e.g., Langfuse, Weave).",
    )
    datasets: dict[str, BaseStorageConfig] = Field(
        default_factory=dict,
        description="Dictionary of predefined storage configurations.",
    )
    save_dir_base: str = Field(
        default_factory=mkdtemp,  # Creates a new temporary directory by default
        validate_default=True,
        description="Base directory for saving session-specific outputs. Defaults to a new temporary directory.",
    )

    _cloud_manager: CloudManager | None = PrivateAttr(default=None)
    _secret_manager: SecretsManager | None = PrivateAttr(default=None)
    _llms_instance: LLMs | None = PrivateAttr(default=None)
    _query_runner: QueryRunner | None = PrivateAttr(default=None)
    _credentials_cached: dict[str, str] | None = PrivateAttr(default=None)
    _initialization_complete: asyncio.Event = PrivateAttr(default=None)
    _initialization_error: Exception | None = PrivateAttr(default=None)

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

    # @pydantic.model_validator(mode="before")  # Changed to model_validator for Pydantic v2
    # @classmethod
    # def _remove_target(cls, values: dict[str, Any]) -> dict[str, Any]:
    #     """Removes the `_target_` attribute commonly added by Hydra from input values.

    #     This is a pre-validation step to clean up configuration data before
    #     it's parsed by Pydantic.

    #     Args:
    #         values: The dictionary of raw input values for the model.

    #     Returns:
    #         dict[str, Any]: The `values` dictionary with `_target_` removed, if present.

    #     """
    #     values.pop("_target_", None)  # Remove if exists, do nothing otherwise
    #     return values

    def __init__(self, **data: Any) -> None:
        """Initializes the BM instance with provided configuration data.

        After standard Pydantic model initialization, it calls `_post_init_setup`
        to perform further setup tasks like logging, directory creation, and
        cloud logins.

        Args:
            **data: Keyword arguments representing the configuration fields for
                `BM` and its parent `SessionInfo`.

        """
        super().__init__(**data)
        self._initialization_complete = asyncio.Event()
        self._initialization_error: Exception | None = None
        self._post_init_setup()

    def _post_init_setup(self) -> None:
        """Performs setup tasks immediately after Pydantic model initialization.

        This includes:
        - Constructing the full `save_dir` path based on `save_dir_base` and session info.
        - Setting up logging (console and potentially cloud logging).
        - Saving the initial configuration to a JSON file in `save_dir`.
        - Starting an asynchronous task to fetch the machine's IP address.
        - Logging into configured cloud providers.

        Note: Logger configuration validation is now handled by Pydantic model validators
        in CloudProviderCfg, providing early validation with better error messages.
        """
        # Construct full save directory path
        save_dir_path = AnyPath(self.save_dir_base) / self.name / self.job / self.run_id
        self.save_dir = str(save_dir_path)  # Store as string

        self.setup_logging(verbose=getattr(self.logger_cfg, "verbose", False) if self.logger_cfg else False)

        # Set GCP environment variables immediately (needed for GCS access)
        self._setup_gcp_environment()

        # Print current config to console - immediate for user feedback
        print("Initialized Buttermilk (bm) with configuration:")  # Use rich print
        print(self.model_dump(exclude_none=True))  # Exclude None for cleaner output

        # Defer non-critical operations to background tasks for faster startup
        self._schedule_background_init()

    def _setup_gcp_environment(self) -> None:
        """Set up GCP environment variables immediately for early GCS access.

        This extracts the environment variable setup from CloudManager
        to ensure they're available before any cloud operations.
        """
        import os

        if not self.clouds:
            return

        # Find GCP cloud config
        gcp_cloud_cfg = next(
            (c for c in self.clouds if c and hasattr(c, "type") and c.type == "gcp"),
            None,
        )

        if gcp_cloud_cfg:
            # Get project_id from config
            project_id = getattr(gcp_cloud_cfg, "project_id", None)
            quota_project_id = getattr(gcp_cloud_cfg, "quota_project_id", project_id)

            if project_id:
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

            if quota_project_id:
                os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = quota_project_id

            logger.debug(f"Set GCP environment: GOOGLE_CLOUD_PROJECT={project_id}, GOOGLE_CLOUD_QUOTA_PROJECT={quota_project_id}")

    def _schedule_background_init(self) -> None:
        """Schedule non-critical initialization tasks in the background.

        This method defers operations that aren't immediately needed for core functionality:
        - Config file saving
        - IP address fetching
        - Cloud provider authentication

        These operations happen asynchronously to improve startup performance.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._background_init())
            else:
                # Event loop exists but not running, run synchronously
                logger.debug("Event loop not running, performing synchronous initialization")
                self._sync_background_init()
        except RuntimeError:
            # No current event loop, run synchronously
            logger.debug("No event loop available, performing synchronous initialization")
            self._sync_background_init()

    async def _background_init(self) -> None:
        """Perform non-critical initialization operations in the background."""
        try:
            # Critical tasks that must complete before work begins
            # 1. Ensure cloud authentication happens early
            if hasattr(self, "_cloud_manager") or self.clouds:
                logger.debug("Performing early cloud authentication...")
                _ = self.cloud_manager  # Trigger lazy initialization and authentication

            # 2. Initialize secret manager to start caching secrets
            if self.secret_provider:
                logger.debug("Initializing secret manager...")
                _ = self.secret_manager  # Trigger lazy initialization

            # 3. Save initial config (non-critical, but do it anyway)
            await asyncio.get_event_loop().run_in_executor(None, self._save_initial_config)

            # 4. Start IP fetching task (non-critical)
            self.start_fetch_ip_task()

            logger.info("Background initialization completed successfully")
            self._initialization_complete.set()
        except Exception as e:
            logger.error(f"Error during background initialization: {e}")
            self._initialization_error = e
            self._initialization_complete.set()  # Set even on error so waiters don't hang

    def _sync_background_init(self) -> None:
        """Fallback synchronous version of background initialization."""
        try:
            # Critical synchronous initialization
            if hasattr(self, "_cloud_manager") or self.clouds:
                logger.debug("Performing synchronous cloud authentication...")
                _ = self.cloud_manager  # Trigger initialization

            if self.secret_provider:
                logger.debug("Initializing secret manager synchronously...")
                _ = self.secret_manager  # Trigger initialization

            self._save_initial_config()
            logger.info("Synchronous background initialization completed")
            # For sync path, mark as complete immediately
            self._initialization_complete.set()
        except Exception as e:
            logger.error(f"Error during synchronous background initialization: {e}")
            self._initialization_error = e
            self._initialization_complete.set()

    async def ensure_initialized(self) -> None:
        """Ensure that BM initialization is complete before proceeding.

        This method should be called before any operations that depend on:
        - Cloud authentication being complete
        - Secrets being available
        - Save directory being properly configured

        Raises:
            RuntimeError: If initialization failed with an error

        """
        await self._initialization_complete.wait()
        if self._initialization_error:
            raise RuntimeError(f"BM initialization failed: {self._initialization_error}") from self._initialization_error
        logger.debug("BM initialization verified complete")

    def _save_initial_config(self) -> None:
        """Save the initial BM configuration to disk."""
        try:
            if self.save_dir:
                # Data to save: BM config and run_info
                config_data_to_save = [
                    self.model_dump(exclude_none=True),  # Current BM instance config
                    self.run_info.model_dump(exclude_none=True),  # Current run_info
                ]
                self.save(  # Use the instance's save method
                    data=config_data_to_save,
                    basename="initial_bm_config",  # More descriptive basename
                    extension=".json",
                    # save_dir is implicitly self.save_dir if not provided to self.save
                )
                logger.debug("Initial BM config saved successfully")
            else:  # Should not happen if save_dir_base defaults to mkdtemp
                logger.warning("BM.save_dir is not set. Skipping saving initial config.")
        except Exception as e:
            logger.error(f"Could not save initial BM config to default save directory: {e!s}")

    @cached_property
    def cloud_manager(self) -> CloudManager:
        """Provides access to the `CloudManager` instance.

        The `CloudManager` handles interactions with various configured cloud
        providers (e.g., GCP, Azure). It's instantiated on first access and
        performs lazy authentication to improve startup performance.

        Returns:
            CloudManager: The initialized `CloudManager` instance.

        """
        if self._cloud_manager is None:
            self._cloud_manager = CloudManager(clouds=self.clouds)
            # Perform cloud login and tracing setup on first access
            self._ensure_cloud_authentication()
        return self._cloud_manager

    def _ensure_cloud_authentication(self) -> None:
        """Ensure cloud providers are authenticated and tracing is set up.

        This method is called lazily when the cloud_manager is first accessed,
        rather than during __post_init__, to improve startup performance.
        """
        try:
            if self._cloud_manager:
                logger.debug("Performing lazy cloud authentication...")
                self._cloud_manager.login_clouds()  # Perform logins

                # Set up tracing if configured and enabled
                if self.tracing and self.tracing.enabled:
                    self._cloud_manager.setup_tracing(self.tracing)

                # Set up cloud logging now that cloud manager is authenticated
                self._setup_cloud_logging()

                logger.debug("Cloud authentication completed")
        except Exception as e:
            logger.warning(f"Error during cloud authentication: {e}")

    def _setup_cloud_logging(self) -> None:
        """Set up Google Cloud Logging after cloud authentication."""
        if self.logger_cfg and self.logger_cfg.type == "gcp" and self._cloud_manager:
            try:
                from google.cloud import logging as gcp_logging
                from google.cloud.logging_v2.handlers import CloudLoggingHandler

                cloud_logging_resource = gcp_logging.Resource(
                    type="generic_task",
                    labels={
                        "project": self.logger_cfg.project,
                        "location": self.logger_cfg.location,
                        "namespace": self.name,
                        "job": self.job,
                        "task_id": self.run_id,
                    },
                )

                cloudHandler = CloudLoggingHandler(
                    client=self._cloud_manager.gcs_log_client(self.logger_cfg),
                    resource=cloud_logging_resource,
                    name=self.name,
                    labels=self.model_dump(include={"run_id", "name", "job", "platform"}),
                )
                cloudHandler.setLevel(logging.INFO)
                logger.addHandler(cloudHandler)
                logger.debug("Cloud logging handler added")
            except Exception as e:
                # Provide better error messages distinguishing between config and service issues
                logger.error(
                    f"Cloud logging setup failed due to configuration issue: {e}. "
                    f"Logger config: type={self.logger_cfg.type}, "
                    f"project={self.logger_cfg.project}, "
                    f"location={self.logger_cfg.location}",
                )

    @cached_property
    def secret_manager(self) -> SecretsManager:
        """Provides access to the `SecretsManager` instance.

        The `SecretsManager` is responsible for retrieving secrets (like API keys)
        from a configured backend (e.g., GCP Secret Manager, Azure Key Vault).
        It's instantiated on first access using `self.secret_provider` config.

        Returns:
            SecretsManager: The initialized `SecretsManager` instance.

        Raises:
            RuntimeError: If `secret_provider` configuration is missing.

        """
        if self._secret_manager is None:
            if not self.secret_provider:
                raise RuntimeError("BM.secret_provider configuration is missing, cannot initialize SecretsManager.")
            # Pass the model directly, SecretsManager will handle unpacking if needed
            self._secret_manager = SecretsManager(**self.secret_provider.model_dump())
        return self._secret_manager

    @cached_property
    def llms(self) -> LLMs:
        """Provides access to the `LLMs` manager instance.

        The `LLMs` manager handles configurations and clients for different
        Language Models. It attempts to load LLM connection configurations first
        from a local cache (`CONFIG_CACHE_PATH`), then from the secret manager
        if the cache is not found or fails to load. Loaded configurations are
        cached locally for subsequent runs if fetched from secrets.

        Returns:
            LLMs: The initialized `LLMs` manager instance.

        Raises:
            RuntimeError: If loading LLM connections from both cache and secrets fails.
            TypeError: If the loaded LLM connections data is not a dictionary.

        """
        if self._llms_instance is None:
            connections_data: dict[str, Any] | None = None
            cache_path = Path(CONFIG_CACHE_PATH)

            # Try to load from local cache file first
            if cache_path.exists() and cache_path.is_file():
                try:
                    from buttermilk.utils.utils import load_json_flexi

                    connections_data = load_json_flexi(cache_path.read_text(encoding="utf-8"))
                    if not isinstance(connections_data, dict):  # Validate type from cache
                        logger.warning(f"LLM connections cache at {cache_path} is not a dict, found {type(connections_data)}. Will try secrets.")
                        connections_data = None
                    else:
                        logger.info(f"Loaded LLM connections from cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to load LLM connections from cache {cache_path}: {e!s}. Will try secrets.")
                    connections_data = None  # Ensure it's None if cache load fails

            # If not loaded from cache, get from secret manager
            if connections_data is None:
                try:
                    connections_data = self.secret_manager.get_secret(cfg_key=_MODELS_CFG_KEY)
                    if not isinstance(connections_data, dict):  # Validate type from secrets
                        raise TypeError(f"LLM connections from secrets is not a dict, got {type(connections_data)}.")
                    logger.info(f"Loaded LLM connections from secret manager (key: '{_MODELS_CFG_KEY}').")

                    # Defer cache writing to background task - Phase 2 optimization
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self._cache_llm_connections_async(connections_data, cache_path))
                        else:
                            # If no event loop is running, cache synchronously as fallback
                            logger.info("No async loop running, caching LLM connections synchronously")
                            self._write_cache_sync(connections_data, cache_path)
                    except RuntimeError:
                        # No event loop available, cache synchronously
                        logger.info("No event loop available, caching LLM connections synchronously")
                        self._write_cache_sync(connections_data, cache_path)
                except Exception as e:
                    logger.error(f"Failed to load LLM connections from secret manager: {e!s}")
                    raise RuntimeError("Failed to load LLM connections from both cache and secrets.") from e

            self._llms_instance = LLMs(connections=connections_data)
        return self._llms_instance

    async def _cache_llm_connections_async(self, connections_data: dict[str, Any], cache_path: Path) -> None:
        """Asynchronously cache LLM connections to avoid blocking startup - Phase 2 optimization."""
        try:
            # Run file operations in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_cache_sync, connections_data, cache_path)
            logger.info(f"Cached LLM connections to: {cache_path}")
        except Exception as e:
            logger.warning(f"Could not cache LLM connections after fetching from secrets: {e!s}")

    def _write_cache_sync(self, connections_data: dict[str, Any], cache_path: Path) -> None:
        """Synchronous cache writing helper for thread pool execution."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        cache_path.write_text(json.dumps(connections_data), encoding="utf-8")

    @cached_property
    def query_runner(self) -> QueryRunner:
        """Provides access to the `QueryRunner` instance.

        The `QueryRunner` is used for executing SQL queries, primarily against
        Google BigQuery, using the `bm.bq` client. Instantiated on first access.

        Returns:
            QueryRunner: The initialized `QueryRunner` instance.

        """
        if self._query_runner is None:
            self._query_runner = QueryRunner(bq_client=self.bq)  # Delegates bq client access
        return self._query_runner

    @property
    def run_info(self) -> SessionInfo:
        """Provides a `SessionInfo` object representing the current execution session.

        This is a snapshot of the session-specific details managed by the `BM` instance.

        Returns:
            SessionInfo: An object containing current session information.

        """
        # Ensure _ip is fetched if the task has completed
        fetched_ip = self._ip
        if self._get_ip_task and self._get_ip_task.done():
            try:
                fetched_ip = self._get_ip_task.result()
            except Exception:  # Catch potential exceptions from the task
                logger.warning("Failed to get IP address from async task result.")

        return SessionInfo(
            platform=self.platform,
            name=self.name,
            job=self.job,
            run_id=self.run_id,  # run_id is from BM instance itself
            node_name=self.node_name,
            ip=fetched_ip,  # Use potentially updated IP
            save_dir=self.save_dir,
            flow_api=self.flow_api,
        )

    @property
    def gcp_credentials(self) -> Any:  # Type hint could be more specific if known (e.g., google.auth.credentials.Credentials)
        """Provides access to Google Cloud Platform (GCP) credentials.

        Delegates to `self.cloud_manager.gcp_credentials`.

        Returns:
            Any: The GCP credentials object.

        """
        return self.cloud_manager.gcp_credentials

    def get_gcp_access_token(self) -> str:
        """Get a valid GCP access token, refreshing if needed.

        Delegates to `self.cloud_manager.get_access_token()`.

        Returns:
            str: A valid OAuth2 access token

        """
        return self.cloud_manager.get_access_token()

    @property
    def gcs(self) -> Any:  # Type hint could be storage.Client
        """Provides access to the Google Cloud Storage (GCS) client.

        Delegates to `self.cloud_manager.gcs`.

        Returns:
            Any: The GCS client instance.

        """
        return self.cloud_manager.gcs

    @property
    def bq(self) -> Any:  # Type hint could be bigquery.Client
        """Provides access to the Google BigQuery client.

        Delegates to `self.cloud_manager.bq`.

        Returns:
            Any: The BigQuery client instance.

        """
        return self.cloud_manager.bq

    @cached_property
    def weave(self) -> Any:  # Type hint could be weave.weave_types.WeaveClient
        """Provides access to the Weights & Biases Weave client for tracing.

        Initializes Weave with a collection name derived from `self.name` (flow name)
        and `self.job` (job name). Handles connection failures gracefully by falling
        back to offline mode or returning a mock client. Cached for performance.

        Returns:
            Any: The initialized Weave client instance, or a mock client if initialization fails.

        """
        import weave  # Ensure weave is imported - deferred until first access

        collection_name = f"{self.name}-{self.job}"  # Construct collection name

        try:
            # Try to initialize weave with a reasonable timeout
            logger.debug(f"Initializing Weave with collection: {collection_name}")
            client = weave.init(collection_name)
            logger.debug("Weave initialized successfully")
            return client
        except Exception as e:
            # Log the error but don't fail the entire initialization
            logger.warning(f"Weave initialization failed: {e}. Continuing without weave tracing.")

            # Return a mock client that provides the basic interface but does nothing
            class MockWeaveClient:
                def __init__(self):
                    self.collection_name = collection_name

                def create_call(self, *args, **kwargs):
                    return None

                def finish_call(self, *args, **kwargs):
                    pass

                def get_call(self, *args, **kwargs):
                    return None

                def __getattr__(self, name):
                    # Return a no-op function for any other method calls
                    def noop(*args, **kwargs):
                        return None

                    return noop

            return MockWeaveClient()

    @property
    def credentials(self) -> dict[str, str]:
        """Retrieves and caches shared system credentials from the secret manager.

        Fetches credentials using `_SHARED_CREDENTIALS_KEY` on first access.

        Returns:
            dict[str, str]: A dictionary of shared credentials.

        Raises:
            TypeError: If the credentials retrieved from the secret manager
                are not a dictionary.

        """
        if self._credentials_cached is None:
            creds = self.secret_manager.get_secret(cfg_key=_SHARED_CREDENTIALS_KEY)
            if not isinstance(creds, dict):
                raise TypeError(f"Expected shared credentials to be a dict, got {type(creds)}")
            self._credentials_cached = creds
        return self._credentials_cached

    def setup_logging(self, verbose: bool = False) -> None:
        """Sets up logging for the Buttermilk application.

        Configures console logging (with colors via `coloredlogs`) and optionally
        Google Cloud Logging if `self.logger_cfg` is set up for GCP.
        Sets logging levels for Buttermilk's logger and other loggers.

        Args:
            verbose (bool): If True, sets Buttermilk logger level to DEBUG and
                enables asyncio debug mode. Otherwise, sets to INFO.
                Defaults to False.

        Raises:
            RuntimeError: If GCP logger is configured but essential attributes
                like 'project' or 'location' are missing in `self.logger_cfg`.

        """
        import sys

        import coloredlogs  # For colored console output

        # Logger config validation is now done in _validate_logger_config() during initialization

        # Clear existing handlers from the root logger to avoid duplicate logs
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set up console logging
        context_filter = ContextFilter()
        logger.addFilter(context_filter)
        # Original format: "%(asctime)s %(hostname)s %(name)s [%(session_id)s:%(agent_id)s] %(filename)s:%(lineno)d %(levelname)s %(message)s"
        # Shorter: Timestamp [short_context] LEVEL filename: Message
        console_format = "%(asctime)s [%(short_context)s] %(levelname)s %(filename)s:%(lineno)d %(message)s"

        coloredlogs.install(
            logger=logger,  # Target Buttermilk's main logger
            fmt=console_format,
            isatty=True,  # Enable colors if output is a TTY
            stream=sys.stdout,  # Log to stdout for better test visibility
            level=logging.DEBUG if verbose else logging.INFO,
        )

        # Add file logging when verbose is True
        if verbose:
            log_filename = f"/tmp/buttermilk_{self.run_id}.log"

            # Create file handler
            file_handler = logging.FileHandler(log_filename, mode="w")
            file_handler.setLevel(logging.DEBUG)

            # Use the same format as console but without colors
            file_formatter = logging.Formatter(console_format)
            file_handler.setFormatter(file_formatter)

            # Add the same context filter
            file_handler.addFilter(context_filter)

            # Add handler to the logger
            logger.addHandler(file_handler)
            logger.info(f"Verbose logging enabled - also writing to: {log_filename}")

        # Defer Google Cloud Logging setup to improve startup performance
        # Cloud logging will be initialized on first cloud operation
        if self.logger_cfg and self.logger_cfg.type == "gcp":
            logger.debug("Cloud logging configuration detected - will be initialized on first cloud access")

        # Set default logging levels for other loggers to WARNING to reduce noise
        root_logger.setLevel(logging.WARNING)
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            # Check if it's a Logger instance to avoid issues with placeholders
            if isinstance(logging.Logger.manager.loggerDict[logger_name], logging.Logger):
                logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Set Buttermilk's own logger level based on verbosity
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # Configure asyncio debug mode based on verbosity
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_running():
                current_loop.set_debug(verbose)
        except RuntimeError:  # No event loop running
            pass

        # Log initialization message
        log_init_message = f"Logging set up for run: {self.run_info}. Save directory: {self.save_dir}"
        # Note: cloud_logging_resource is only available if cloud logging is active
        # It's set up in _setup_cloud_logging() which is called lazily

        logger.info(log_init_message, extra={"run_details": self.run_info.model_dump(exclude_none=True)})

        # Log Buttermilk version if available
        try:
            from importlib.metadata import version

            bm_version = version("buttermilk")  # Assumes package is named 'buttermilk'
            logger.debug(f"Buttermilk version: {bm_version}")
        except Exception:  # importlib.metadata.PackageNotFoundError or other issues
            logger.debug("Could not determine Buttermilk version.")

    def start_fetch_ip_task(self) -> None:
        """Starts an asynchronous task to fetch the machine's external IP address.

        The IP address is stored in `self._ip` upon completion. This task is
        initiated if an event loop is running and the task hasn't been started already.
        """
        from buttermilk.utils import get_ip  # Utility function to get IP

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Start task only if it hasn't been started or is already done
                if not hasattr(self, "_get_ip_task") or self._get_ip_task is None or self._get_ip_task.done():

                    async def _fetch_and_set_ip():
                        self._ip = await get_ip()
                        logger.debug(f"Fetched IP address: {self._ip}")

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
        elif self.save_dir:
            effective_save_dir_str = self.save_dir
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
            logger.info(  # Log as a dictionary for structured logging if supported
                {
                    "message": f"Successfully saved data to: {saved_file_path}",
                    "uri": str(saved_file_path),  # Ensure URI is a string
                    "run_id": self.run_id,  # Include run_id for context
                },
            )
            return str(saved_file_path)  # Return path as string
        except Exception as e:
            logger.error(f"Failed to save data to '{effective_save_dir_str}' with extension '{effective_extension}': {e!s}")
            return None  # Indicate save failure

    def run_query(
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
            save_dir=self.save_dir,  # Pass BM's default save directory
            return_df=return_df,
        )

    def get_storage(self, config: StorageConfig | dict | None = None) -> Any:
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
        if not isinstance(config, BaseStorageConfig):
            # Convert OmegaConf objects to StorageConfig
            # This is necessary for Hydra integration
            try:
                from omegaconf import DictConfig, OmegaConf

                if isinstance(config, DictConfig):
                    config_dict = OmegaConf.to_container(config, resolve=True)
                    config = StorageFactory.create_config(config_dict)
                else:
                    raise ValueError(f"Config must be a BaseStorageConfig or OmegaConf DictConfig, got {type(config)}")
            except ImportError:
                raise ValueError("Config must be a BaseStorageConfig object") from None

        # Use the storage factory to create the appropriate storage instance
        from buttermilk._core.storage_config import StorageFactory

        return StorageFactory.create_storage(config, self)

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

    def get_bigquery_storage(self, dataset_name: str, **kwargs) -> Any:
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
