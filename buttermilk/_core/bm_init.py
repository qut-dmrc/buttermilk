"""Buttermilk initialization and resource management.

This module provides the main singleton (BM) that serves as the central access point for
all Buttermilk resources such as LLMs, cloud providers, and configuration.

BM implements a clean singleton pattern that allows for easy access to resources from
anywhere in the codebase while maintaining proper initialization and configuration.
"""

from __future__ import annotations  # Enable postponed annotations

import asyncio
import datetime
import logging
import platform
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import psutil
import pydantic
import shortuuid
from cloudpathlib import AnyPath, CloudPath
from pydantic import BaseModel, Field, PrivateAttr
from rich import print

from buttermilk._core.cloud import CloudManager
from buttermilk._core.config import CloudProviderCfg, DataSourceConfig, Tracing
from buttermilk._core.keys import SecretsManager
from buttermilk._core.llms import LLMs
from buttermilk._core.log import ContextFilter, logger
from buttermilk._core.query import QueryRunner
from buttermilk._core.utils.lazy_loading import cached_property
from buttermilk.utils import save

# Constants
CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
_MODELS_CFG_KEY = "models_secret"
_SHARED_CREDENTIALS_KEY = "credentials_secret"

# Global run ID for all BM instances
_global_run_id = ""


def _make_run_id() -> str:
    """Generate a unique run ID for the current execution.
    
    The ID includes timestamp, machine info, and a random component to ensure uniqueness.
    
    Returns:
        A unique string identifier for this run

    """
    global _global_run_id
    if _global_run_id:
        return _global_run_id

    # Create a unique identifier for this run
    node_name = platform.uname().node
    username = psutil.Process().username()
    # get rid of windows domain if present
    username = str.split(username, "\\")[-1]

    # The ISO 8601 format has too many special characters for a filename, so we'll use a simpler format
    run_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    _global_run_id = run_id
    return run_id


class SessionInfo(BaseModel):
    """Information about the current execution session."""

    platform: str = "local"
    name: str
    job: str
    run_id: str = Field(default_factory=_make_run_id)
    ip: str | None = Field(default=None)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    save_dir: str | None = None
    flow_api: str | None = None

    _get_ip_task: asyncio.Task | None = PrivateAttr(default=None)
    _ip: str | None = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
        }


class BM(SessionInfo):
    """Central singleton for Buttermilk providing access to all resources.
    
    BM serves as a gateway to all major components such as cloud clients,
    LLM connections, and configuration. It handles initialization of these
    components and provides a unified interface for accessing them.
    
    Typical usage:
    ```python
    from buttermilk._core.dmrc import get_bm
    
    # Access client
    bm = get_bm()
    
    # Use a cloud client
    result = bm.gcs.list_blobs("my-bucket")
    
    # Use an LLM
    response = bm.llms.gpt41.create(messages=[...])
    ```
    """

    connections: list[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg | None = Field(default=None)
    logger_cfg: CloudProviderCfg | None = Field(default=None)
    pubsub: CloudProviderCfg | None = Field(default=None)
    clouds: list[CloudProviderCfg] = Field(default_factory=list)
    tracing: Tracing | None = Field(default_factory=Tracing)
    datasets: dict[str, DataSourceConfig] = Field(default_factory=dict)

    save_dir_base: str = Field(
        default_factory=mkdtemp,
        validate_default=True,
    )  # Default to temp dir

    # Private attributes for delegation to specialized classes
    _cloud_manager: CloudManager | None = PrivateAttr(default=None)
    _secret_manager: SecretsManager | None = PrivateAttr(default=None)
    _llms_instance: LLMs | None = PrivateAttr(default=None)
    _query_runner: QueryRunner | None = PrivateAttr(default=None)
    _credentials_cached: dict[str, str] | None = PrivateAttr(default=None)

    # --- Property Validators ---

    @pydantic.validator("save_dir_base", pre=True)
    def get_save_dir(cls, save_dir_base) -> str:
        """Validate and normalize save directory path."""
        if isinstance(save_dir_base, str):
            pass
        elif isinstance(save_dir_base, Path):
            save_dir_base = save_dir_base.as_posix()
        elif isinstance(save_dir_base, CloudPath):
            save_dir_base = save_dir_base.as_uri()
        else:
            raise ValueError(
                f"save_dir_base must be a string, Path, or CloudPath, got {type(save_dir_base)}",
            )
        return save_dir_base

    @pydantic.root_validator(pre=True)
    @classmethod
    def _remove_target(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Remove hydra target attribute from values."""
        values.pop("_target_", None)
        return values

    def __init__(self, **data: Any) -> None:
        """Initialize BM with configuration and set up resources."""
        super().__init__(**data)
        self._post_init_setup()

    def _post_init_setup(self) -> None:
        """Perform post-initialization setup."""
        # Set up save directory
        save_dir = AnyPath(self.save_dir_base) / self.name / self.job / self.run_id
        self.save_dir = str(save_dir)

        # Set up logging
        self.setup_logging(verbose=getattr(self.logger_cfg, "verbose", False))

        # Print config to console and save to default save dir
        print(self.dict())
        try:
            # Ensure save_dir is set before saving
            if self.save_dir:
                self.save(
                    data=[self.dict(), self.run_info.dict()],
                    basename="config",
                    extension=".json",
                    save_dir=self.save_dir,
                )
            else:
                logger.warning("save_dir is not set. Skipping saving config.")
        except Exception as e:
            logger.error(f"Could not save config to default save dir: {e}")

        # Start async operations
        self.start_fetch_ip_task()

        # Initialize cloud connections
        self._login_clouds()

    # --- Delegation Properties ---

    @cached_property
    def cloud_manager(self) -> CloudManager:
        """Get the cloud manager instance."""
        if self._cloud_manager is None:
            self._cloud_manager = CloudManager(clouds=self.clouds)
        return self._cloud_manager

    @cached_property
    def secret_manager(self) -> SecretsManager:
        """Get the secret manager instance."""
        if self._secret_manager is None:
            if not self.secret_provider:
                raise RuntimeError("Secret provider configuration is missing.")
            self._secret_manager = SecretsManager(**self.secret_provider.dict())
        return self._secret_manager

    @cached_property
    def llms(self) -> LLMs:
        """Get the LLMs manager instance."""
        if self._llms_instance is None:
            connections = None
            try:
                # Try to load from cache file first
                cache_path = Path(CONFIG_CACHE_PATH)
                if cache_path.exists():
                    from buttermilk.utils.utils import load_json_flexi
                    connections = load_json_flexi(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

            # If not loaded from cache, get from secret manager
            if not connections:
                try:
                    connections = self.secret_manager.get_secret(cfg_key=_MODELS_CFG_KEY)

                    # Cache for next time
                    try:
                        cache_path = Path(CONFIG_CACHE_PATH)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        import json
                        cache_path.write_text(json.dumps(connections), encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Could not cache LLM connections: {e}")
                except Exception as e:
                    logger.error(f"Failed to load LLM connections: {e}")
                    raise RuntimeError("Failed to load LLM connections") from e

            # Ensure connections is a dictionary
            if not isinstance(connections, dict):
                raise TypeError(f"LLM connections is not a dict: {type(connections)}")

            self._llms_instance = LLMs(connections=connections)

        return self._llms_instance

    @cached_property
    def query_runner(self) -> QueryRunner:
        """Get the query runner instance."""
        if self._query_runner is None:
            self._query_runner = QueryRunner(bq_client=self.bq)
        return self._query_runner

    @property
    def run_info(self) -> SessionInfo:
        """Get information about the current session."""
        return SessionInfo(
            platform=self.platform,
            name=self.name,
            job=self.job,
            node_name=self.node_name,
            ip=self._ip,
            save_dir=self.save_dir,
            flow_api=self.flow_api,
        )

    # --- Delegated Properties for Cloud Clients ---

    @property
    def gcp_credentials(self):
        """Get GCP credentials from cloud manager."""
        return self.cloud_manager.gcp_credentials

    @property
    def gcs(self):
        """Get Google Cloud Storage client."""
        return self.cloud_manager.gcs

    @property
    def bq(self):
        """Get BigQuery client."""
        return self.cloud_manager.bq

    @property
    def weave(self):
        """Get Weights & Biases weave client for tracing."""
        import weave
        collection = f"{self.name}-{self.job}"
        return weave.init(collection)

    @property
    def credentials(self) -> dict[str, str]:
        """Get all system credentials."""
        if self._credentials_cached is None:
            creds = self.secret_manager.get_secret(cfg_key=_SHARED_CREDENTIALS_KEY)
            if not isinstance(creds, dict):
                raise TypeError(f"Expected credentials to be a dict, got {type(creds)}")
            self._credentials_cached = creds
        return self._credentials_cached

    # --- Core Methods ---

    def setup_logging(self, verbose: bool = False) -> None:
        """Set up logging for Buttermilk.
        
        Args:
            verbose: Whether to enable debug logging

        """
        import sys

        import coloredlogs

        # Validate logger config if GCP logging is enabled
        if self.logger_cfg and self.logger_cfg.type == "gcp":
            if not hasattr(self.logger_cfg, "project"):
                raise RuntimeError("GCP logger config missing 'project' attribute")
            if not hasattr(self.logger_cfg, "location"):
                raise RuntimeError("GCP logger config missing 'location' attribute")

        # Clear existing handlers from root logger
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
            logger=logger,
            fmt=console_format,
            isatty=True,
            stream=sys.stderr,
            level=logging.DEBUG if verbose else logging.INFO,
        )

        # Set up cloud logging if configured
        resource = None
        if self.logger_cfg and self.logger_cfg.type == "gcp":
            from google.cloud import logging as gcp_logging

            resource = gcp_logging.Resource(
                type="generic_task",
                labels={
                    "project_id": getattr(self.logger_cfg, "project", ""),
                    "location": getattr(self.logger_cfg, "location", ""),
                    "namespace": self.name,
                    "job": self.job,
                    "task_id": self.run_id,
                },
            )

            # Get log client from cloud manager
            from google.cloud.logging_v2.handlers import CloudLoggingHandler

            # session_id and agent_id are added to the LogRecord by ContextFilter.
            # The CloudLoggingHandler is expected to automatically pick up these extra fields
            # and include them in the structured log entry sent to Google Cloud Logging.
            cloudHandler = CloudLoggingHandler(
                client=self.cloud_manager.gcs_log_client(self.logger_cfg),
                resource=resource,
                name=self.name,
                labels=self.dict(include={"run_id", "name", "job", "platform"}),
            )
            cloudHandler.setLevel(logging.INFO)  # Cloud logs at INFO level, not DEBUG
            logger.addHandler(cloudHandler)

        # Set default logging levels
        root_logger.setLevel(logging.WARNING)
        for logger_str in list(logging.Logger.manager.loggerDict.keys()):
            try:
                if isinstance(logging.Logger.manager.loggerDict[logger_str], logging.Logger):
                    logging.getLogger(logger_str).setLevel(logging.WARNING)
            except Exception:
                pass

        # Set Buttermilk logger level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.set_debug(True)
            except RuntimeError:
                pass  # No event loop
        else:
            logger.setLevel(logging.INFO)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.set_debug(False)
            except RuntimeError:
                pass  # No event loop

        # Log initialization
        message = f"Logging set up for: {self.run_info}. Ready. Save dir: {self.save_dir}"
        if resource:
            message = f"{message} {resource}"

        logger.info(
            message,
            extra=dict(run=self.run_info.dict() if self.run_info else {}),
        )

        # Log version info
        try:
            from importlib.metadata import version
            logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except Exception:
            pass

    def start_fetch_ip_task(self) -> None:
        """Start an async task to fetch the client's IP address."""
        from buttermilk.utils import get_ip

        # Initialize IP task asynchronously if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if not hasattr(self, "_get_ip_task") or self._get_ip_task is None:
                    self._get_ip_task = asyncio.create_task(get_ip())
            else:
                # No event loop running, skip IP task creation
                pass
        except RuntimeError:
            # No current event loop
            pass

    def _login_clouds(self) -> None:
        """Initialize cloud provider connections."""
        self.cloud_manager.login_clouds()

        # Set up tracing if configured
        if self.tracing and self.tracing.enabled:
            self.cloud_manager.setup_tracing(self.tracing)

    def save(
        self,
        data,
        save_dir: str | AnyPath | None = None,
        extension: str | None = None,
        **kwargs,
    ) -> str | None:
        """Save data to a file with standardized naming.
        
        Args:
            data: The data to save
            save_dir: Directory to save to (defaults to self.save_dir)
            extension: File extension to use (defaults to .json)
            **kwargs: Additional arguments to pass to save function
            
        Returns:
            Path to the saved file, or None if save failed

        """
        effective_save_dir = save_dir or self.save_dir

        # If we still don't have a save_dir, create a temp directory
        if not effective_save_dir:
            import tempfile
            effective_save_dir = tempfile.mkdtemp()
            logger.warning(f"No save_dir specified, using temp directory: {effective_save_dir}")
        # Provide default extension if None
        effective_extension = extension if extension is not None else ".json"

        try:
            result = save.save(
                data=data,
                save_dir=effective_save_dir,  # Use our processed directory
                extension=effective_extension,
                **kwargs,
            )
            logger.info(
                dict(
                    message=f"Saved data to: {result}",
                    uri=result,
                    run_id=self.run_id,
                ),
            )
            return result
        except Exception as e:
            logger.error(f"Failed to save data to {save_dir}: {e}")
            return None  # Indicate save failure

    def run_query(
        self,
        sql,
        destination=None,
        overwrite=False,
        do_not_return_results=False,
        save_to_gcs=False,
        return_df=True,
    ) -> Any:
        """Run a BigQuery SQL query.
        
        This is a delegation to the QueryRunner's run_query method.
        
        See QueryRunner.run_query for full documentation.
        """
        return self.query_runner.run_query(
            sql=sql,
            destination=destination,
            overwrite=overwrite,
            do_not_return_results=do_not_return_results,
            save_to_gcs=save_to_gcs,
            save_dir=self.save_dir,
            return_df=return_df,
        )
