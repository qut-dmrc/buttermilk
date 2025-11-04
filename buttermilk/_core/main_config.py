"""Type-safe main configuration model for Buttermilk.

This module provides the root Pydantic model that encompasses the entire
Buttermilk configuration structure, making it clear what options control what
and where they should go.
"""

from typing import Any

from omegaconf import DictConfig
from pydantic import BaseModel, Field, field_validator, model_validator

from buttermilk._core.bm_init import SessionInfo
from buttermilk._core.cloud_config import CloudProvider, LoggerConfig
from buttermilk._core.run_config import RunConfig
from buttermilk._core.storage_config import BaseStorageConfig


class TracingProviderConfig(BaseModel):
    """Configuration for a single tracing provider (Weave, Traceloop, OTEL)."""

    enabled: bool = Field(default=False, description="Whether this tracing provider is enabled")
    project_id: str | None = Field(default=None, description="Project/entity ID for the tracing provider")
    api_key: str | None = Field(default=None, description="API key for authentication (may be None for local/OTEL)")
    endpoint: str | None = Field(default=None, description="Custom endpoint URL for the tracing provider")

    model_config = {
        "extra": "allow",  # Allow provider-specific fields
    }


class TracingConfig(BaseModel):
    """Configuration for all tracing providers.

    Supports multiple tracing providers (Weave, Traceloop, OTEL) that can be
    enabled simultaneously.
    """

    weave: TracingProviderConfig = Field(
        default_factory=lambda: TracingProviderConfig(enabled=False), description="Weights & Biases Weave tracing configuration"
    )
    traceloop: TracingProviderConfig = Field(
        default_factory=lambda: TracingProviderConfig(enabled=False), description="Traceloop tracing configuration"
    )
    otel: TracingProviderConfig = Field(
        default_factory=lambda: TracingProviderConfig(enabled=False), description="OpenTelemetry tracing configuration"
    )

    model_config = {
        "extra": "allow",  # Allow additional tracing providers
    }


class InfrastructureConfig(BaseModel):
    """Configuration for all infrastructure components.

    This is the central container for clouds, LLMs, tracing, and logging
    configuration. It replaces the previous scattered approach.
    """

    # Cloud providers (GCP, AWS, Azure)
    clouds: list[CloudProvider] = Field(default_factory=list, description="List of cloud provider configurations")

    # LLM configurations
    llms: dict[str, Any] = Field(default_factory=dict, description="LLM model configurations keyed by model identifier")

    # Unified tracing configuration
    tracing: TracingConfig | dict[str, Any] = Field(default_factory=TracingConfig, description="Tracing provider configurations")

    # Logging configuration
    logging: LoggerConfig | dict[str, Any] | None = Field(default=None, description="Logging configuration")

    # Dataset configurations
    datasets: dict[str, BaseStorageConfig] = Field(default_factory=dict, description="Named dataset storage configurations")

    model_config = {
        "extra": "allow",  # Allow additional infrastructure components
        "arbitrary_types_allowed": True,  # Allow complex types
    }

    @field_validator("tracing", mode="before")
    @classmethod
    def parse_tracing_config(cls, v: Any) -> TracingConfig | dict:
        """Parse tracing configuration from dict or TracingConfig."""
        if isinstance(v, TracingConfig):
            return v
        if isinstance(v, dict):
            # Convert nested dicts to TracingProviderConfig
            tracing_dict = {}
            for provider, config in v.items():
                if isinstance(config, dict):
                    tracing_dict[provider] = TracingProviderConfig(**config)
                else:
                    tracing_dict[provider] = config
            return TracingConfig(**tracing_dict)
        return v


class ButtermilkConfig(BaseModel):
    """Root configuration model for Buttermilk.

    This is the main configuration object that encompasses all settings
    and makes it clear what options control what and where they should go.

    Structure:
    - Root level: Universal essentials only (project_name, job, verbose)
    - run: All execution parameters including mode (flow, limit, host, port, pipeline, etc.)
    - session: Session information (direct SessionInfo)
    - infrastructure: Cloud providers, LLMs, tracing, logging
    - flows: Flow definitions (configuration, not execution)
    - storage: Named storage configurations

    Key Principles:
    - Mode is INSIDE run config (loaded from run=api, run=batch, etc.)
    - job and project_name are top-level universal essentials but get shifted into session info
    - All execution params in run config
    - Session is direct SessionInfo (no wrapper)
    - Unified limit parameter in run config
    """

    # Root level: Universal essentials only
    verbose: bool = Field(default=False, description="Enable verbose logging output")

    # Run configuration: All execution parameters (including mode)
    run: RunConfig | dict[str, Any] = Field(
        default_factory=RunConfig, description="All execution parameters including mode (flow, limit, host, port, pipeline, etc.)"
    )

    # Session information (direct, no wrapper)
    session: SessionInfo = Field(description="Session-specific information and tracking")

    # Infrastructure configuration
    infrastructure: InfrastructureConfig = Field(
        default_factory=InfrastructureConfig, description="Infrastructure components (clouds, LLMs, tracing, logging)"
    )

    # Flow definitions (configuration, not execution)
    # MOVED TO run.flows - kept for backward compatibility only via validators
    #     flows: dict[str, Any] = Field(default_factory=dict, description="Flow definitions keyed by flow name")

    # Storage configurations
    storage: dict[str, BaseStorageConfig | dict[str, Any]] = Field(default_factory=dict, description="Named storage configurations")

    model_config = {
        "extra": "allow",  # Allow additional fields for flexibility
        "arbitrary_types_allowed": True,  # Allow complex types
    }

    @model_validator(mode="before")
    @classmethod
    def move_job_project_to_session(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Move job and project_name into session if they exist at root level."""
        job = values.pop("job", None)
        project_name = values.pop("project_name", None)
        if "session" not in values or not isinstance(values["session"], dict):
            values["session"] = {}
        if job is not None:
            values["session"]["job"] = job
        if project_name is not None:
            values["session"]["project_name"] = project_name
        # Migrate flows from root to run.flows for backward compatibility
        if "flows" in values and values["flows"]:
            if "run" not in values:
                values["run"] = {}
            if isinstance(values["run"], dict) and "flows" not in values["run"]:
                values["run"]["flows"] = values.pop("flows")
        return values

    @field_validator("run", mode="before")
    @classmethod
    def parse_run_config(cls, v: Any) -> RunConfig:
        """Parse run configuration from dict or RunConfig."""
        if isinstance(v, RunConfig):
            return v
        if isinstance(v, dict):
            return RunConfig(**v)
        raise ValueError(f"Invalid run configuration type: {type(v)}")

    @field_validator("session", mode="before")
    @classmethod
    def parse_session_info(cls, v: Any) -> SessionInfo:
        """Parse session information from SessionInfo or dict."""
        if isinstance(v, SessionInfo):
            return v
        if isinstance(v, dict):
            return SessionInfo(**v)
        raise ValueError(f"Invalid session configuration type: {type(v)}")

    def get_storage_config(self, name: str) -> BaseStorageConfig | None:
        """Get a named storage configuration.

        Args:
            name: Storage configuration name

        Returns:
            StorageConfig instance or None if not found
        """
        storage = self.storage.get(name)
        if storage is None:
            return None
        if isinstance(storage, BaseStorageConfig):
            return storage
        if isinstance(storage, dict):
            from buttermilk._core.storage_config import StorageFactory

            return StorageFactory.create_config(storage)
        return None


def create_config_from_hydra(cfg: DictConfig) -> ButtermilkConfig:
    """Create a typed ButtermilkConfig from a Hydra DictConfig.

    This function converts an OmegaConf DictConfig (from Hydra) into a
    fully typed Pydantic model, preserving all the flexibility of Hydra
    composition while adding type safety.

    Args:
        cfg: Hydra DictConfig from compose() or @hydra.main

    Returns:
        Typed ButtermilkConfig instance

    Example:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="conf"):
        ...     cfg = compose(config_name="config")
        ...     typed_cfg = create_config_from_hydra(cfg)
        ...     assert isinstance(typed_cfg, ButtermilkConfig)
    """
    from omegaconf import OmegaConf

    # Convert DictConfig to dict, resolving all interpolations
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create typed config
    return ButtermilkConfig(**cfg_dict)
