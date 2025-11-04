"""Run configuration with all execution parameters consolidated.

This module provides the RunConfig class that contains all execution-related parameters
including mode, flow selection, limits, API settings, and pipeline configuration.
The RunMode enum defines valid execution modes.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    pass


class RunMode(str, Enum):
    """Valid execution modes for Buttermilk.

    Each mode determines how Buttermilk executes and what parameters are relevant:

    - console: Interactive CLI execution
    - batch: Create batch jobs only
    - batch_run: Process existing batch jobs
    - batch_all: Create and process batch jobs
    - api: Run FastAPI server
    - pipeline: Multi-stage data processing
    - streamlit: Streamlit web interface
    - pubsub: Google Cloud Pub/Sub listener
    - slackbot: Slack bot integration
    """

    CONSOLE = "console"
    BATCH = "batch"
    BATCH_RUN = "batch_run"
    BATCH_ALL = "batch_all"
    API = "api"
    PIPELINE = "pipeline"
    STREAMLIT = "streamlit"
    PUBSUB = "pub/sub"
    SLACKBOT = "slackbot"


class RunConfig(BaseModel):
    """Configuration for all execution parameters.

    This class consolidates all execution-related configuration in one place,
    making it clear which parameters control program execution versus which
    are static configuration.

    All mode-specific parameters are here with clear defaults and documentation.
    The main config root contains only universal essentials: project_name, job, verbose.

    Mode is set via Hydra config groups: run=api loads conf/run/api.yaml which has mode: api.

    Examples:
        Console mode (run=console):
            run:
              mode: console
              flow: trans
              record_id: "123"

        API mode (run=api):
            run:
              mode: api
              host: 0.0.0.0
              port: 8000
              workers: 4
              reload: true

        Batch mode (run=batch):
            run:
              mode: batch_run
              flow: trans
              limit: 50

        Pipeline mode (run=pipeline):
            run:
              mode: pipeline
              pipeline:
                source: {...}
                output: {...}
                concurrency: 20
    """

    # Execution mode (loaded from run config group)
    mode: RunMode = Field(default=RunMode.CONSOLE, description="Execution mode (set via run=api, run=batch, etc.)")

    # Flow definitions (configuration, not execution)
    flows: dict[str, Any] = Field(
        default_factory=dict, description="Flow definitions keyed by flow name. Each flow must conform to OrchestratorProtocol"
    )

    # Flow execution parameters
    flow: str | None = Field(default=None, description="Which flow to execute (required for console/batch modes)")
    limit: int | None = Field(default=None, description="Unified limit for records/jobs to process (replaces max_records/max_jobs)")
    record_id: str | None = Field(default=None, description="Specific record ID for console mode testing")

    # API mode settings
    host: str = Field(default="0.0.0.0", description="API server host (api mode)")
    port: int = Field(default=8000, description="API server port (api mode)")
    workers: int = Field(default=1, description="Number of worker processes (api mode)")
    reload: bool = Field(default=False, description="Enable hot reloading for development (api mode)")
    log_level: str = Field(default="info", description="Logging level (api mode)")

    # Pipeline mode configuration
    pipeline: Any = Field(  # Will be PipelineConfig but avoid circular import
        default=None, description="Pipeline processing configuration (pipeline mode)"
    )

    # Storage override for batch modes
    storage_config: dict[str, Any] | None = Field(default=None, description="Storage configuration override for batch modes")

    model_config = {
        "extra": "allow",  # Allow additional fields for flexibility
        "arbitrary_types_allowed": True,  # Allow complex types like PipelineConfig
    }

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: Any) -> RunMode:
        """Validate mode is a recognized run mode.

        Accepts both RunMode enum values and string values.
        """
        if isinstance(v, RunMode):
            return v
        if isinstance(v, str):
            # Try to match against enum values
            try:
                return RunMode(v)
            except ValueError:
                valid_modes = [m.value for m in RunMode]
                raise ValueError(f"Invalid run mode: {v}. Must be one of: {', '.join(valid_modes)}")
        raise ValueError(f"Mode must be a string or RunMode, got {type(v)}")
