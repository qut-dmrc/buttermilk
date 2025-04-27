from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    List,
    Literal,
    Optional,
    Self,
)

import cloudpathlib
from google.cloud.bigquery.schema import SchemaField
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from shortuuid import uuid

from buttermilk._core.exceptions import FatalError
from buttermilk._core.log import logger
from buttermilk.utils.utils import expand_dict
from buttermilk.utils.validators import (
    convert_omegaconf_objects,
    uppercase_validator,  # Pydantic validators
)

from .defaults import BQ_SCHEMA_DIR
from .types import SessionInfo, RunRequest

BASE_DIR = Path(__file__).absolute().parent


CloudProvider = Literal[
    "gcp",
    "bq",
    "aws",
    "azure",
    "env",
    "local",
    "gsheets",
    "vertex",
]


class CloudProviderCfg(BaseModel):
    type: CloudProvider

    model_config = ConfigDict(
        # Exclude fields with None values when serializing
        exclude_none=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        # Ignore extra fields not defined in the model
        extra="allow",
        exclude_unset=True,
        include_extra=True,
    )


class SaveInfo(CloudProviderCfg):
    destination: str | cloudpathlib.AnyPath | None = None
    db_schema: str = Field(..., description="Local name or path for schema file")
    dataset: str | None = Field(default=None)

    _loaded_schema: List[SchemaField] = PrivateAttr(default=[])

    # model_config = ConfigDict(
    #     json_encoders={
    #         np.bool_: bool,
    #         datetime.datetime: lambda v: v.isoformat(),
    #         ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
    #         DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
    #     },
    # )

    @field_validator("db_schema")
    def file_must_exist(cls, v):
        if v:
            try:
                if Path(v).exists():
                    return v.as_posix()
            except Exception:
                pass
            f = Path(BQ_SCHEMA_DIR) / v
            if f.exists():
                return f.as_posix()
            raise ValueError(f"File '{v}' does not exist.")
        return v

    @model_validator(mode="after")
    def check_destination(self) -> "SaveInfo":
        if not self.destination and not self.dataset:
            if self.type == "gsheets":
                return self  # We'll create a new sheet when we need to
            raise ValueError(
                "Nowhere to save to! Either destination or dataset must be provided.",
            )
        return self

    @computed_field
    @property
    def bq_schema(self) -> List[SchemaField]:
        if not self._loaded_schema:
            from buttermilk.bm import bm

            self._loaded_schema = bm.bq.schema_from_json(self.db_schema)
        return self._loaded_schema


class DataSourceConfig(BaseModel):
    max_records_per_group: int = -1
    type: Literal[
        "job",
        "file",
        "bq",
        "generator",
        "plaintext",
        "chromadb",
        "outputs",
    ]
    path: str = Field(
        default="",
    )
    glob: str = Field(default="**/*")
    filter: Mapping[str, str | Sequence[str] | None] | None = Field(
        default_factory=dict,
    )
    join: Mapping[str, str] | None = Field(default_factory=dict)
    index: list[str] | None = None
    agg: bool | None = Field(default=False)
    group: Mapping[str, str] | None = Field(default_factory=dict)
    columns: Mapping[str, str | Mapping] | None = Field(default_factory=dict)
    last_n_days: int = Field(default=7)
    db: Mapping[str, str] = Field(default={})
    embedding_model: str = Field(default="")
    dimensionality: int = Field(default=-1)
    persist_directory: str = Field(default="")
    collection_name: str = Field(default="")

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )


class DataSouce(DataSourceConfig):
    pass


class ToolConfig(BaseModel):
    description: str = Field(default="")
    tool_obj: str = Field(default="")

    data: Mapping[str, DataSourceConfig] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )

    def get_functions(self) -> list[Any]:
        """Create function definitions for this tool."""
        raise NotImplementedError()

    async def _run(self, **kwargs) -> list[Any] | None:
        raise NotImplementedError()


class Tracing(BaseModel):
    enabled: bool = False
    api_key: str = ""
    provider: str = ""
    endpoint: str | None = None
    otlp_headers: Mapping | None = Field(default_factory=dict)


class Project(BaseModel):
    connections: Sequence[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg = Field(default=None)
    logger_cfg: CloudProviderCfg = Field(default=None)
    pubsub: CloudProviderCfg = Field(default=None)
    clouds: list[CloudProviderCfg] = Field(default_factory=list)
    tracing: Tracing | None = Field(default_factory=Tracing)
    run_info: Optional[SessionInfo] = Field(
        default=None,
        description="Information about the context in which this project runs",
    )
    datasets: dict[str, DataSourceConfig] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )


# --- Agent Configuration ---
class AgentConfig(BaseModel):
    """
    Base Pydantic model defining the configuration structure for all Buttermilk agents.

    Loaded and instantiated by Hydra based on YAML configuration files. Includes
    core identification, behavior parameters, and connections to data/tools.
    """

    # Core Identification
    id: str = Field(default="", description="Unique identifier for the agent instance, generated automatically.")
    role: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        default="",
        description="The functional role this agent plays in the workflow (e.g., 'judge', 'conductor').",
    )
    name: str = Field(default="", description="A human-friendly name for the agent instance, often including the role and a unique ID.")
    description: str = Field(
        default="",
        description="A brief explanation of the agent's purpose and capabilities.",
    )

    # Behavior & Connections
    tools: list[ToolConfig] = Field(
        default_factory=list,  # Use factory for mutable default
        description="Configuration for tools (functions) that the agent can potentially use.",
    )
    data: Mapping[str, DataSourceConfig] = Field(
        default_factory=dict,  # Use factory for mutable default
        description="Configuration for data sources the agent might need access to.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration parameters (e.g., model name, template name, thresholds).",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Defines mappings for how incoming data should populate the agent's input context (using JMESPath).",
    )
    # TODO: 'outputs' field seems unused currently. Define its purpose or remove.
    outputs: dict[str, Any] = Field(default_factory=dict)

    # Pydantic Model Configuration
    model_config = {
        "extra": "allow",  # Allow extra fields not explicitly defined (useful with Hydra).
        "arbitrary_types_allowed": False,  # Disallow arbitrary types unless explicitly handled.
        "populate_by_name": True,  # Allow population by field name.
    }

    # Private Attributes (Internal state, often defaults or generated)
    unique_identifier: str = Field(default_factory=lambda: uuid()[:6])  # Short unique ID component.
    base_name: str | None = Field(default=None, description="Base name component, initially derived from 'name'.", exclude=False)
    # Defines which attributes are combined to create the human-friendly 'name'.
    _name_components: list[str] = ["base_name", "_id"]

    # Field Validators
    # Ensure OmegaConf objects (like DictConfig) are converted to standard Python dicts before validation.
    _validate_parameters = field_validator("parameters", "inputs", "outputs", mode="before")(convert_omegaconf_objects())

    @model_validator(mode="after")
    def _generate_name_and_id(self) -> Self:
        """
        Generates the unique `id` and formatted `name` for the agent instance after validation.

        - Sets `id` based on `role` and `_id`.
        - Sets `name` by combining components defined in `_name_components`.
        """
        # Generate the full unique ID: e.g., "judge-a1b2c3"
        self.id = f"{self.role}-{self.unique_identifier}"

        # Set the base_name from the initially provided 'name' if not already set.
        if not self.base_name:
            if not self.name:  # Ensure a base name was provided in the config.
                raise ValueError("AgentConfig requires a human-friendly 'name' field in configuration.")
            self.base_name = self.name  # Store the original human-friendly name part.

        # Construct the final display 'name' from specified components.
        name_parts = [getattr(self, comp, None) for comp in self._name_components]
        self.name = " ".join([str(part) for part in name_parts if part])  # Join non-None parts.
        return self


class AgentVariants(AgentConfig):
    """
    A factory for creating Agent instance variants based on parameter combinations.

    Defines two types of variants:
    1. `parallel_variants`: Parameters whose combinations create distinct agent instances
       (e.g., different models). These agents can potentially run in parallel.
    2. `sequential_variants`: Parameters whose combinations define sequential tasks
       executed by *each* agent instance created from `parallel_variants`.

    Example:
    ```yaml
    - id: ANALYST
      role: "Analyst"
      agent_obj: LLMAgent
      num_runs: 1
      variants:
        model: ["gpt-4", "claude-3"]    # Creates 2 parallel agent instances
      tasks:
        criteria: ["accuracy", "speed"] # Each agent instance runs 2 tasks sequentially
        temperature: [0.5, 0.8]         # Total 4 sequential tasks per agent
                                        # (accuracy/0.5, accuracy/0.8, speed/0.5, speed/0.8)
      parameters:
        template: analyst               # parameter sets shared for each task
      inputs:
        results: othertask.outputs.results  # dynamic inputs mapped from other data
    ```
    """
    #--- Variant configuration: fields used to generate AgentConfig objects ---
    agent_obj: str = Field(  # Class name used by Hydra for instantiation.
        default="",
        description="The Python class name of the agent implementation to instantiate (e.g., 'Judge', 'LLMAgent').",
    )
    variants: dict = Field(default_factory=dict, description="Parameters for parallel agent variations.")
    tasks: dict = Field(default_factory=dict, description="Parameters for sequential tasks within each parallel variation.")
    num_runs: int = Field(default=1, description="Number of times to replicate each parallel variant configuration.")
    extra_params: list[str] = Field(default=[], description="Extra parameters to look for in runtime request.")
    #--- Variant configuration: fields used to generate AgentConfig objects ---

    def get_configs(self, params: RunRequest|None = None) -> list[tuple[type, AgentConfig]]:
        """
        Generates agent configurations based on parallel and sequential variants.
        """
        # Get static config (base attributes excluding variant fields)
        static_config = self.model_dump(
            exclude={
                "parallel_variants",
                "id","unique_identifier",
                "sequential_variants",
                "num_runs",
                "parameters",
                "tasks",
            }
        )
        base_parameters = self.parameters.copy()  # Base parameters common to all

        # Get extra parameters passed in at runtime if requested
        if params:
            for key in self.extra_params:
                if key not in params.model_fields_set:
                    raise ValueError(f"Cannot find parameter {key} in runtime dict for agent {self.id}.")
                base_parameters[key] = getattr(params, key)

        # Get agent class
        from buttermilk._core.variants import AgentRegistry
        agent_class = AgentRegistry.get(self.agent_obj)

        # Expand parallel variants
        parallel_variant_combinations = expand_dict(self.variants)
        if not parallel_variant_combinations:
            parallel_variant_combinations = [{}]  # Ensure at least one base agent config

        # Expand sequential variants
        sequential_task_sets = expand_dict(self.tasks)
        if not sequential_task_sets:
            sequential_task_sets = [{}]  # Default: one task with no specific sequential parameters

        generated_configs = []
        # Create agent configs based on combinations of parallel and sequential variants, and num_runs
        for i in range(self.num_runs):
            for parallel_params in parallel_variant_combinations:
                for task_params in sequential_task_sets:
                    # Start with static config
                    cfg_dict = static_config.copy()

                    # Combine base parameters, parallel variant parameters, and sequential task parameters
                    # Order matters: task parameters overwrite parallel, parallel overwrite base
                    combined_params = {**base_parameters, **parallel_params, **task_params}
                    cfg_dict["parameters"] = combined_params

                    # Create and add the AgentConfig instance
                    try:
                        # Ensure AgentConfig allows extra fields if needed, or filter cfg_dict
                        # AgentConfig currently has extra='allow', so unknown fields are okay
                        agent_config_instance = AgentConfig(**cfg_dict)
                        generated_configs.append((agent_class, agent_config_instance))
                    except Exception as e:
                        logger.error(f"Error creating AgentConfig for {cfg_dict.get('role', 'unknown')} with parameters {combined_params}: {e}")
                        raise  # Re-raise by default

        if not generated_configs:  # Check if list is empty
            raise FatalError(f"Could not create any agent variant configs for {self.role} {self.name}")

        return generated_configs
