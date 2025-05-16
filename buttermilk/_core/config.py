from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
)

import cloudpathlib
import jmespath
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
from .types import RunRequest

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

    _loaded_schema: list[SchemaField] = PrivateAttr(default=[])

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
    def bq_schema(self) -> list[SchemaField]:
        if not self._loaded_schema:
            from buttermilk import buttermilk as bm
            from buttermilk._core.log import logger  # noqa

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
        raise NotImplementedError

    async def _run(self, **kwargs) -> list[Any] | None:
        raise NotImplementedError


class Tracing(BaseModel):
    enabled: bool = False
    api_key: str = ""
    provider: str = ""
    endpoint: str | None = None
    otlp_headers: Mapping | None = Field(default_factory=dict)


# --- Agent Configuration ---
class AgentConfig(BaseModel):
    """Base Pydantic model defining the configuration structure for all Buttermilk agents.

    Loaded and instantiated by Hydra based on YAML configuration files. Includes
    core identification, behavior parameters, and connections to data/tools.
    """

    # Core Identification
    agent_id: str = Field(default="", description="Unique identifier for the agent instance, generated automatically.", validate_default=True)

    role: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        default="",
        description="The functional role this agent plays in the workflow (e.g., 'judge', 'conductor').",
    )
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
        serialization_alias="mapping_data",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration parameters (e.g., model name, template name, thresholds).",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Defines mappings for how incoming data should populate the agent's input context (using JMESPath).",
        serialization_alias="mapping_inputs",
    )
    # TODO: 'outputs' field seems unused currently. Define its purpose or remove.
    outputs: dict[str, Any] = Field(default_factory=dict,
        serialization_alias="mapping_outputs")

    # Defines which attributes are combined to create the human-friendly 'name'.
    name_components: list[str] = Field(default=["role", "unique_identifier"], exclude=False)

    # Pydantic Model Configuration
    model_config = {
        "extra": "allow",  # Allow extra fields not explicitly defined (useful with Hydra).
        "arbitrary_types_allowed": False,  # Disallow arbitrary types unless explicitly handled.
        "populate_by_name": True,  # Allow population by field name.
    }

    # Private Attributes (Internal state, often defaults or generated)
    unique_identifier: str = Field(default_factory=lambda: uuid()[:6])  # Short unique ID component.
    _agent_name:    str = PrivateAttr()  # Human-friendly name for the agent instance.
    # Field Validators
    # Ensure OmegaConf objects (like DictConfig) are converted to standard Python dicts before validation.
    _validate_parameters = field_validator("parameters", "inputs", "outputs", mode="before")(convert_omegaconf_objects())

    @computed_field
    @property
    def agent_name(self) -> str:
        """Generates a human-friendly name for the agent instance based on its role and unique identifier.
        
        The name is constructed from the `role`, `unique_identifier`, and any other specified components.
        You can use JMESPath expressions to extract values from the agent's inputs and parameters.

        Returns:
            str: The generated name for the agent instance.

        """
        return self._agent_name

    @model_validator(mode="after")
    def _generate_id(self) -> Self:
        """Generates an ID and human-friendly name for the agent instance based on its role and unique identifier.
        
        Designed to be idempotent to work with validate_assignment=True.
        
        - Sets `id` based on `role` and `unique_identifier`.

        - By default `name` is constructed from the `role`, `unique_identifier`, and any other specified components.

        You can use JMESPath expressions to extract values from the agent's inputs and parameters.

        Returns:
            Self: The updated instance of the AgentConfig class with the generated ID and name.

        """
        # 2. Calculate the intended final ID
        intended_id = f"{self.role}-{self.unique_identifier}"

        if self.agent_id != intended_id:
            self.agent_id = intended_id

        # Calculate the intended final name based on components
        name_parts = []
        inputs_dict = self.model_dump(exclude={"agent_name"})
        if "variants" in self.model_fields_set:
            inputs_dict.update(self.variants)
        inputs_dict.update({**self.inputs, **self.parameters})

        for comp in self.name_components:
            # Get other components like unique_identifier, role etc.
            part = None
            with suppress(Exception):
                # Search the data structure using the JMESPath expression
                part = jmespath.search(comp, inputs_dict)
            if part is not None and str(part):  # Ensure part is not None and not empty string
                name_parts.append(str(part))
            elif comp and comp not in inputs_dict and len(comp) <= 8:
                # If the component is not a JMESPath expression, but instead is
                # a short string, use it directly
                name_parts.append(comp)

        name = " ".join(name_parts).strip()
        self._agent_name = name or self.agent_id

        return self


class AgentVariants(AgentConfig):
    """A factory for creating Agent instance variants based on parameter combinations.

    Defines two types of variants:
    1. `parallel_variants`: Parameters whose combinations create distinct agent instances
       (e.g., different models). These agents can potentially run in parallel.
    2. `sequential_variants`: Parameters whose combinations define sequential tasks
       executed by *each* agent instance created from `parallel_variants`.

    Example:
    ```yaml
    any_key_name:
      id: ANALYST
      role: "Analyst"
      agent_obj: LLMAgent
      num_runs: 1
      variants:
        model: ["gpt-4", "claude-3"]    # Creates 2 parallel agent instances
      tasks:
        criteria: ["accuracy", "speed"] # Each agent instance runs 2 tasks sequentially
        temperature: [0.5, 0.8]         # Total 4 sequential tasks per agent
                                        # (accuracy/0.5, accuracy/0.8, speed/0.5, speed/0.8)

    Parameters
    ----------
        template: analyst               # parameter sets shared for each task
      inputs:
        results: othertask.outputs.results  # dynamic inputs mapped from other data
    ```

    """

    # --- Variant configuration: fields used to generate AgentConfig objects ---
    agent_obj: str = Field(  # Class name used by Hydra for instantiation.
        default="",
        description="The Python class name of the agent implementation to instantiate (e.g., 'Judge', 'LLMAgent').",
    )
    variants: dict = Field(default_factory=dict, description="Parameters for parallel agent variations.")
    tasks: dict = Field(default_factory=dict, description="Parameters for sequential tasks within each parallel variation.")
    num_runs: int = Field(default=1, description="Number of times to replicate each parallel variant configuration.")
    extra_params: list[str] = Field(default=[], description="Extra parameters to look for in runtime request.")
    # --- Variant configuration: fields used to generate AgentConfig objects ---

    def get_configs(self, params: RunRequest | None = None) -> list[tuple[type, AgentConfig]]:
        """Generates agent configurations based on parallel and sequential variants.
        """
        # Get static config (base attributes excluding variant fields)
        static_config = self.model_dump(
            exclude={
                "parallel_variants",
                "id", "unique_identifier",
                "sequential_variants",
                "num_runs",
                "parameters",
                "tasks",
            },
        )
        base_parameters = self.parameters.copy()  # Base parameters common to all

        # Get extra parameters passed in at runtime if requested
        if params:
            for key in self.extra_params:
                if key not in params.model_fields_set:
                    raise ValueError(f"Cannot find parameter {key} in runtime dict for agent {self.agent_id}.")
                base_parameters[key] = getattr(params, key)

        # And parameters passed in the request by the user
        request_params = params.parameters if params else {}

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

                    # Combine base parameters, parallel variant parameters, sequential task parameters, and parameters passed in from the UI in the request
                    # Order matters: task parameters overwrite parallel, parallel overwrite base
                    combined_params = {**base_parameters, **parallel_params, **task_params, **request_params}
                    cfg_dict["parameters"] = combined_params

                    # Create and add the AgentConfig instance
                    try:
                        # Filter dict so we only provide values that belong in the final AgentConfig instance
                        cfg_dict = {k: v for k, v in cfg_dict.items() if k in AgentConfig.model_fields}
                        agent_config_instance = AgentConfig(**cfg_dict)
                        generated_configs.append((agent_class, agent_config_instance))
                    except Exception as e:
                        logger.error(f"Error creating AgentConfig for {cfg_dict.get('role', 'unknown')} with parameters {combined_params}: {e}")
                        raise  # Re-raise by default

        if not generated_configs:  # Check if list is empty
            raise FatalError(f"Could not create any agent variant configs for {self.role} {self.agent_name}")

        return generated_configs
