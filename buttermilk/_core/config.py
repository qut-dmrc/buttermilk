from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    List,
    Literal,
)

import cloudpathlib
from google.cloud.bigquery.schema import SchemaField
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from .types import SessionInfo
from .defaults import BQ_SCHEMA_DIR

from buttermilk.utils.validators import convert_omegaconf_objects, lowercase_validator  # Pydantic validators

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
            self._loaded_schema= bm.bq.schema_from_json(self.db_schema)
        return self._loaded_schema


class DataSourceConfig(BaseModel):
    name: str
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
    role: str = Field(default="")
    description: str = Field(default="")
    tool_obj: str = Field(default="")

    data: list[DataSourceConfig] = Field(
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
    run_info: SessionInfo = Field(
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
    role: Annotated[str, AfterValidator(lowercase_validator)] = Field(
        ...,  # Role is required
        description="The functional role this agent plays in the workflow (e.g., 'judge', 'conductor'). Must be lowercase.",
    )
    name: str = Field(default="", description="A human-friendly name for the agent instance, often including the role and a unique ID.")
    description: str = Field(
        ...,  # Description is required
        description="A brief explanation of the agent's purpose and capabilities.",
    )
    agent_obj: str = Field(  # Class name used by Hydra for instantiation.
        default="",
        description="The Python class name of the agent implementation to instantiate (e.g., 'Judge', 'LLMAgent').",
        exclude=True,  # Excluded from model serialization, as it's primarily for loading.
    )

    # Behavior & Connections
    tools: list[ToolConfig] = Field(
        default_factory=list,  # Use factory for mutable default
        description="Configuration for tools (functions) that the agent can potentially use.",
    )
    data: list[DataSourceConfig] = Field(
        default_factory=list,  # Use factory for mutable default
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
    _id: str = PrivateAttr(default_factory=lambda: uuid()[:6])  # Short unique ID component.
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
        self.id = f"{self.role}-{self._id}"

        # Set the base_name from the initially provided 'name' if not already set.
        if not self.base_name:
            if not self.name:  # Ensure a base name was provided in the config.
                raise ValueError("AgentConfig requires a human-friendly 'name' field in configuration.")
            self.base_name = self.name  # Store the original human-friendly name part.

        # Construct the final display 'name' from specified components.
        name_parts = [getattr(self, comp, None) for comp in self._name_components]
        self.name = " ".join([str(part) for part in name_parts if part])  # Join non-None parts.
        return self
