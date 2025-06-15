"""Configuration models for Buttermilk components using Pydantic.

This module defines various Pydantic models that structure the configuration
for different aspects of the Buttermilk framework, such as data sources,
saving information, agent behaviors, and tracing. These models are typically
instantiated by Hydra based on YAML configuration files.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
)

import cloudpathlib  # For handling cloud storage paths
import jmespath  # For JSON query language processing
from google.cloud.bigquery.schema import SchemaField  # For BigQuery schema types
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
from shortuuid import uuid  # For generating short unique identifiers

from buttermilk._core.exceptions import FatalError  # Custom exceptions
from buttermilk._core.log import logger  # Centralized logger
from buttermilk.utils.utils import clean_empty_values, expand_dict  # Utility for dictionary expansion
from buttermilk.utils.validators import (
    convert_omegaconf_objects,  # Pydantic validators
    uppercase_validator,
)

from .constants import BQ_SCHEMA_DIR  # Default directory for BigQuery schemas
from .types import RunRequest  # Type for run requests

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
"""Specifies the cloud provider or storage type.

Allowed values:
    - "gcp": Google Cloud Platform (generic).
    - "bq": Google BigQuery.
    - "aws": Amazon Web Services.
    - "azure": Microsoft Azure.
    - "env": Environment variables.
    - "local": Local filesystem.
    - "gsheets": Google Sheets.
    - "vertex": Google Vertex AI.
"""


class CloudProviderCfg(BaseModel):
    """Base configuration for components interacting with cloud providers or specific storage types.

    Attributes:
        type (CloudProvider): The type of cloud provider or storage.
        model_config (ConfigDict): Pydantic model configuration.
            - `exclude_none`: True - Exclude fields with None values during serialization.
            - `arbitrary_types_allowed`: True - Allow arbitrary types.
            - `populate_by_name`: True - Allow population by field name (alias support).
            - `extra`: "allow" - Allow extra fields not explicitly defined.
            - `exclude_unset`: True - Exclude fields that were not explicitly set.
            - `include_extra`: True - Include extra fields during serialization.

    """

    type: CloudProvider = Field(description="The type of cloud provider or storage backend.")

    model_config = ConfigDict(
        exclude_none=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
        exclude_unset=True,
        include_extra=True,
    )


class SaveInfo(CloudProviderCfg):
    """Configuration for saving data to a specified destination.

    Inherits from `CloudProviderCfg` to specify the storage type and adds
    destination-specific details like paths, schema, and dataset/table names.

    Attributes:
        destination (str | cloudpathlib.AnyPath | None): The full path or identifier
            for the save location (e.g., file path, BigQuery table ID like
            `project.dataset.table`, GCS bucket URI).
        db_schema (str): The local name or path to a schema file (e.g., a JSON
            file defining a BigQuery schema). This path is resolved relative to
            `BQ_SCHEMA_DIR` if not an absolute path.
        dataset (str | None): The name of the dataset or equivalent grouping
            (e.g., a BigQuery dataset name, a directory path).
        _loaded_schema (list[SchemaField]): A private attribute to cache the loaded
            BigQuery schema once read from `db_schema`.

    """

    destination: str | cloudpathlib.AnyPath | None = Field(
        default=None,
        description="Full path or identifier for the save location (e.g., file path, "
                    "BigQuery table ID `project.dataset.table`, GCS URI).",
    )
    db_schema: str = Field(
        description="Local name or path to a schema file (e.g., JSON for BigQuery schema). "
                    "Resolved relative to BQ_SCHEMA_DIR if not absolute.",
    )
    dataset: str | None = Field(
        default=None,
        description="Name of the dataset or equivalent (e.g., BigQuery dataset, directory path).",
    )

    _loaded_schema: list[SchemaField] = PrivateAttr(default_factory=list)

    @field_validator("db_schema")
    @classmethod
    def file_must_exist(cls, v: str) -> str:
        """Validates that the `db_schema` file exists.

        Tries the path as is, then tries it relative to `BQ_SCHEMA_DIR`.

        Args:
            v: The path to the schema file.

        Returns:
            The validated, existing schema file path as a POSIX string.

        Raises:
            ValueError: If the schema file does not exist at the given path or
                relative to `BQ_SCHEMA_DIR`.

        """
        if v:
            try:
                path_v = Path(v)
                if path_v.exists() and path_v.is_file():
                    return path_v.as_posix()
            except Exception:  # Catch potential errors with Path construction
                pass  # Try next option

            # Try resolving relative to BQ_SCHEMA_DIR
            f = Path(BQ_SCHEMA_DIR) / v
            if f.exists() and f.is_file():
                return f.as_posix()
            raise ValueError(f"Schema file '{v}' does not exist or is not accessible.")
        return v  # Should not happen if field is required, but handles if it's optional

    @model_validator(mode="after")
    def check_destination(self) -> "SaveInfo":
        """Ensures that either `destination` or `dataset` is provided, unless type is 'gsheets'.

        Raises:
            ValueError: If neither `destination` nor `dataset` is set and the
                type is not 'gsheets' (as new sheets can be created implicitly).

        """
        if not self.destination and not self.dataset:
            if self.type == "gsheets":
                return self  # A new sheet can be created implicitly
            raise ValueError(
                "Save destination ambiguous: Either 'destination' (e.g., table ID, file path) "
                "or 'dataset' (e.g., BQ dataset, directory) must be provided.",
            )
        return self

    @computed_field
    @property
    def bq_schema(self) -> list[SchemaField]:
        """Loads and returns the BigQuery schema from the `db_schema` file path.

        The schema is loaded on first access and cached in `_loaded_schema`.

        Returns:
            list[SchemaField]: A list of `SchemaField` objects representing the
                BigQuery schema.

        """
        if not self._loaded_schema:
            from buttermilk import buttermilk as bm  # Lazy import to avoid circular deps

            # Ensure bm.bq is available; it might not be if only core is used.
            if not hasattr(bm, "bq") or bm.bq is None:
                logger.error("BigQuery client (bm.bq) not initialized. Cannot load schema.")
                # Depending on strictness, could raise an error or return empty.
                # For now, let it proceed, schema_from_json will likely fail informatively.

            try:
                self._loaded_schema = bm.bq.schema_from_json(self.db_schema)
            except Exception as e:
                logger.error(f"Failed to load BigQuery schema from {self.db_schema}: {e}")
                # Optionally re-raise or handle as appropriate
                # For now, it will return an empty list if loading fails and _loaded_schema remains empty.
        return self._loaded_schema


class DataSourceConfig(BaseModel):
    """Configuration for defining a data source for agents or other components.

    Specifies the type of data source, path or query details, filtering,
    joining, and other parameters for data retrieval and preparation.

    Attributes:
        max_records_per_group (int): Maximum records to process per group, if
            grouping is applied. -1 means no limit.
        type (Literal): The type of the data source.
            Allowed values: "job", "file", "bq" or "bigquery", "generator",
            "plaintext", "chromadb", "outputs" (from a previous step).
        path (str): Path to the data source (e.g., file path, BigQuery table ID,
            URL). Meaning depends on the `type`.
        glob (str): Glob pattern for matching files if `type` is "file".
        filter (Mapping[str, str | Sequence[str] | None] | None): Filtering
            criteria to apply to the data source (e.g., column-value pairs).
        join (Mapping[str, str] | None): Configuration for joining with other
            data sources. Keys are aliases, values are join conditions/targets.
        index (list[str] | None): Columns to use as an index for the data.
        agg (bool | None): Whether to aggregate results.
        group (Mapping[str, str] | None): Grouping configuration. Keys are new
            group column names, values are expressions or original column names.
        columns (Mapping[str, str | Mapping] | None): Specifies columns to select
            or transformations to apply to columns.
        last_n_days (int): For time-series data, specifies to retrieve data
            from the last N days.
        db (Mapping[str, str]): Database-specific connection parameters if `type`
            is a database type like "bq" or "chromadb".
        embedding_model (str): Name or path of the embedding model to use,
            particularly for "chromadb" or vector search.
        dimensionality (int): Dimensionality of embeddings, if applicable.
        persist_directory (str): Directory to persist data for "chromadb" or
            other file-based vector stores.
        collection_name (str): Name of the collection for "chromadb".
        model_config (ConfigDict): Pydantic model configuration.
            - `arbitrary_types_allowed`: False.
            - `populate_by_name`: True.
            - `exclude_none`: True.
            - `exclude_unset`: True.

    """

    max_records_per_group: int = Field(
        default=-1,
        description="Maximum records to process per group if grouping is applied. -1 for no limit.",
    )
    type: Literal[
        "job",
        "file",
        "bq",
        "bigquery",
        "generator",
        "plaintext",
        "chromadb",
        "outputs",
        "huggingface",
    ] = Field(description="The type of the data source.")
    path: str = Field(
        default="",
        description="Path to the data source (e.g., file path, BigQuery table ID, URL). Depends on 'type'.",
    )
    glob: str = Field(
        default="**/*",
        description="Glob pattern for matching files if type is 'file'.",
    )
    filter: Mapping[str, str | Sequence[str] | None] | None = Field(
        default_factory=dict,
        description="Filtering criteria (e.g., column-value pairs).",
    )
    join: Mapping[str, str] | None = Field(
        default_factory=dict,
        description="Configuration for joining with other data sources.",
    )
    index: list[str] | None = Field(
        default=None, description="Columns to use as an index.",
    )
    agg: bool | None = Field(
        default=False, description="Whether to aggregate results.",
    )
    group: Mapping[str, str] | None = Field(
        default_factory=dict,
        description="Grouping configuration (new_group_col: original_col_or_expr).",
    )
    columns: Mapping[str, str | Mapping] | None = Field(
        default_factory=dict,
        description=(
            "Column mapping for renaming data source fields to Record fields. "
            "Dictionary where keys are target Record field names and values are source field names. "
            "Can be empty ({}) if no field renaming is needed. "
            "Example: {'content': 'text', 'ground_truth': 'expected'}"
        ),
    )
    last_n_days: int = Field(
        default=7, description="For time-series data, retrieve from the last N days.",
    )
    db: Mapping[str, str] = Field(
        default_factory=dict,
        description="Database-specific connection parameters (e.g., for 'bq', 'chromadb').",
    )
    # BigQuery-specific fields
    project_id: str | None = Field(
        default=None,
        description="Google Cloud project ID for BigQuery data sources."
    )
    dataset_id: str | None = Field(
        default=None,
        description="BigQuery dataset ID."
    )
    table_id: str | None = Field(
        default=None,
        description="BigQuery table ID."
    )
    randomize: bool | None = Field(
        default=None,
        description="Whether to randomize BigQuery query results."
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Batch size for BigQuery operations."
    )
    embedding_model: str = Field(
        default="",
        description="Name or path of embedding model (for 'chromadb'/vector search).",
    )
    dimensionality: int = Field(
        default=-1, description="Dimensionality of embeddings, if applicable.",
    )
    persist_directory: str = Field(
        default="",
        description="Directory for persisting data (for 'chromadb' or file-based vector stores).",
    )
    collection_name: str = Field(
        default="", description="Name of the collection for 'chromadb'.",
    )
    name: str = Field(
        default="", description="Name/subset for HuggingFace datasets.",
    )
    split: str = Field(
        default="train", description="Split for HuggingFace datasets (e.g., 'train', 'test').",
    )

    model_config = ConfigDict(
        extra="ignore",  # Ignore extra fields not defined in the model
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )


class DataSouce(DataSourceConfig):
    """Alias for `DataSourceConfig` for backward compatibility or alternative naming.

    Refer to `DataSourceConfig` for detailed documentation.
    """


class BigQueryConfig(BaseModel):
    """Configuration for BigQuery-related operations.

    Provides default values for BigQuery operations including data loading,
    table creation, and migration utilities.

    Attributes:
        project_id (str | None): Google Cloud project ID. If None, will use
            the default project from credentials.
        dataset_id (str): BigQuery dataset ID. Defaults to "buttermilk".
        table_id (str): BigQuery table ID. Defaults to "records".
        randomize (bool): Whether to randomize query results. Defaults to True.
        batch_size (int): Batch size for operations. Defaults to 1000.
        auto_create (bool): Whether to auto-create tables if they don't exist.
        clustering_fields (list[str]): Default clustering fields for new tables.
    """

    project_id: str | None = Field(
        default=None,
        description="Google Cloud project ID. If None, uses default from credentials."
    )
    dataset_id: str = Field(
        default="buttermilk",
        description="BigQuery dataset ID."
    )
    table_id: str = Field(
        default="records",
        description="BigQuery table ID."
    )
    randomize: bool = Field(
        default=True,
        description="Whether to randomize query results."
    )
    batch_size: int = Field(
        default=1000,
        ge=1,
        description="Batch size for operations."
    )
    auto_create: bool = Field(default=True, description="Whether to auto-create tables if they don't exist.")
    clustering_fields: list[str] = Field(
        default=["record_id", "dataset_name"],
        description="Default clustering fields for new tables."
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )

    @model_validator(mode="after")
    def set_project_id_from_env(self) -> "BigQueryConfig":
        """Set project_id from environment if not already set."""
        import os
        self.project_id = self.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        return self


class ToolConfig(BaseModel):
    """Configuration for a tool (function) that an agent can use.

    Defines the tool's description, how to invoke it (tool_obj), and any
    data sources it might require.

    Attributes:
        description (str): A textual description of what the tool does. This is
            often used in prompts for LLMs to understand when to use the tool.
        tool_obj (str): The identifier or path to the actual Python callable or
            object that implements the tool's logic.
        data (Mapping[str, DataSourceConfig]): A mapping where keys are names/aliases
            for data sources and values are `DataSourceConfig` objects defining
            how to load that data. This allows tools to dynamically access data.

    """

    description: str = Field(
        default="",
        description="Textual description of the tool's purpose and capabilities, often used for LLM prompts.",
    )
    tool_obj: str = Field(
        default="",
        description="Identifier or path to the Python callable/object implementing the tool.",
    )
    data: Mapping[str, DataSourceConfig] = Field(
        default_factory=dict,  # Changed from list to dict as per typical usage patterns
        description="Specifications for data sources the tool should load, keyed by a name/alias.",
    )

    def get_functions(self) -> list[Any]:
        """Generates function definitions for this tool, typically for an LLM.

        This method should be implemented by subclasses or specific tool integrations
        to return a list of function definitions in a format expected by the
        consuming system (e.g., OpenAI function calling schema).

        Returns:
            list[Any]: A list of function definitions.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError("Subclasses must implement get_functions to define tool structure.")

    async def _run(self, **kwargs: Any) -> list[Any] | None:
        """Executes the tool's core logic.

        This asynchronous method should be implemented by subclasses to perform
        the actual work of the tool.

        Args:
            **kwargs: Arbitrary keyword arguments that might be passed to the tool
                during execution, often representing the parameters the tool needs.

        Returns:
            list[Any] | None: The result of the tool's execution, typically a list
            of outputs or None. The exact type can vary.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError("Subclasses must implement _run to execute tool logic.")


class Tracing(BaseModel):
    """Configuration for tracing agent and system activities.

    Specifies whether tracing is enabled, the provider to use (e.g., Langfuse, Weave),
    API keys, and other provider-specific settings.

    Attributes:
        enabled (bool): If `True`, tracing is enabled for operations.
        api_key (str): API key for the tracing provider.
        provider (str): Name of the tracing provider (e.g., "langfuse", "weave").
        endpoint (str | None): Optional endpoint URL for the tracing provider,
            if different from the default.
        otlp_headers (Mapping | None): Optional OTLP (OpenTelemetry Protocol)
            headers for providers that support it.

    """

    enabled: bool = Field(default=False, description="Enable or disable tracing.")
    api_key: str = Field(default="", description="API key for the tracing provider.")
    provider: str = Field(default="", description="Name of the tracing provider (e.g., 'langfuse', 'weave').")
    endpoint: str | None = Field(default=None, description="Optional custom endpoint for the tracing provider.")
    otlp_headers: Mapping[str, str] | None = Field(  # Made value type str for typical headers
        default_factory=dict, description="Optional OTLP headers for providers supporting it.",
    )


# --- Agent Configuration ---
class AgentConfig(BaseModel):
    """Base Pydantic model defining the configuration for Buttermilk agents.

    This model is typically loaded and instantiated by Hydra from YAML configuration
    files. It includes core agent identification (like `agent_id`, `role`),
    behavioral parameters, definitions for tools the agent can use, data sources
    it might access, and mappings for processing inputs and outputs.

    Attributes:
        agent_id (str): A unique identifier for the agent instance. Automatically
            generated based on `role` and `unique_identifier` if not provided.
        role (str): The functional role this agent plays in a workflow (e.g.,
            'DATA_EXTRACTOR', 'SUMMARIZER'). Automatically converted to uppercase.
        description (str): A human-readable explanation of the agent's purpose
            and capabilities.
        tools (list[ToolConfig]): A list of `ToolConfig` objects, defining the
            tools (functions) available to this agent.
        data (Mapping[str, DataSourceConfig]): Configuration for data sources the
            agent might need to access, keyed by a descriptive name.
            Serialized as `mapping_data`.
        parameters (dict[str, Any]): Agent-specific configuration parameters that
            control its behavior (e.g., LLM model name, specific template names,
            thresholds for decision-making).
        inputs (dict[str, Any]): Defines mappings for how incoming data (from
            messages or other sources) should populate the agent's input context.
            Uses JMESPath for flexible data extraction and transformation.
            Serialized as `mapping_inputs`.
        outputs (dict[str, Any]): Defines mappings for how the agent's results
            should be structured or transformed before being sent out.
            (Currently, its usage might be pending full implementation).
            Serialized as `mapping_outputs`.
        name_components (list[str]): A list of attribute names or JMESPath expressions
            used to construct the human-friendly `agent_name`. Defaults to
            `["role", "unique_identifier"]`.
        model_config (ConfigDict): Pydantic model configuration.
            - `extra`: "allow" - Allows extra fields not explicitly defined, useful with Hydra.
            - `arbitrary_types_allowed`: False.
            - `populate_by_name`: True.
        unique_identifier (str): A short, automatically generated unique ID component
            used in constructing `agent_id` and `agent_name`.
        _agent_name (str): Private attribute storing the generated human-friendly name.
        _validate_parameters: Pydantic field validator to convert OmegaConf objects
            (like DictConfig) in `parameters`, `inputs`, and `outputs` to
            standard Python dicts before further validation.

    """

    # Core Identification
    agent_id: str = Field(
        default="",  # Will be generated if not provided
        description="Unique identifier for the agent instance. Automatically generated from 'role' and 'unique_identifier' if empty.",
        validate_default=True,  # Ensures _generate_id_and_name runs even if id is not explicitly set
    )
    role: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        description="The functional role this agent plays in the workflow (e.g., 'JUDGE', 'SUMMARIZER'). Converted to uppercase.",
        min_length=1,  # Ensure role is not empty
    )
    description: str = Field(
        default="",
        description="A brief human-readable explanation of the agent's purpose and capabilities.",
    )
    name: str = Field(
        default="",
        description="A human-readable name for the agent, used in UIs and displays.",
        min_length=0,  # Allow empty name, but validate it's a string
    )

    # Behavior & Connections
    tools: list[ToolConfig] = Field(
        default_factory=list,
        description="Configuration for tools (functions) that the agent can potentially use.",
    )
    data: Mapping[str, DataSourceConfig] = Field(
        default_factory=dict,
        description="Configuration for data sources the agent might need access to, keyed by a descriptive name.",
        alias="mapping_data",  # For serialization/deserialization consistency if needed
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration parameters (e.g., LLM model name, template name, decision thresholds).",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Defines JMESPath mappings for how incoming data should populate the agent's input context.",
        alias="mapping_inputs",
    )
    outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Defines mappings for how the agent's results should be structured or transformed. (Usage may be evolving).",
        alias="mapping_outputs",
    )

    name_components: list[str] = Field(
        default=["role", "unique_identifier"],
        description="List of attribute names or JMESPath expressions to construct the 'agent_name'.",
        exclude=False,  # Ensure it's included in model_dump etc.
    )

    # Pydantic Model Configuration
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=False,
        populate_by_name=True,  # Allows using alias for population
        validate_assignment=True,  # Ensures validators run on assignment
    )

    # Private Attributes
    unique_identifier: str = Field(
        default_factory=lambda: uuid()[:6],  # Generates a short unique ID
        description="A short unique ID component, auto-generated if not provided.",
        exclude=True,  # Typically not set directly by user, internal detail
    )
    _agent_name: str = PrivateAttr()

    # Field Validators
    _validate_parameters = field_validator(
        "parameters", "inputs", "outputs", "data",  # Added data
        mode="before",
    )(convert_omegaconf_objects)

    @computed_field(repr=False)  # repr=False to avoid circularity if used in name_components
    @property
    def agent_name(self) -> str:
        """A human-friendly name for the agent instance.

        This name is dynamically constructed based on the `name_components`
        attribute, which can include the agent's `role`, `unique_identifier`,
        or other values extracted via JMESPath from its configuration
        (`inputs` and `parameters`).

        Returns:
            str: The generated human-friendly name for the agent.

        """
        # Ensure _agent_name is initialized, _generate_id_and_name might not have run if accessed early
        if not hasattr(self, "_agent_name") or not self._agent_name:
            self._generate_id_and_name()  # Call the combined method
        return self._agent_name

    @model_validator(mode="after")
    def _generate_id_and_name(self) -> Self:
        """Generates/updates `agent_id` and `_agent_name` for the agent instance.

        This validator runs after initial model creation and on subsequent
        assignments if `validate_assignment` is True. It ensures that:
        - `agent_id` is consistently derived from `role` and `unique_identifier`.
          If `agent_id` is provided but differs, it will be updated to the canonical derived ID.
        - `_agent_name` (accessed via `agent_name` property) is constructed based
          on `name_components`, allowing for dynamic naming using JMESPath
          expressions on the agent's configuration.

        This method is designed to be idempotent.

        Returns:
            Self: The instance of AgentConfig with `agent_id` and `_agent_name` populated/updated.

        """
        # Part 1: Generate/Update agent_id
        current_role = self.role
        current_unique_id = self.unique_identifier

        intended_agent_id = f"{current_role}-{current_unique_id}" if current_role else current_unique_id

        if self.agent_id != intended_agent_id:
            if self.agent_id and self.agent_id != "":
                logger.debug(
                    f"Agent ID for role '{current_role}' (uid: {current_unique_id}) is being updated. "
                    f"Old/Provided ID: '{self.agent_id}', New Canonical ID: '{intended_agent_id}'. ",
                )
            object.__setattr__(self, "agent_id", intended_agent_id)  # Use object.__setattr__ to bypass Pydantic validation cycle here

        # Part 2: Generate agent_name
        name_parts = []

        # Construct the context for JMESPath search manually to avoid recursion.
        # This context should contain fields that name_components might refer to,
        # respecting aliases and excluding None values. Ensure 'unique_identifier'
        # and the canonical 'agent_id' are in the context.
        context_for_jmespath = {**self.model_dump(include={"unique_identifier", "agent_id", "role"}), **self.parameters}

        for comp_path in self.name_components:
            part = None
            try:
                part = jmespath.search(comp_path, context_for_jmespath)
            except Exception:
                part = None

            if part is not None and str(part).strip():
                name_parts.append(str(part).strip())
            # Fallback for literal short strings if JMESPath fails/not applicable
            # and comp_path itself is not a key that yielded a value from context_for_jmespath
            # and len(comp_path) <= 4:
            elif part is None and comp_path and comp_path not in context_for_jmespath and len(comp_path) <= 4:
                name_parts.append(comp_path)

        name = " ".join(filter(None, name_parts)).strip()  # Filter None before join
        # Use object.__setattr__ for private attributes to avoid triggering validators if not desired
        object.__setattr__(self, "_agent_name", name or self.agent_id)  # Fallback to the canonical agent_id

        return self

    def get_display_name(self) -> str:
        """Get the display name for this agent.
        
        Returns the agent_name which is consistently formatted across UIs.
        LLM agents may override this to include model information.
        
        Returns:
            str: The display name for the agent
        """
        return self.agent_name


class AgentVariants(AgentConfig):
    """A factory for creating multiple `AgentConfig` instances (variants).
     
    based on
    parameter combinations. This is useful for running experiments with different
    agent settings or for creating ensembles of agents.

    It extends `AgentConfig` to inherit base configuration fields and adds
    specific fields for defining variant parameters.

    Variants can be defined in two main ways:
    1.  **`variants` (Parallel Variants)**: A dictionary where keys are parameter names
        and values are lists of possible settings. Combinations of these create
        distinct agent configurations that can potentially run in parallel.
        Example: `variants: {model: ["gpt-4", "claude-3"], temperature: [0.7, 0.9]}`
        would generate four base configurations.
    2.  **`tasks` (Sequential Variants/Tasks)**: Similar to `variants`, but these
        combinations define sequential tasks or sub-configurations to be executed
        by *each* agent instance created from the parallel `variants`.
        Example: `tasks: {prompt_style: ["detailed", "concise"]}`. If there were
        two parallel variants, each would run these two sequential tasks.

    The `num_runs` attribute replicates each parallel variant configuration a
    specified number of times, useful for repeated trials.

    Attributes:
        agent_obj (str): The Python class name of the agent implementation to
            instantiate (e.g., 'LLMAgent', 'SummarizationAgent'). This class
            should be registered in `AgentRegistry`.
        variants (dict): Dictionary defining parameters for parallel agent variations.
            Keys are parameter names, values are lists of settings for that parameter.
        tasks (dict): Dictionary defining parameters for sequential tasks or
            sub-configurations within each parallel variation.
        num_runs (int): Number of times to replicate each parallel variant
            configuration.
        extra_params (list[str]): A list of parameter names that should be sourced
            from the runtime `RunRequest` and merged into the agent's parameters.
            This allows for dynamic configuration at execution time.

    """

    agent_obj: str = Field(
        description="The Python class name of the agent implementation to instantiate (e.g., 'LLMAgent'). Must be registered in AgentRegistry.",
        min_length=1,  # Ensure it's not empty
    )
    variants: dict[str, list[Any]] = Field(  # More specific type hint
        default_factory=dict,
        description="Parameters for parallel agent variations (e.g., {'model': ['gpt-4', 'claude-3']}).",
    )
    tasks: dict[str, list[Any]] = Field(  # More specific type hint
        default_factory=dict,
        description="Parameters for sequential tasks within each parallel variation (e.g., {'prompt_style': ['concise', 'detailed']}).",
    )
    num_runs: int = Field(
        default=1,
        ge=1,  # Ensure num_runs is at least 1
        description="Number of times to replicate each parallel variant configuration.",
    )
    extra_params: list[str] = Field(
        default_factory=list,
        description="List of parameter names to source from the runtime RunRequest and merge into agent parameters.",
    )

    @field_validator("agent_obj")
    @classmethod
    def validate_agent_obj(cls, v: str) -> str:
        """Validate that agent_obj is not empty and exists in the AgentRegistry.

        Args:
            v: The agent_obj value to validate

        Returns:
            str: The validated agent_obj value

        Raises:
            ValueError: If agent_obj is empty or not found in registry
        """
        if not v or not v.strip():
            raise ValueError("agent_obj cannot be empty. Must specify a valid agent class name.")

        # Optional: Check if the agent exists in registry (may cause circular imports in some cases)
        # For now, we'll just ensure it's not empty. The registry check happens in get_configs()
        return v.strip()

    def get_configs(self, params: RunRequest | None = None, flow_default_params: dict = {}) -> list[tuple[type[Any], AgentConfig]]:
        """Generates a list of agent configurations based on defined variants and tasks.

        This method expands the `variants` and `tasks` dictionaries to create all
        possible combinations of parameters. Each combination, along with base
        parameters and any runtime `extra_params`, forms a distinct `AgentConfig`.
        The `num_runs` setting further replicates these configurations.

        Args:
            params (RunRequest | None): An optional `RunRequest` object containing
                runtime parameters. If `extra_params` are defined for this
                `AgentVariants` instance, their values are sourced from this
                `RunRequest`.

        Returns:
            list[tuple[type[Any], AgentConfig]]: A list of tuples, where each tuple
            contains the agent class (obtained from `AgentRegistry` via `agent_obj`)
            and the generated `AgentConfig` instance.

        Raises:
            ValueError: If an `extra_param` is specified but not found in the
                provided `RunRequest` `params`.
            FatalError: If no agent configurations can be generated (e.g., due to
                empty variants and tasks without a base configuration).
            TypeError: If `agent_obj` is not found in the `AgentRegistry`.

        """
        static_config_dict = self.model_dump(
            exclude={
                "agent_obj",  # Exclude agent_obj as it's used to get the class
                "variants",
                "tasks",
                "num_runs",
                "extra_params",
                # Also exclude fields that are part of AgentConfig's identity if they are recalculated
                "agent_id", "unique_identifier", "_agent_name",
                # Keep 'parameters' to use as base, but it will be overwritten/merged
            },
            exclude_none=True,  # Exclude None values to avoid overriding defaults in AgentConfig
        )
        static_config_dict = clean_empty_values(static_config_dict)

        # Ensure 'parameters' exists and is a dict, even if empty from model_dump
        base_parameters = static_config_dict.pop("parameters", {})
        if base_parameters is None:
            base_parameters = {}

        # Merge extra parameters from RunRequest if provided
        if params and self.extra_params:
            for key in self.extra_params:
                if not hasattr(params, key) or getattr(params, key) is None:  # Check if param exists in RunRequest
                    raise ValueError(
                        f"Required extra_param '{key}' not found or is None in RunRequest for agent variant '{self.agent_id or self.role}'.")
                base_parameters[key] = getattr(params, key)

        # Merge parameters from the RunRequest.parameters (user-provided overrides)
        if params and params.parameters:
            base_parameters.update(clean_empty_values(params.parameters))

        from buttermilk._core.variants import AgentRegistry  # Lazy import
        try:
            agent_class = AgentRegistry.get(self.agent_obj)
            if agent_class is None:  # AgentRegistry.get might return None if not found and not raising
                raise TypeError(f"Agent class '{self.agent_obj}' not found in AgentRegistry.")
        except KeyError:  # Assuming AgentRegistry might raise KeyError
            raise TypeError(f"Agent class '{self.agent_obj}' not found in AgentRegistry.")

        # Filter out variant parameters that are overridden by RunRequest parameters
        filtered_variants = self.variants.copy() if self.variants else {}
        if params and params.parameters:
            # Remove any variant keys that are explicitly set in params.parameters
            for key in params.parameters.keys():
                filtered_variants.pop(key, None)

        parallel_variant_combinations = expand_dict(clean_empty_values(filtered_variants)) if filtered_variants else [{}]

        # Only use explicitly defined tasks, not flow default parameters
        sequential_task_sets = expand_dict(clean_empty_values(self.tasks)) if self.tasks else [{}]

        generated_configs: list[tuple[type[Any], AgentConfig]] = []
        for _ in range(self.num_runs):  # Loop for num_runs
            for parallel_params in parallel_variant_combinations:
                for task_params in sequential_task_sets:
                    # Start with the static parts of AgentVariants config
                    current_config_dict = static_config_dict.copy()

                    # Combine parameters: flow defaults, then base (agent + RunRequest), then parallel, then task-specific.
                    # This order defines precedence - later values override earlier ones.
                    final_params = {**flow_default_params, **base_parameters, **parallel_params, **task_params}
                    current_config_dict["parameters"] = clean_empty_values(final_params)

                    # Ensure all necessary fields for AgentConfig are present or defaulted
                    # Role and description might come from static_config_dict or need defaults
                    current_config_dict.setdefault("role", self.role or "VARIANT_AGENT")
                    current_config_dict.setdefault("description", self.description or "Generated variant agent")

                    # Explicitly remove fields not in AgentConfig before instantiation
                    # This is safer than relying solely on AgentConfig.model_config['extra'] = 'ignore'
                    # if AgentConfig itself doesn't have 'extra':'allow' or if strictness is desired.
                    valid_agent_config_fields = AgentConfig.model_fields.keys()
                    filtered_cfg_dict = {
                        k: v for k, v in current_config_dict.items() if k in valid_agent_config_fields
                    }

                    # Add 'parameters' back as it's a valid field in AgentConfig
                    filtered_cfg_dict["parameters"] = final_params

                    try:
                        agent_config_instance = AgentConfig(**filtered_cfg_dict)
                        generated_configs.append((agent_class, agent_config_instance))
                    except Exception as e:
                        logger.error(msg :=
                            f"Error creating AgentConfig for role '{filtered_cfg_dict.get('role', 'unknown')}' "
                            f"with parameters {final_params}: {e}",
                        )
                        raise FatalError(msg) from e

        if not generated_configs:
            logger.warning(f"No agent configurations were generated for AgentVariants: {self.agent_id or self.role}. "
                           f"This might be due to empty 'variants' and 'tasks' with num_runs=0, or misconfiguration.")
            # Depending on desired behavior, could raise FatalError or return empty list.
            # Current behavior: returns empty list, which might be handled by caller.
            # However, the original code raised FatalError, so let's keep that.
            logger.error(msg := f"Could not create any agent variant configs for {self.role or self.agent_name}")
            raise FatalError(msg)

        return generated_configs
