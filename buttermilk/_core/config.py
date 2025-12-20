"""Configuration models for Buttermilk components using Pydantic.

This module defines various Pydantic models that structure the configuration
for different aspects of the Buttermilk framework, such as data sources,
saving information, agent behaviors, and tracing. These models are typically
instantiated by Hydra based on YAML configuration files.
"""

import copy
from collections.abc import Mapping
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
)

import jmespath  # For JSON query language processing

# BigQuery import - now a core dependency
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
from buttermilk.utils.utils import (
    clean_empty_values,
    expand_dict,
)  # Utility for dictionary expansion
from buttermilk.utils.validators import (
    convert_omegaconf_objects,  # Pydantic validators
    make_class_import_validator,
    uppercase_validator,
)

from .types import RunRequest  # Type for run requests

CloudProvider = Literal[
    "gcp",
    "bq",
    "aws",
    "azure",
    "env",
    "local",
    "gsheets",
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
"""


class CloudProviderCfg(BaseModel):
    """Base configuration for components interacting with cloud providers or specific storage types.

    This class handles different cloud provider types with their specific requirements.
    Uses dynamic validation based on the provider type rather than rigid field requirements.

    Attributes:
        type (CloudProvider): The type of cloud provider or storage.
        project_id (str | None): GCP project ID (required for GCP-based providers).
        location (str | None): Cloud region/location (required for specific services).
        model_config (ConfigDict): Pydantic model configuration.

    """

    type: CloudProvider = Field(
        description="The type of cloud provider or storage backend."
    )
    project_id: str | None = Field(default=None, description="Cloud project ID")
    location: str | None = Field(default=None, description="Cloud region/location")

    model_config = ConfigDict(
        exclude_none=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
        exclude_unset=True,
        include_extra=True,
    )


class LoggerConfig(CloudProviderCfg):
    """Specialized cloud provider configuration for logging with strict validation.

    This extends CloudProviderCfg with specific validation requirements
    for logging configurations, which need both project and location for GCP.
    """

    @model_validator(mode="after")
    def validate_logger_requirements(self) -> "LoggerConfig":
        """Validate that required fields are present for logger configurations."""
        if self.type == "gcp":
            missing_fields = []
            if not self.project_id:
                missing_fields.append("project_id")
            if not self.location:
                missing_fields.append("location")

            if missing_fields:
                fields_str = ", ".join(missing_fields)
                raise ValueError(
                    f"GCP logger configuration requires these fields: {fields_str}. Please ensure your logger_cfg includes all required fields.",
                )
        elif self.type == "local":
            # Local logging has no requirements
            pass
        else:
            raise ValueError(
                f"Unsupported logger type: '{self.type}'. Supported logger types are: 'gcp', 'local'",
            )

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
        data (Mapping[str, StorageConfig]): A mapping where keys are names/aliases
            for data sources and values are `StorageConfig` objects defining
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
    data: Mapping[str, Any] = Field(
        default_factory=dict,  # Changed from list to dict as per typical usage patterns
        description="Specifications for storage backends the tool should load, keyed by a name/alias. Use BaseStorageConfig types.",
    )

    @field_validator("data", mode="after")
    @classmethod
    def validate_storage_configs(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        """Convert YAML dictionaries to BaseStorageConfig objects."""
        if not v:
            return v

        from buttermilk._core.storage_config import StorageFactory

        converted = {}
        for key, config in v.items():
            if isinstance(config, dict):
                # Convert dict from YAML to appropriate BaseStorageConfig subclass
                try:
                    converted[key] = StorageFactory.create_config(config)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert storage config '{key}': {e}. Using raw dict."
                    )
                    converted[key] = config
            else:
                # Already a BaseStorageConfig object or other type
                converted[key] = config
        return converted

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
        raise NotImplementedError(
            "Subclasses must implement get_functions to define tool structure."
        )

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
        raise NotImplementedError(
            "Subclasses must implement _run to execute tool logic."
        )


class Tracing(BaseModel):
    """Configuration for tracing agent and system activities.

    for each tracing provider, specifies whether tracing is enabled,
    API keys, and other provider-specific settings.

    Attributes:
        enabled (bool): If `True`, tracing is enabled for operations.
        api_key (str): API key for the tracing provider.
        endpoint (str | None): Optional endpoint URL for the tracing provider,
            if different from the default.
        otlp_headers (Mapping | None): Optional OTLP (OpenTelemetry Protocol)
            headers for providers that support it.
        project_id (str | None): Optional project ID for the tracing provider.

    """

    enabled: bool = Field(default=False, description="Enable or disable tracing.")
    api_key: str | None = Field(
        default=None, description="API key for the tracing provider."
    )
    endpoint: str | None = Field(
        default=None, description="Optional custom endpoint for the tracing provider."
    )
    otlp_headers: Mapping[str, str] | None = (
        Field(  # Made value type str for typical headers
            default_factory=dict,
            description="Optional OTLP headers for providers supporting it.",
        )
    )
    project_id: str | None = Field(
        default=None, description="Optional project ID for the tracing provider."
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
        output_model (type[pydantic.BaseModel] | None): Optional Pydantic model
                for structured output parsing.
        tools (dict[str, ToolConfig]): A dictionary of tool configurations, defining the
            tools (functions) available to this agent, keyed by tool name.
        data (Mapping[str, StorageConfig]): Configuration for data sources the
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
        description="Unique identifier for the agent instance. Automatically generated as UUID if empty.",
        validate_default=True,  # Ensures _generate_id_and_name runs even if id is not explicitly set
    )
    role: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        default="",  # Should typically be set in YAML
        description="The functional role this agent plays in the workflow (e.g., 'JUDGE', 'SUMMARIZER'). Converted to uppercase.",
    )
    description: str = Field(
        default="",
        description="A brief human-readable explanation of the agent's purpose and capabilities.",
    )

    # Behavior & Connections
    output_model: type[BaseModel] | None = Field(
        default=None, description="Pydantic model for structured output parsing."
    )

    tools: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for tools (functions) that the agent can potentially use, keyed by tool name. Can be ToolConfig objects or direct tool instances.",
    )
    data: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Configuration for storage backends the agent might need access to, keyed by a descriptive name. Use BaseStorageConfig types.",
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
    record: str | None = Field(
        default=None,
        description="JMESPath expression for extracting record from flow state. "
                    "Maps directly to AgentInput.record field. "
                    "Example: '[FETCH.outputs]||*.record' (tries FETCH first, falls back to other sources)",
    )
    context: str | None = Field(
        default=None,
        description="JMESPath expression for extracting conversation context from flow state. "
                    "Maps directly to AgentInput.context field.",
    )

    name_components: list[str] = Field(
        default=["role", "agent_id"],
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
    _agent_name: str = PrivateAttr()
    _unique_identifier: str = PrivateAttr(
        default=None
    )  # Used in constructing `agent_id` and `agent_name`.

    # Field Validators
    _validate_parameters = field_validator(
        "parameters",
        "inputs",
        "outputs",
        "data",  # Added data
        mode="before",
    )(convert_omegaconf_objects)

    _validate_output_model = field_validator("output_model", mode="before")(
        make_class_import_validator(BaseModel)
    )

    @field_validator("data", mode="after")
    @classmethod
    def validate_storage_configs(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        """Convert YAML dictionaries to BaseStorageConfig objects."""
        if not v:
            return v

        from buttermilk._core.storage_config import StorageFactory

        converted = {}
        for key, config in v.items():
            if isinstance(config, dict):
                # Convert dict from YAML to appropriate BaseStorageConfig subclass
                try:
                    converted[key] = StorageFactory.create_config(config)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert storage config '{key}': {e}. Using raw dict."
                    )
                    converted[key] = config
            else:
                # Already a BaseStorageConfig object or other type
                converted[key] = config
        return converted

    @computed_field(
        repr=False
    )  # repr=False to avoid circularity if used in name_components
    @property
    def agent_name(self) -> str:
        """A human-friendly name for the agent instance.

        This name is dynamically constructed based on the `name_components`
        attribute, which can include the agent's `role`, `agent_id`,
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
        """Generates `agent_id` and `_agent_name` for the agent instance.

        This validator runs after initial model creation and on subsequent
        assignments if `validate_assignment` is True. It ensures that:
        - `agent_id` is generated as a UUID if not already provided.
        - `_agent_name` (accessed via `agent_name` property) is constructed based
          on `name_components`, allowing for dynamic naming using JMESPath
          expressions on the agent's configuration.

        This method is designed to be idempotent and conditional.

        Returns:
            Self: The instance of AgentConfig with `agent_id` and `_agent_name` populated/updated.

        """
        # Part 1: Generate agent_id only if not already set (conditional)
        if not self.agent_id or not self.agent_id.strip():
            # Generate a simple UUID
            self._unique_identifier = str(uuid()[:6]).upper()
            generated_id = f"{self.role}-{self._unique_identifier}"
            # Use object.__setattr__ to bypass Pydantic validation cycle here
            object.__setattr__(self, "agent_id", generated_id)  # noqa: PLC2801

        # Part 2: Generate agent_name
        name_parts = []

        # Construct the context for JMESPath search manually to avoid recursion.
        # This context should contain fields that name_components might refer to,
        # respecting aliases and excluding None values. Ensure the current
        # 'agent_id' is in the context.
        context_for_jmespath = {
            **self.model_dump(include={"agent_id", "role"}),
            **self.parameters,
        }

        # Manually add unique_identifier as a special case (it's not in parameters)
        context_for_jmespath["unique_identifier"] = self._unique_identifier

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
            elif (
                part is None
                and comp_path
                and comp_path not in context_for_jmespath
                and len(comp_path) <= 4
            ):
                name_parts.append(comp_path)

        name = " ".join(filter(None, name_parts)).strip()  # Filter None before join

        # Use object.__setattr__ for private attributes to avoid triggering validators if not desired
        object.__setattr__(
            self, "_agent_name", name or self.agent_id
        )  # Fallback to the canonical agent_id

        return self

    @model_validator(mode="after")
    def _validate_no_record_context_in_inputs(self) -> Self:
        """Ensure record/context aren't specified in both top-level AND inputs dict."""
        if self.inputs:
            if "record" in self.inputs and self.record is not None:
                raise ValueError(
                    "Configuration error: 'record' field is ambiguous. "
                    "Cannot specify 'record' in both top-level field AND inputs dict. "
                    "Use only the top-level 'record:' field."
                )
            if "context" in self.inputs and self.context is not None:
                raise ValueError(
                    "Configuration error: 'context' field is ambiguous. "
                    "Cannot specify 'context' in both top-level field AND inputs dict. "
                    "Use only the top-level 'context:' field."
                )
        return self


class AgentVariants(AgentConfig):
    """A factory for creating multiple `AgentConfig` instances (variants).

    Creates variants based on parameter combinations. Allows flows to produce
    results with multiple different agent settings or to create ensembles of agents.

    It extends `AgentConfig` to inherit base configuration fields and adds
    specific fields for defining variant parameters.

    The `variants` dictionary defines parameter combinations. Keys are parameter names
    and values are lists of possible settings. All combinations are expanded to create
    distinct agent configurations that run in parallel.
    Example: `variants: {model: ["gpt-4", "claude-3"], temperature: [0.7, 0.9]}`
    would generate four agent configurations.

    The `num_runs` attribute replicates each variant configuration a specified
    number of times, useful for repeated trials.

    Attributes:
        agent_obj (str): The Python class name of the agent implementation to
            instantiate (e.g., 'LLMAgent', 'SummarizationAgent'). This class
            should be registered in `AgentRegistry`.
        variants (dict): Dictionary defining parameters for agent variations.
            Keys are parameter names, values are lists of settings for that parameter.
        num_runs (int): Number of times to replicate each variant configuration.
        extra_params (list[str]): A list of parameter names that should be sourced
            from the runtime `RunRequest` and merged into the agent's parameters.
            This allows for dynamic configuration at execution time.

    """

    agent_obj: str = Field(
        description="The Python class name of the agent implementation to instantiate (e.g., 'LLMAgent'). Must be registered in AgentRegistry.",
    )
    variants: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameters for agent variations (e.g., {'model': ['gpt-4', 'claude-3']}).",
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

    def get_configs(
        self, params: RunRequest | None = None, flow_default_params: dict = {}
    ) -> list[tuple[type[Any], AgentConfig]]:
        """Generates a list of agent configurations based on defined variants.

        This method expands the `variants` dictionary to create all possible
        combinations of parameters. Each combination, along with base parameters
        and any runtime `extra_params`, forms a distinct `AgentConfig`.
        The `num_runs` setting further replicates these configurations.

        Args:
            params (RunRequest | None): An optional `RunRequest` object containing
                runtime parameters. If `extra_params` are defined for this
                `AgentVariants` instance, their values are sourced from this
                `RunRequest`.
            flow_default_params (dict): Default parameters from the flow configuration.

        Returns:
            list[tuple[type[Any], AgentConfig]]: A list of tuples, where each tuple
            contains the agent class (obtained from `AgentRegistry` via `agent_obj`)
            and the generated `AgentConfig` instance.

        Raises:
            ValueError: If an `extra_param` is specified but not found in the
                provided `RunRequest` `params`.
            FatalError: If no agent configurations can be generated.
            TypeError: If `agent_obj` is not found in the `AgentRegistry`.

        """
        static_config_dict = self.model_dump(
            exclude={
                "agent_obj",  # Exclude agent_obj as it's used to get the class
                "variants",
                "num_runs",
                "extra_params",
                # Also exclude fields that are part of AgentConfig's identity if they are recalculated
                "agent_id",
                "_agent_name",
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
                if (
                    not hasattr(params, key) or getattr(params, key) is None
                ):  # Check if param exists in RunRequest
                    raise ValueError(
                        f"Required extra_param '{key}' not found or is None in RunRequest for agent variant '{self.agent_id or self.role}'."
                    )
                base_parameters[key] = getattr(params, key)

        # Merge parameters from the RunRequest.parameters (user-provided overrides)
        if params and params.parameters:
            base_parameters.update(clean_empty_values(params.parameters))

        from buttermilk._core.variants import AgentRegistry  # Lazy import

        try:
            agent_class = AgentRegistry.get(self.agent_obj)
            if (
                agent_class is None
            ):  # AgentRegistry.get might return None if not found and not raising
                raise TypeError(
                    f"Agent class '{self.agent_obj}' not found in AgentRegistry."
                )
        except KeyError:  # Assuming AgentRegistry might raise KeyError
            raise TypeError(
                f"Agent class '{self.agent_obj}' not found in AgentRegistry."
            )

        # Filter out variant parameters that are overridden by RunRequest parameters
        filtered_variants = self.variants.copy() if self.variants else {}
        if params and params.parameters:
            # Remove any variant keys that are explicitly set in params.parameters
            for key in params.parameters.keys():
                filtered_variants.pop(key, None)

        variant_combinations = (
            expand_dict(clean_empty_values(filtered_variants))
            if filtered_variants
            else [{}]
        )

        generated_configs: list[tuple[type[Any], AgentConfig]] = []
        for _ in range(self.num_runs):  # Loop for num_runs
            for variant_params in variant_combinations:
                # Start with the static parts of AgentVariants config
                current_config_dict = copy.deepcopy(static_config_dict)

                # Combine parameters: flow defaults, then base (agent + RunRequest), then variant.
                # This order defines precedence - later values override earlier ones.
                final_params = {
                    **flow_default_params,
                    **base_parameters,
                    **variant_params,
                }
                current_config_dict["parameters"] = clean_empty_values(final_params)

                # Ensure all necessary fields for AgentConfig are present or defaulted
                # Role and description might come from static_config_dict or need defaults
                current_config_dict.setdefault("role", self.role or "VARIANT_AGENT")
                current_config_dict.setdefault(
                    "description", self.description or "Generated variant agent"
                )

                # Explicitly remove fields not in AgentConfig before instantiation
                # This is safer than relying solely on AgentConfig.model_config['extra'] = 'ignore'
                # if AgentConfig itself doesn't have 'extra':'allow' or if strictness is desired.
                valid_agent_config_fields = AgentConfig.model_fields.keys()
                filtered_cfg_dict = {
                    k: v
                    for k, v in current_config_dict.items()
                    if k in valid_agent_config_fields
                }

                # Ensure 'parameters' contains the final merged parameters
                filtered_cfg_dict["parameters"] = final_params

                try:
                    agent_config_instance = AgentConfig(**filtered_cfg_dict)
                    generated_configs.append((agent_class, agent_config_instance))
                except Exception as e:
                    logger.error(
                        msg
                        := f"Error creating AgentConfig for role '{filtered_cfg_dict.get('role', 'unknown')}' "
                        f"with parameters {final_params}: {e}",
                    )
                    raise FatalError(msg) from e

        if not generated_configs:
            logger.warning(
                f"No agent configurations were generated for AgentVariants: {self.agent_id or self.role}. "
                f"This might be due to empty 'variants' with num_runs=0, or misconfiguration."
            )
            # Depending on desired behavior, could raise FatalError or return empty list.
            # Current behavior: returns empty list, which might be handled by caller.
            # However, the original code raised FatalError, so let's keep that.
            logger.error(
                msg
                := f"Could not create any agent variant configs for {self.role or self.agent_name}"
            )
            raise FatalError(msg)

        return generated_configs


class SessionProgressConfig(BaseModel):
    """Configuration for session progress tracking."""

    current_step: int = Field(default=0, description="Current step in the session")
    total_steps: int = Field(default=100, description="Total steps in the session")
    status: str = Field(default="waiting", description="Current session status")


class SessionDefaultsConfig(BaseModel):
    """Default configuration for session data structures."""

    progress: SessionProgressConfig = Field(default_factory=SessionProgressConfig)
    scores: dict[str, Any] = Field(
        default_factory=dict, description="Default scores structure"
    )
    outcomes: list[Any] = Field(
        default_factory=list, description="Default outcomes list"
    )
    pending_agents: list[str] = Field(
        default_factory=list, description="Default pending agents list"
    )


class SessionConfig(BaseModel):
    """Configuration for session management."""

    timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    cleanup_interval_minutes: int = Field(
        default=15, description="Cleanup interval in minutes"
    )
    max_concurrent_sessions: int = Field(
        default=50, description="Maximum concurrent sessions"
    )
    defaults: SessionDefaultsConfig = Field(default_factory=SessionDefaultsConfig)
