"""Defines the abstract base class for Orchestrators in Buttermilk.

Orchestrators are central components in the Buttermilk framework, responsible for
managing the entire lifecycle of an agent-based workflow (often referred to as a "flow").
This includes setting up the necessary environment, loading configurations for data
sources and agents, executing the sequence of operations defined in the flow,
facilitating communication and data exchange between agents, handling user
interactions (if any), collecting results, and ensuring proper cleanup of resources
after the flow execution is complete.

The module provides `OrchestratorProtocol` as a Pydantic model to define the
expected structure of a flow configuration (typically loaded from YAML). The
`Orchestrator` class itself is an abstract base class (ABC) that implements
this protocol and provides the core logic and interface for orchestration.
Subclasses of `Orchestrator` must implement specific abstract methods to define
the setup, execution loop, and cleanup logic pertinent to their particular
orchestration strategy (e.g., a linear sequence, a graph-based execution, or
an Autogen-based multi-agent conversation).
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Self

import pandas as pd
import shortuuid  # For generating unique IDs
import weave  # For tracing capabilities
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from buttermilk import buttermilk as bm  # Global Buttermilk instance for framework access

# Buttermilk core imports
from buttermilk._core.config import (  # Configuration models
    AgentVariants,
    DataSourceConfig,
    SaveInfo,  # Added SaveInfo
)
from buttermilk._core.contract import FlowMessage
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.log import logger
from buttermilk._core.message_data import clean_empty_values
from buttermilk._core.types import (
    Record,  # Data types
    RunRequest,
)
from buttermilk.runner.helpers import prepare_step_df  # Helper for data preparation
from buttermilk.utils.media import download_and_convert  # Media utilities
from buttermilk.utils.templating import KeyValueCollector  # State management utility
from buttermilk.utils.validators import convert_omegaconf_objects  # Pydantic validators

# Re-importing for clarity, though already imported above.
# from .config import AgentVariants, DataSourceConfig, SaveInfo
# from .types import Record


class OrchestratorProtocol(BaseModel):
    """Defines the Pydantic model for a flow configuration.

    This model specifies the overall structure expected when a flow configuration
    is loaded (e.g., from a YAML file). It serves as the blueprint for initializing
    a new orchestrated flow, detailing the orchestrator to use, its name,
    description, data sources, agent configurations, and other parameters.

    Attributes:
        orchestrator (str): The name or identifier of the orchestrator object/class
            that should be used to execute this flow. This is a mandatory field.
        name (str): A human-friendly name for this specific flow configuration.
            Useful for logging and identification.
        description (str): A brief description explaining the purpose and goals
            of this flow.
        save (SaveInfo | None): Optional configuration for saving the results of
            the flow (e.g., to a file, database, or cloud storage). If None,
            results might not be persisted automatically by the orchestrator.
        data (Mapping[str, DataSourceConfig]): A mapping where keys are descriptive
            names for data sources and values are `DataSourceConfig` objects
            defining how to load and configure each data source for the flow.
        agents (Mapping[str, AgentVariants]): A mapping where keys are agent role
            names (typically in uppercase, e.g., "SUMMARIZER") and values are
            `AgentVariants` configurations. `AgentVariants` define how one or
            more instances of an agent (with potential parameter variations)
            are created for that role.
        observers (Mapping[str, AgentVariants]): Similar to `agents`, but these
            agents are typically passive listeners in a discussion or flow. They
            are initialized and can receive messages but might not be directly
            called upon to produce output in the main sequence.
        parameters (Mapping[str, Any]): A dictionary of flow-level parameters
            that can be accessed by agents within their context. This allows for
            global configuration or shared values across the flow.
        _validate_parameters: Pydantic field validator to convert OmegaConf objects
            (like DictConfig) in `parameters`, `data`, `agents`, and `observers`
            to standard Python dicts/lists before further validation.
        model_config (ConfigDict): Pydantic model configuration.
            - `arbitrary_types_allowed`: True.
            - `extra`: "ignore" - Ignores extra fields not defined in the model.

    """

    orchestrator: str = Field(
        ...,  # Mandatory field
        description="Name or identifier of the orchestrator object/class to use for this flow.",
    )
    name: str = Field(
        default="",
        description="Human-friendly name identifying this specific flow configuration.",
    )
    description: str = Field(
        default="",
        description="Short description explaining the purpose and goals of this flow.",
    )
    save: SaveInfo | None = Field(
        default=None,
        description="Optional configuration for saving flow results (e.g., to disk, database).",
    )
    data: Mapping[str, DataSourceConfig] = Field(
        default_factory=dict,
        description="Configuration for data sources to be loaded for the flow, keyed by a descriptive name.",
    )
    agents: Mapping[str, AgentVariants] = Field(
        default_factory=dict,
        description="Mapping of agent roles (typically uppercase) to their `AgentVariants` configurations.",
    )
    observers: Mapping[str, AgentVariants] = Field(
        default_factory=dict,
        description="Agents that are present (e.g., in a discussion) but not actively called upon in the main sequence.",
    )
    parameters: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Flow-level parameters accessible by agents, providing shared configuration or context.",
    )

    _validate_parameters: classmethod = field_validator(
        "parameters", "data", "agents", "observers", mode="before",
    )(convert_omegaconf_objects)  # type: ignore

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allows for flexibility if some configs are complex types
        extra="ignore",  # Ignores fields in the input data not defined in this model
    )


class Orchestrator(OrchestratorProtocol, ABC):
    """Abstract Base Class for orchestrators that manage and execute agent-based flows.

    Orchestrators are responsible for the entire lifecycle of a flow, including:
    - Loading and validating the flow configuration (data sources, agents, parameters)
      based on `OrchestratorProtocol`.
    - Setting up the execution environment, which might include initializing
      communication runtimes (e.g., for Autogen-based multi-agent systems).
    - Managing the sequence of operations or steps within the flow. This can be
      a predefined sequence, determined by a conductor agent, or influenced by
      user interactions.
    - Facilitating communication between agents and ensuring proper data flow.
    - Handling optional user interactions through a designated MANAGER interface.
    - Collecting results from agent executions and potentially saving them according
      to the `save` configuration.
    - Ensuring all resources are properly cleaned up after the flow execution,
      regardless of success or failure.

    Subclasses must implement the `_setup`, `_cleanup`, and `_run` abstract methods.
    The `_run` method typically contains the main control loop or execution logic
    specific to the orchestration strategy.

    Internal State Attributes:
        trace_id (str): A unique ID for the current flow execution session, primarily
            used for tracing purposes (e.g., with Weave). Defaults to a new short UUID.
        _flow_data (KeyValueCollector): An internal state collector used to store
            and manage data passed between steps or used for templating within the flow.
        _data_sources (dict[str, Any]): A dictionary to store loaded data sources,
            keyed by the names defined in the `data` configuration. Values are typically
            Pandas DataFrames or similar data structures.
        _records (list[Record]): A list of `Record` objects currently loaded or
            being processed by the flow.
        model_config (ConfigDict): Pydantic model configuration.
            - `extra`: "forbid" - Disallows extra fields not explicitly defined.
            - `arbitrary_types_allowed`: False.
            - `populate_by_name`: True.
    """

    trace_id: str = Field(
        default_factory=shortuuid.uuid,
        description="Unique ID for this specific flow execution session, used for tracing.",
    )
    _flow_data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)
    _data_sources: dict[str, Any] = PrivateAttr(default_factory=dict)  # Changed default to factory
    _records: list[Record] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_and_initialize_agents(self) -> Self:
        """Validates agent configurations and initializes internal agent-related structures.

        This method is a Pydantic model validator that runs after the initial
        model instance is created. It processes the `agents` mapping from the
        configuration:
        1. Ensures all role keys in `self.agents` are uppercase for consistency.
        2. Validates that each agent definition conforms to the `AgentVariants` model.
           If a plain dictionary is provided, it attempts to parse it as an `AgentVariants`.
        3. Initializes the internal `_flow_data` state collector with the list of
           validated agent roles.

        Returns:
            Self: The validated and updated orchestrator instance.

        Raises:
            ValueError: If an agent configuration is invalid or cannot be parsed
                as `AgentVariants`.
            TypeError: If an agent definition is not a dictionary or `AgentVariants` instance.

        """
        logger.debug(f"Validating agents for orchestrator '{self.name}'.")
        validated_agents: dict[str, AgentVariants] = {}
        agent_roles: list[str] = []
        for step_name, defn in self.agents.items():
            role_upper = step_name.upper()
            agent_roles.append(role_upper)
            if isinstance(defn, AgentVariants):
                validated_agents[role_upper] = defn
            elif isinstance(defn, Mapping):  # Check if it's a dict-like object
                try:
                    validated_agents[role_upper] = AgentVariants(**defn)  # type: ignore
                except Exception as e:
                    logger.error(f"Invalid AgentVariants configuration for role '{role_upper}': {defn}. Error: {e}")
                    raise ValueError(f"Invalid AgentVariants config for role '{role_upper}'") from e
            else:
                raise TypeError(f"Invalid type for agent definition '{role_upper}': {type(defn)}. Expected dict or AgentVariants.")
        self.agents = validated_agents
        logger.debug(f"Agent roles validated: {list(self.agents.keys())}")

        self._flow_data.init(agent_roles)
        return self

    async def load_data(self) -> None:
        """Loads data from the configured data sources.

        Populates `self._data_sources` by processing the `DataSourceConfig`
        entries in `self.data` using `prepare_step_df`.
        This method should be called before attempting to access data via
        `get_record_dataset` if data sources are defined.
        """
        if self.data:  # Only load if data sources are configured
            self._data_sources = await prepare_step_df(self.data)
            logger.info(f"Data sources loaded for orchestrator '{self.name}': {list(self._data_sources.keys())}")
        else:
            logger.info(f"No data sources configured for orchestrator '{self.name}'.")

    async def run(self, request: RunRequest) -> None:
        """Public entry point to start the orchestrator's flow execution.

        This method sets up the Weave tracing context for the entire run and then
        calls the internal `_run` method, which contains the core execution logic.
        It ensures that tracing is properly initialized and finalized around the
        actual flow execution.

        Args:
            request: A `RunRequest` object containing initial data (e.g., records,
                a specific record_id, or a URI to fetch data from) and any
                runtime parameters for the flow. It also includes tracing attributes.

        """
        display_name = f"{self.name} {request.name}"
        logger.info(f"Starting run for orchestrator flow {display_name}.")

        # Get the singleton instance using our new module-level function
        # Define attributes for logging and tracing.
        op = weave.op(self._run, call_display_name=display_name)
        orchestrator_trace = bm.weave.create_call(op, inputs=clean_empty_values(request.model_dump(mode="json")),
                                                    display_name=display_name,
                                                    attributes=request.tracing_attributes)
        self.trace_id = orchestrator_trace.trace_id
        try:
            # Execute the core run logic.
            await self._run(request=request)
            logger.info(f"Orchestrator '{self.name}' run '{request.name}' finished successfully.")
        except Exception as e:
            logger.exception(f"Orchestrator '{self.name}' run '{request.name}' failed: {e!s}")
            # Optionally re-raise or handle the error further.
            # For now, it's logged, and the trace will be finalized.
        finally:
            # Ensure the Weave call is marked as finished, regardless of success or failure.
            bm.weave.finish_call(orchestrator_trace, op=op)

    @abstractmethod
    async def _setup(self, request: RunRequest) -> None:
        """Abstract method for orchestrator-specific setup tasks.

        Subclasses must implement this method to perform any necessary
        initialization before the main execution loop begins. This can include:
        - Initializing communication runtimes (e.g., Autogen's group chat).
        - Establishing connections to databases or external services.
        - Pre-loading essential components or models.
        - Fetching initial records based on the `request` if not already loaded.

        This method is called once at the beginning of the `_run` method.

        Args:
            request: The `RunRequest` object containing initial parameters and
                data for the flow. Implementations should use this to fetch
                initial records if `self._records` is empty.

        """
        # Example of how initial records might be fetched. Subclasses should adapt this.
        if not self._records:  # Only fetch if no records are already present
            await self._fetch_initial_records(request)
        # raise NotImplementedError("Orchestrator subclasses must implement _setup.") # Keep if base does nothing else

    @abstractmethod
    async def _cleanup(self) -> None:
        """Abstract method for orchestrator-specific cleanup tasks.

        Subclasses must implement this method to release any resources acquired
        during `_setup` or the main execution. This can include:
        - Stopping communication runtimes.
        - Closing database connections or network sessions.
        - Cleaning up temporary files or state.

        This method is called in a `finally` block within the public `run` method
        to ensure it executes even if errors occur during the flow.
        """
        raise NotImplementedError("Orchestrator subclasses must implement _cleanup.")

    @abstractmethod
    async def _run(self, request: RunRequest) -> None:
        """Abstract method containing the main execution logic or control loop for the flow.

        This method is called by the public `run` method after Weave tracing has
        been set up. Subclasses **MUST** implement this to define the core
        orchestration logic. This includes:
        - Potentially loading initial data if not handled in `_setup` (though
          `_setup` is preferred for data loading based on `request`).
        - Determining the sequence of steps or agent interactions (e.g., based on
          a fixed sequence, a conductor agent's decisions, or user input).
        - Executing agent steps, possibly using a helper method like `_execute_step`
          (which would be defined by the subclass).
        - Managing data flow and state between steps.

        Args:
            request: The `RunRequest` object containing initial parameters and
                data for the flow. This provides the starting context.

        """
        # Base implementation should ensure setup is called.
        # Subclasses will override this entirely with their loop logic.
        try:
            await self._setup(request=request)
            # --- Subclass implementation of the main execution loop goes here ---
            logger.warning(
                f"Orchestrator subclass {self.__class__.__name__} did not fully override "
                f"the _run method's execution loop. Only setup was called.",
            )
        except Exception as e:
            logger.exception(f"Error during orchestrator _run for '{self.name}': {e!s}")
            # Optionally re-raise as FatalError or a more specific orchestrator error
            # raise ProcessingError(f"Orchestrator _run failed: {e!s}") from e
        # Cleanup is handled by the public `run` method's finally block.

    async def _fetch_initial_records(self, request: RunRequest) -> None:
        """Fetches initial records based on the `RunRequest` if not already loaded.

        This helper method checks if `self._records` is empty. If so, and if the
        `request` provides a `record_id` or `uri`, it attempts to fetch the
        corresponding record(s). If `request.records` is already populated,
        those are used directly.

        Args:
            request: The `RunRequest` object containing potential sources for
                initial records.

        Raises:
            FatalError: If fetching by `record_id` or `uri` fails.

        """
        if self._records:  # Records already exist (e.g., set by subclass or previous step)
            logger.debug("Orchestrator already has records; skipping initial fetch from RunRequest.")
            return

        if request.records:  # Records provided directly in RunRequest
            logger.debug(f"Using {len(request.records)} records provided directly in RunRequest.")
            self._records = request.records
            return

        # If no records yet, try fetching via record_id or uri from RunRequest
        if request.record_id or request.uri:
            logger.debug(f"Attempting to fetch initial record(s) based on RunRequest: id='{request.record_id}', uri='{request.uri}'.")
            try:
                record_to_add: Record | None = None
                fetch_source_id = ""
                if request.record_id:
                    record_to_add = await self.get_record_dataset(request.record_id)
                    fetch_source_id = request.record_id
                elif request.uri:
                    record_to_add = await download_and_convert(request.uri)  # Assumes download_and_convert returns a Record
                    fetch_source_id = request.uri

                if record_to_add:
                    logger.debug(f"Initial record fetched: {record_to_add.record_id} from source '{fetch_source_id}'.")
                    # Standardize metadata for fetched records
                    record_to_add.metadata["fetch_source_id"] = fetch_source_id
                    record_to_add.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                    self._records = [record_to_add]
                else:
                    logger.warning(f"No record found for record_id='{request.record_id}' or uri='{request.uri}'.")

            except Exception as e:
                msg = f"Error fetching initial record from request (id='{request.record_id}', uri='{request.uri}'): {e!s}"
                logger.error(msg)
                raise FatalError(msg) from e
        else:
            logger.info("No initial records, record_id, or URI provided in RunRequest. Orchestrator starts with empty records list.")

    async def get_record_dataset(self, record_id: str) -> Record:
        """Retrieves a specific record by its ID from the loaded data sources.

        This method first ensures that data sources are loaded (by calling
        `self.load_data()` if `self._data_sources` is empty). It then iterates
        through the loaded datasets (typically Pandas DataFrames) and queries
        for the given `record_id`.

        Args:
            record_id: The unique identifier of the record to retrieve.

        Returns:
            Record: The found `Record` object.

        Raises:
            ProcessingError: If the specified `record_id` cannot be found in any
                of the loaded data sources.
            ValueError: If multiple records are found for the given `record_id`
                (implying non-unique IDs within or across datasets).

        """
        if not self._data_sources:  # Ensure data sources are loaded
            await self.load_data()
            if not self._data_sources:  # Still no data sources after attempting load
                raise ProcessingError(f"No data sources loaded. Cannot find record: {record_id}")

        for source_name, dataset in self._data_sources.items():
            # Assuming dataset is a Pandas DataFrame, adjust if other types are used
            if not isinstance(dataset, pd.DataFrame):
                logger.warning(f"Data source '{source_name}' is not a Pandas DataFrame. Skipping query for record_id '{record_id}'.")
                continue

            try:
                # Ensure record_id is a valid column or index name before querying
                # This check might be too simplistic depending on DataFrame structure
                if "record_id" in dataset.columns or dataset.index.name == "record_id":
                    # Querying logic might need adjustment based on how record_id is stored (column vs. index)
                    # This example assumes 'record_id' can be directly queried.
                    # If record_id is the index: found_records = dataset.loc[[record_id]]
                    # If record_id is a column: found_records = dataset.query("record_id == @record_id")
                    # For flexibility, let's try to be adaptive, but this could be fragile.
                    if "record_id" in dataset.columns:
                        found_records_df: pd.DataFrame = dataset.query("record_id == @record_id")
                    elif dataset.index.name == "record_id" and record_id in dataset.index:
                        found_records_df = dataset.loc[[record_id]]
                    else:  # record_id not found as column or index name in this dataset
                        continue
                else:  # 'record_id' column or index not present in this dataset
                    continue

                if found_records_df.shape[0] == 1:
                    data_dict = found_records_df.iloc[0].to_dict()
                    # Ensure 'record_id' is in the dict, taking from index if it was the source
                    if "record_id" not in data_dict and found_records_df.index.name == "record_id":
                        data_dict["record_id"] = found_records_df.index[0]

                    return Record(**data_dict)  # type: ignore
                if found_records_df.shape[0] > 1:
                    raise ValueError(
                        f"Multiple records found in source '{source_name}' for record_id == '{record_id}'. Record IDs must be unique.",
                    )
            except Exception as e:  # Catch errors during query or Record instantiation
                logger.warning(f"Error querying or processing record from source '{source_name}' for record_id '{record_id}': {e!s}")
                continue  # Try next data source

        raise ProcessingError(f"Unable to find requested record: '{record_id}' in any loaded data source.")

    def make_publish_callback(self) -> Callable[[FlowMessage], Awaitable[None]]:
        """Creates and returns an asynchronous callback function for publishing messages.

        This callback is typically passed to agents or other components that need
        to send messages (e.g., status updates, results) back to a central point,
        often for UI updates or inter-agent communication brokered by the orchestrator.

        The default implementation provided here is a no-op (it does nothing).
        Subclasses of `Orchestrator` should override this method to provide a
        concrete implementation relevant to their communication strategy (e.g.,
        sending messages over WebSockets, adding to a message queue, or calling
        a UI update service).

        Returns:
            Callable[[FlowMessage], Awaitable[None]]: An asynchronous function
            that takes a `FlowMessage` as input and performs the publishing action.

        """
        async def publish_callback(message: FlowMessage) -> None:
            """Default no-op publish callback. Subclasses should implement actual publishing logic."""
            logger.debug(f"Orchestrator '{self.name}' received message via default (no-op) publish_callback: {type(message).__name__}")
            # In a real implementation, this would involve:
            # - Sending the message to connected UI clients (e.g., via WebSockets).
            # - Placing the message on a queue for other services.
            # - Logging the message to a persistent store.
            # Default implementation does nothing.

        return publish_callback
