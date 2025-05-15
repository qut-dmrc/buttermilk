"""Defines the abstract base class for Orchestrators in Buttermilk.

Orchestrators are responsible for managing the setup, execution, and cleanup
of agent-based workflows (flows).
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Self

import pandas as pd
import shortuuid  # For generating unique IDs
import weave  # For tracing
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

# Buttermilk core imports
# Agent base class and related types
from buttermilk._core.config import (  # Configuration models
    AgentVariants,  # Agent variant configuration
    DataSourceConfig,
)
from buttermilk._core.contract import FlowMessage
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.types import (
    Record,  # Data types
    RunRequest,
)
from buttermilk.bm import BM, logger  # Buttermilk global instance and logger

from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.templating import KeyValueCollector  # State management utility
from buttermilk.utils.validators import convert_omegaconf_objects

from .config import AgentVariants, DataSourceConfig, SaveInfo  # Core configuration models
from .types import Record  # Core data types

bm = BM()

class OrchestratorProtocol(BaseModel):
    """Defines the overall structure expected for a flow configuration (e.g., loaded from YAML).
    Used to initialise a new orchestrated flow.

    Attributes:
        session_id (str): Unique ID for the current flow execution session.
        name (str): Human-friendly name for the flow.
        description (str): Description of the flow's purpose.
        save (SaveInfo | None): Configuration for saving results (optional).
        data (Sequence[DataSourceConfig]): List of data sources for the flow.
        agents (Mapping[str, AgentVariants]): Dictionary mapping role names to agent variant configurations.
        parameters (dict): Flow-level parameters accessible by agents.
        _flow_data (KeyValueCollector): Internal state collector for the flow.
        _records (list[Record]): List of data records currently loaded/used in the flow.

    """

    # --- Configuration Fields ---
    orchestrator: str = Field(..., description="Name of the orchestrator object to use.")
    name: str = Field(
        default="",
        description="Human-friendly name identifying this flow configuration.",
    )
    description: str = Field(
        default="",  # Default to empty string
        description="Short description explaining the purpose of this flow.",
    )
    save: SaveInfo | None = Field(default=None, description="Configuration for saving results (e.g., to disk, database). Optional.")
    data: Mapping[str, DataSourceConfig] = Field(
        default_factory=dict,
        description="Configuration for data sources to be loaded for the flow.",
    )
    agents: Mapping[str, AgentVariants] = Field(
        default_factory=dict,
        description="Mapping of agent roles (uppercase) to their variant configurations.",
    )

    observers: Mapping[str, AgentVariants] = Field(
        default_factory=dict, description="Agents that will not be called upon but are still present in a discussion.",
    )
    parameters: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Flow-level parameters accessible by agents via their context.",
    )

    # Ensure OmegaConf objects (like DictConfig) are converted to standard Python dicts before validation.
    _validate_parameters = field_validator("parameters", "data", "agents", "observers", mode="before")(convert_omegaconf_objects())

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )


class Orchestrator(OrchestratorProtocol, ABC):
    """Abstract Base Class for orchestrators that manage agent-based flows.

    Orchestrators handle:
    - Loading flow configurations (data sources, agents, parameters).
    - Setting up the execution environment (e.g., communication runtimes).
    - Managing the sequence of steps (via internal logic or conductor agents).
    - Facilitating agent communication and data flow.
    - Optional user interaction via a MANAGER interface.
    - Collecting and potentially saving results.
    - Cleaning up resources post-execution.

    Subclasses must implement `_setup`, `_cleanup`, and `_execute_step`.
    The `_run` method typically contains the main control loop logic.
    """

    # --- Internal State ---
    session_id: str = Field(
        default_factory=shortuuid.uuid,
        description="A unique session id for this specific flow execution.",
    )
    # Collects data passed between steps or used for templating.
    _flow_data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)
    _data_sources: dict[str, Any] = PrivateAttr(default={})
    # Holds the primary data records being processed by the flow.
    _records: list[Record] = PrivateAttr(default_factory=list)

    # Pydantic Model Configuration
    model_config = ConfigDict(
        extra="forbid",  # Disallow extra fields in config unless explicitly handled by subclasses.
        arbitrary_types_allowed=False,  # Requires explicit handling for non-standard types.
        populate_by_name=True,  # Allows using field names in config keys.
    )

    # --- Validators ---

    @model_validator(mode="after")
    def validate_and_initialize_agents(self) -> Self:
        """Validates agent configurations and initializes internal data structures."""
        logger.debug(f"Validating agents for orchestrator '{self.name}'.")
        validated_agents = {}
        agent_roles = []
        for step_name, defn in self.agents.items():
            role_upper = step_name.upper()  # Ensure role keys are uppercase
            agent_roles.append(role_upper)
            if isinstance(defn, AgentVariants):
                validated_agents[role_upper] = defn
            elif isinstance(defn, Mapping):
                try:
                    # Validate dict against AgentVariants model
                    validated_agents[role_upper] = AgentVariants(**defn)
                except Exception as e:
                    logger.error(f"Invalid AgentVariants configuration for role '{role_upper}': {defn}. Error: {e}")
                    raise ValueError(f"Invalid AgentVariants config for role '{role_upper}'") from e
            else:
                raise TypeError(f"Invalid type for agent definition '{role_upper}': {type(defn)}. Expected dict or AgentVariants.")
        self.agents = validated_agents
        logger.debug(f"Agent roles validated: {list(self.agents.keys())}")

        # Initialize the KeyValueCollector with the known agent roles.
        self._flow_data.init(agent_roles)
        return self

    # --- Records ---
    async def load_data(self):
        self._data_sources = await prepare_step_df(self.data)

    # --- Public Execution Method ---
    async def run(self, request: RunRequest) -> None:
        """Public entry point to start the orchestrator's flow execution.

        Sets up tracing context for the run and calls the internal `_run` method.

        Args:
            request: An optional RunRequest containing initial data (records, record_id, uri, prompt)
                     or parameters for the flow.

        """
        logger.info(f"Starting run for orchestrator '{self.name}', session '{self.session_id}'.")

        # Define attributes for logging and tracing.
        try:
            assert bm.weave

            with weave.attributes(request.tracing_attributes):
                await self._run(request=request, __weave={"display_name": request.name})

                logger.info(f"Orchestrator '{request.name}' run finished successfully.")

        except Exception as e:
            # Catch errors originating from _run or its setup/cleanup phases.
            logger.exception(f"Orchestrator '{self.name}' run '{request.name}' failed: {e}")
            # Optionally re-raise or handle the error further.

    # --- Abstract & Core Internal Methods ---

    @abstractmethod
    async def _setup(self, request: RunRequest) -> None:
        """Abstract method for orchestrator-specific setup.

        Implementations should initialize resources like communication runtimes
        (e.g., Autogen runtime), database connections, or pre-load essential components.
        Called once at the beginning of the `_run` method.
        """
        if request:
            await self._fetch_initial_records(request)  # Use helper for clarity
        raise NotImplementedError("Orchestrator subclasses must implement _setup.")

    @abstractmethod
    async def _cleanup(self) -> None:
        """Abstract method for orchestrator-specific cleanup.

        Implementations should release resources acquired during `_setup` or
        execution (e.g., stop runtimes, close connections). Called in a `finally`
        block within `run` to ensure execution even if errors occur.
        """
        raise NotImplementedError("Orchestrator subclasses must implement _cleanup.")

    @weave.op
    @abstractmethod
    async def _run(self, request: RunRequest):
        """Abstract method containing the main execution logic/control loop for the flow.

        Called by the public `run` method after Weave tracing setup. Subclasses
        must implement this to define how steps are determined (e.g., fixed sequence,
        conductor agent, user interaction) and executed via `_execute_step`. It should
        also handle initial data loading based on the `request` argument.

        Args:
            request: Optional `RunRequest` containing initial data/parameters.

        """
        # Base implementation handles initial setup, data fetching (if needed), and cleanup.
        # Subclasses MUST override this to provide the actual step execution loop.
        try:
            await self._setup(request=request)

            # --- !!! Subclass implementation needed here !!! ---
            # Example placeholder - replace with actual loop logic:
            # while True:
            #     next_step = await self._determine_next_step(...)
            #     if next_step.role == END: break
            #     await self._execute_step(next_step)
            #     await asyncio.sleep(1) # Example delay
            logger.warning(f"Orchestrator subclass {self.__class__.__name__} did not override _run method. No steps executed.")

        except Exception as e:
            logger.exception(f"Error during orchestrator _run execution: {e}")
            # Optionally re-raise as FatalError if appropriate
            # raise FatalError from e
        # Cleanup is handled by the public `run` method's finally block.

    # --- Optional Overridable Methods ---

    async def _fetch_initial_records(self, request: RunRequest) -> None:
        """Helper method to fetch records based on RunRequest if needed."""
        if not self._records and not request.records:  # Only fetch if no records exist yet
            if request.record_id or request.uri:
                logger.debug("Fetching initial record(s) based on request.")
                try:
                    rec = None
                    if request.record_id:
                        rec = await self.get_record_dataset(request.record_id)
                        rec.metadata["fetch_source_id"] = request.record_id
                    elif request.uri:
                        # Fetch record by URL
                        rec = await download_and_convert(request.uri)
                        rec.metadata["fetch_source_id"] = request.uri
                    if rec:
                        logger.debug(f"Initial record found: {rec.record_id} from {rec.metadata['fetch_source_id']}")
                        rec.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                        rec.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                        self._records = [rec]

                except Exception as e:
                    msg = f"Error fetching initial record: {e}"
                    raise FatalError(msg)
            else:
                logger.warning("No initial records, record_id, or uri provided in request.")
        elif request.records:
            logger.debug(f"Using {len(request.records)} records provided directly in RunRequest.")
            self._records = request.records  # Use records provided in request if available
        else:
            logger.debug("Orchestrator already has records, skipping initial fetch.")

    async def get_record_dataset(self, record_id: str) -> Record:
        if not self._data_sources:
            await self.load_data()

        for dataset in self._data_sources.values():
            rec: pd.Series = dataset.query("record_id==@record_id")
            if rec.shape[0] == 1:
                data = rec.iloc[0].to_dict()
                data["record_id"] = rec.index[0]  # Add the index explicitly
                if "components" in data and not data["components"]:
                    data["components"] = "\n".join([d["content"] for d in data["components"]])
                return Record(**data)
            if rec.shape[0] > 1:
                raise ValueError(
                    f"More than one record found for query record_id == {record_id}",
                )

        raise ProcessingError(f"Unable to find requested record: {record_id}")

    def make_publish_callback(self) -> Callable[[FlowMessage], Awaitable[None]]:
        """Creates a callback function for publishing messages to the UI.

        Returns:
            An async function that sends a FlowMessage to the flow agents. 

        """
        async def publish_callback(message: FlowMessage) -> None:
            # Implement the logic to publish the message to the UI.
            # This could involve sending it over a WebSocket or other communication channel.
            pass

        return publish_callback
