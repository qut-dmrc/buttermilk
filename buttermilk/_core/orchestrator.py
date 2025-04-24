"""
Defines the abstract base class for Orchestrators in Buttermilk.

Orchestrators are responsible for managing the setup, execution, and cleanup
of agent-based workflows (flows).
"""

from abc import ABC, abstractmethod
from ast import arguments  # TODO: Unused import?
import asyncio
from collections.abc import Mapping, Sequence
from enum import Enum  # TODO: Unused import?
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Self

import shortuuid  # For generating session IDs
from autogen_core.model_context import UnboundedChatCompletionContext  # Used in Agent, maybe needed for type hints?
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
import weave  # For tracing

# Buttermilk core imports
from buttermilk._core import TaskProcessingComplete  # Status message type

# Agent base class and related types
from buttermilk._core.agent import Agent,  ChatCompletionContext, FatalError, ProcessingError

from buttermilk._core.config import AgentConfig
from buttermilk._core.config import DataSourceConfig, SaveInfo  # Configuration models
from buttermilk._core.contract import END, AgentInput, ManagerResponse, StepRequest, AgentOutput  # Core message types
from buttermilk._core.flow import KeyValueCollector  # State management utility
from buttermilk._core.types import Record, RunRequest  # Data types
from buttermilk._core.variants import AgentVariants  # Agent variant configuration
from buttermilk.agents.fetch import FetchRecord  # Agent for data fetching
from buttermilk.bm import BM, bm, logger  # Global instance and logger

# TODO: BASE_DIR seems unused. Consider removing.
# BASE_DIR = Path(__file__).absolute().parent


class Orchestrator(BaseModel, ABC):
    """
    Abstract Base Class for orchestrators that manage agent-based flows.

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
    session_id: str = Field(
        default_factory=lambda: shortuuid.uuid()[:8],
        description="A unique session id for this specific flow execution.",
    )
    name: str = Field(
        ...,  # Name is required
        description="Human-friendly name identifying this flow configuration.",
    )
    description: str = Field(
        default="",  # Default to empty string
        description="Short description explaining the purpose of this flow.",
    )
    save: SaveInfo | None = Field(default=None, description="Configuration for saving results (e.g., to disk, database). Optional.")
    data: Sequence[DataSourceConfig] = Field(
        default_factory=list,
        description="Configuration for data sources to be loaded for the flow.",
    )
    agents: Mapping[str, AgentVariants] = Field(
        default_factory=dict,
        description="Mapping of agent roles (uppercase) to their variant configurations.",
    )
    tools: Mapping[str, AgentConfig] = Field(
        default_factory=dict,
        description="Mapping of agent roles (uppercase) to their variant configurations.",
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Flow-level parameters accessible by agents via their context.",
        # exclude=True, # Why exclude? Parameters seem important to serialize/log. Reconsider.
    )

    # --- Internal State ---
    # Collects data passed between steps or used for templating.
    _flow_data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)
    # Holds the primary data records being processed by the flow.
    _records: list[Record] = PrivateAttr(default_factory=list)

    # Pydantic Model Configuration
    model_config = ConfigDict(
        extra="forbid",  # Disallow extra fields in config unless explicitly handled by subclasses.
        arbitrary_types_allowed=False,  # Requires explicit handling for non-standard types.
        populate_by_name=True,  # Allows using field names in config keys.
    )

    # --- Validators ---

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Sequence[DataSourceConfig | Mapping]) -> list[DataSourceConfig]:
        """Ensures all items in the 'data' list are valid DataSourceConfig objects."""
        validated_data = []
        for i, source in enumerate(value):
            if isinstance(source, Mapping):
                try:
                    validated_data.append(DataSourceConfig(**source))
                except Exception as e:
                    logger.error(f"Invalid DataSource configuration at index {i}: {source}. Error: {e}")
                    raise ValueError(f"Invalid DataSource config at index {i}") from e
            elif isinstance(source, DataSourceConfig):
                validated_data.append(source)
            else:
                raise TypeError(f"Invalid type for data source at index {i}: {type(source)}. Expected dict or DataSourceConfig.")
        return validated_data

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

    # --- Public Execution Method ---

    async def run(self, request: RunRequest | None = None) -> None:
        """
        Public entry point to start the orchestrator's flow execution.

        Sets up Weave tracing context for the run and calls the internal `_run` method.

        Args:
            request: An optional RunRequest containing initial data (records, record_id, uri, prompt)
                     or parameters for the flow.
        """
        logger.info(f"Starting run for orchestrator '{self.name}', session '{self.session_id}'.")
        # Define attributes for Weave tracing.
        tracing_attributes = {
            **self.parameters,
            "session_id": self.session_id,
            "orchestrator": self.__class__.__name__,  # Use class name
            "flow_name": self.name,
            "flow_description": self.description,
        }
        # Use weave.op to trace the internal _run method.
        try:
            display_name = self.name
            if "criteria" in self.parameters:
                display_name = f"{display_name} {self.parameters['criteria'][0]}"
            if request and request.record_id:
                display_name = f"{display_name}: {request.record_id}"
            with weave.attributes(tracing_attributes):
                # This creates a traced version of the _run method.
                traced_run_op = weave.op(self._run, call_display_name=display_name)
                # Execute the traced operation within a Weave context.
                await traced_run_op(request=request)

                logger.info(f"Orchestrator '{self.name}' run finished successfully.")

        except Exception as e:
            # Catch errors originating from _run or its setup/cleanup phases.
            logger.exception(f"Orchestrator '{self.name}' run failed: {e}")
            # Optionally re-raise or handle the error further.

    # --- Abstract & Core Internal Methods ---

    @abstractmethod
    async def _setup(self) -> None:
        """
        Abstract method for orchestrator-specific setup.

        Implementations should initialize resources like communication runtimes
        (e.g., Autogen runtime), database connections, or pre-load essential components.
        Called once at the beginning of the `_run` method.
        """
        raise NotImplementedError("Orchestrator subclasses must implement _setup.")

    @abstractmethod
    async def _cleanup(self) -> None:
        """
        Abstract method for orchestrator-specific cleanup.

        Implementations should release resources acquired during `_setup` or
        execution (e.g., stop runtimes, close connections). Called in a `finally`
        block within `run` to ensure execution even if errors occur.
        """
        raise NotImplementedError("Orchestrator subclasses must implement _cleanup.")

    @abstractmethod
    async def _execute_step(self, step: StepRequest) -> AgentOutput | None:
        """
        Abstract method to execute a single step defined by a `StepRequest`.

        Implementations handle the mechanism for invoking the correct agent
        (based on `step.role`) with the appropriate input (`step.prompt`, context, etc.)
        and returning the agent's `AgentOutput`. This might involve direct calls or
        messaging via a runtime.

        Args:
            step: The `StepRequest` detailing the step to execute.

        Returns:
            The `AgentOutput` from the executed agent, or None if execution failed.
        """
        raise NotImplementedError("Orchestrator subclasses must implement _execute_step.")

    @abstractmethod
    async def _run(self, request: RunRequest | None = None) -> None:
        """
        Abstract method containing the main execution logic/control loop for the flow.

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
            await self._setup()
            if request:
                await self._fetch_initial_records(request)  # Use helper for clarity

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

    async def _in_the_loop(self, step: StepRequest | None = None, prompt: str = "") -> ManagerResponse:
        """
        Placeholder for human-in-the-loop interaction.

        Interactive orchestrators (like `Selector`) override this to send `ManagerRequest`
        messages to the UI/user and wait for a `ManagerResponse`. The base implementation
        skips interaction and automatically confirms progression.

        Args:
            step: The proposed step for confirmation (optional).
            prompt: A message to display if no step is provided (optional).

        Returns:
            A `ManagerResponse` indicating automatic confirmation.
        """
        logger.debug("Base _in_the_loop called: Auto-confirming step.")
        return ManagerResponse(confirm=True)  # Default: automatically confirm

    async def execute(self, request: StepRequest) -> AgentOutput | None:
        """
        Allows direct execution of a single step (e.g., for testing or specific control flows).

        Args:
            request: The `StepRequest` defining the step to execute.

        Returns:
            The `AgentOutput` from the step, or None on failure.
        """
        logger.debug(f"Directly executing step for role: {request.role}")
        try:
            output = await self._execute_step(request)
            return output
        except Exception as e:
            logger.error(f"Error during direct execute for step '{request.role}': {e}")
            return None

    async def __call__(self, request: RunRequest | None = None) -> None:
        """Makes the orchestrator instance callable, triggering its `run` method."""
        await self.run(request=request)
        # Typically __call__ in this pattern doesn't return a value. Results are handled internally or via saving.

    async def _fetch_initial_records(self, request: RunRequest) -> None:
        """Helper method to fetch records based on RunRequest if needed."""
        if not self._records and not request.records:  # Only fetch if no records exist yet
            if request.uri or request.record_id:
                logger.debug(f"Fetching initial records for request (ID: {request.record_id}, URI: {request.uri})...")
                try:
                    # Use the FetchRecord agent directly (consider if this should be part of the flow instead)
                    fetch_agent = FetchRecord(role="fetch_init", data=list(self.data))
                    fetch_output = await fetch_agent._run(uri=request.uri, record_id=request.record_id, prompt=request.prompt)
                    if fetch_output and fetch_output.results:
                        self._records = fetch_output.results
                        logger.debug(f"Successfully fetched {len(self._records)} initial record(s).")
                    else:
                        logger.warning("Initial fetch did not return any results.")
                except ImportError:
                    logger.error("Could not import FetchRecord agent for initial fetch.")
                except Exception as e:
                    logger.error(f"Error fetching initial record: {e}")
                    # Decide if fetch failure is fatal
            else:
                logger.debug("No initial records, record_id, or uri provided in request.")
        elif request.records:
            logger.debug(f"Using {len(request.records)} records provided directly in RunRequest.")
            self._records = request.records  # Use records provided in request if available
        else:
            logger.debug("Orchestrator already has records, skipping initial fetch.")

    async def _evaluate_step(
        self,
        output: AgentOutput,
        ground_truth_record: Record | None,
        criteria: Any | None,
        weave_call: Any | None,  # Weave call object for logging
    ) -> None:
        """
        Placeholder for evaluating an agent's output.

        Subclasses can override this to find and execute appropriate 'scorer' agents,
        potentially logging results back to the Weave trace associated with `weave_call`.

        Args:
            output: The AgentOutput to evaluate.
            ground_truth_record: The ground truth data (if available).
            criteria: The criteria used for evaluation (if available).
            weave_call: The Weave call object associated with the `output` generation.
        """
        logger.debug(f"Base _evaluate_step called for output from agent {output.agent_id}. No evaluation performed.")
        pass  # Default implementation does nothing.

    @abstractmethod
    async def _execute_step(
        self,
        step: AgentInput,
    ) -> AgentOutput | None:
        # Run step
        raise NotImplementedError


# --- Orchestrator Protocol (for Type Hinting/Hydra) ---


# Defines the expected structure of the configuration object *after* Hydra instantiation.
# Used primarily for type hinting in the `cli.py` entry point.
class OrchestratorProtocol(BaseModel):
    """Defines the expected structure of the Hydra configuration object after instantiation."""

    bm: BM  # The core Buttermilk instance.
    flows: Mapping[str, Orchestrator]  # Dictionary of configured flow orchestrators.
    ui: Literal["console", "api", "pub/sub", "slackbot"]  # The selected UI mode.
    flow: str  # The name of the specific flow selected to run (e.g., 'batch', 'panel').
    # Optional command-line overrides for the 'console' UI mode.
    record_id: str = ""
    uri: str = ""
    prompt: str = ""
