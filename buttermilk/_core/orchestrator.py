from abc import ABC, abstractmethod
from ast import arguments
import asyncio
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Self
import shortuuid
from autogen_core.model_context import UnboundedChatCompletionContext
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
import weave

from buttermilk._core import TaskProcessingComplete  # Removed AgentOutput from here
from buttermilk._core.agent import Agent, ChatCompletionContext, FatalError, ProcessingError  # Added Agent
from buttermilk._core.config import DataSourceConfig, SaveInfo
from buttermilk._core.contract import END, AgentInput, ManagerResponse, StepRequest, AgentOutput  # Added AgentOutput here
from buttermilk._core.flow import KeyValueCollector

# from buttermilk._core.job import Job # Job seems unused here
from buttermilk._core.types import Record, RunRequest  # Import RunRequest
from buttermilk._core.variants import AgentVariants
from buttermilk.agents.fetch import FetchRecord
from buttermilk.bm import BM, bm, logger

BASE_DIR = Path(__file__).absolute().parent


class Orchestrator(BaseModel, ABC):
    """
    Abstract Base Class for orchestrators that manage the execution of agent-based flows.

    Orchestrators are responsible for setting up the execution environment, loading data,
    instantiating agents, managing the sequence of steps (either statically or dynamically),
    handling communication between agents (potentially via a runtime), interacting with
    a user interface if needed, collecting results, and cleaning up resources.

    Concrete subclasses must implement `_setup`, `_cleanup`, and `_execute_step`.
    They typically also override `_run` to define the main control loop logic.

    The Orchestrator is responsible for coordinating the execution of steps in a flow,
    managing agent interactions, handling data flow between components, and collecting results.

    Attributes:
        session_id (str): A unique identifier for this flow execution session
        description (str): Short description of the flow's purpose
        save (SaveInfo | None): Configuration for saving flow results
        data (Sequence[DataSource]): Data sources available to the flow
        agents (Mapping[str, AgentVariants]): Agent variants available to run in the flow
        params (dict): Flow-level parameters that can be used by agents

    """

    session_id: str = Field(
        default_factory=lambda: shortuuid.uuid()[:8],
        description="A unique session id for this set of flow runs.",
    )
    name: str = Field(
        ...,
        description="Friendly name of this flow",
    )
    description: str = Field(
        default_factory=shortuuid.uuid,
        description="Short description of this flow",
    )
    save: SaveInfo | None = Field(default=None)
    data: Sequence[DataSourceConfig] = Field(default_factory=list)
    agents: Mapping[str, AgentVariants] = Field(
        default_factory=dict,
        description="Agent factories available to run.",
    )
    params: dict = Field(
        default={},
        description="Flow-level parameters available for use by agents.",
        exclude=True,
    )

    _flow_data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)
    _records: list[Record] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        # Removed exclude_none=True, exclude_unset=True
    )

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Sequence[DataSourceConfig | dict]) -> list[DataSourceConfig]:
        """Ensures all data sources are proper DataSource objects.

        Args:
            value: A sequence of data sources, either DataSource objects or dictionaries

        Returns:
            list[DataSource]: List of DataSource objects

        """
        _data = []
        for source in value:
            if not isinstance(source, DataSourceConfig):
                source = DataSourceConfig(**source)
                _data.append(source)
        return _data

    @model_validator(mode="after")
    def validate_agents(self) -> Self:
        # Ensure that agents is a dict of AgentVariants specifications
        agent_dict = {}
        for step_name, defn in self.agents.items():
            if isinstance(defn, (AgentVariants)):
                agent_dict[step_name.upper()] = defn
            else:
                agent_dict[step_name.upper()] = AgentVariants(**defn)

        self.agents = agent_dict

        # initialise the data cache
        self._flow_data.init(list(self.agents.keys()))

        return self

    async def run(self, request: RunRequest | None = None) -> None:
        """
        Public entry point to start the orchestrator's flow execution.

        Sets up Weave tracing context and calls the internal `_run` method.

        Args:
            request: An optional RunRequest containing initial data or parameters for the flow.
        """
        client = bm.weave
        tracing_attributes = {**self.params, "session_id": self.session_id, "orchestrator": self.__repr_name__()}
        with weave.attributes(tracing_attributes):
            _traced = weave.op(
                self._run,
                call_display_name=f"{self.name} {self.params.get('criteria','')}",
            )
        try:
            output, call = await _traced.call(request=request)
        finally:
            client.finish_call(call)  # type: ignore (weave promises never to raise on .call())

        logger.info(f"Finished...")
        return

    async def _run(self, request: RunRequest | None = None) -> None:
        """Main execution method that sets up agents and manages the flow."""
        try:
            # Abstract setup method for subclasses (e.g., start runtime)
            await self._setup()  # Ensure _setup is defined or handled

            # Handle initial request if provided
            if request:
                if not request.records:
                    if request.uri or request.record_id:
                        # Fixed: Pass list(self.data)
                        fetch = FetchRecord(role="fetch", description="fetch records and urls", data=list(self.data))
                        fetch_output = await fetch._run(uri=request.uri, record_id=request.record_id)
                        # Fixed: Assign results list
                        if fetch_output and fetch_output.results:
                            self._records = fetch_output.results
                else:
                    # Store records from the request
                    self._records = request.records

            # Removed main loop logic - this should be implemented in subclasses
            # Subclasses should implement their own _run logic calling _setup,
            # handling steps, and _cleanup.

        except Exception as e:  # Catch any exception during setup/initial fetch
            logger.exception(f"Error during orchestrator initial setup/fetch: {e}")
            # Optionally re-raise as FatalError if appropriate
            # raise FatalError from e
        finally:
            # Ensure cleanup runs regardless of how the setup/fetch/run exits
            logger.info("Orchestrator cleaning up resources...")
            await self._cleanup()  # Ensure _cleanup is defined or handled
            logger.info("Orchestrator cleanup complete.")

    @abstractmethod
    async def _setup(self):  # Ensure abstract _setup method is defined
        """
        Abstract method for setting up orchestrator-specific resources.

        This could include initializing communication runtimes (like Autogen),
        connecting to external services, or instantiating agents.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    async def _cleanup(self):
        """
        Abstract method for cleaning up any resources allocated during setup or execution.

        Examples include stopping runtimes, closing connections, or releasing locks.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def _in_the_loop(self, step: StepRequest | None = None, prompt: str = "") -> ManagerResponse:
        """
        Placeholder for human-in-the-loop interaction or confirmation.

        Interactive orchestrators (like Selector) override this to communicate with
        the user interface agent, presenting proposed steps and receiving confirmation,
        feedback, or alternative instructions.

        The default implementation automatically confirms progression.

        Args:
            step: The proposed StepRequest for user confirmation (optional).
            prompt: An alternative prompt to display if no step is provided (optional).

        Returns:
            A ManagerResponse indicating confirmation status and any user input.
        """
        # Subclasses can override this to interact with a UI agent, etc.
        return ManagerResponse(confirm=True)

    async def execute(self, request: StepRequest) -> AgentOutput | None:
        """Execute a single step directly (potentially for testing or specific control flows).

        Args:
            request: The StepRequest defining the step to execute.

        Returns:
            AgentOutput | None: The output from the executed step, if any.

        """

        output = await self._execute_step(request)

        return output

    async def __call__(self, request: RunRequest | None = None) -> None:  # Accept RunRequest
        """Makes the orchestrator callable, allowing it to be used as a function.

        Args:
            request: Optional RunRequest input data for starting the flow

        Returns:
            None: This method typically doesn't return a value directly in this pattern.

        """
        # Pass the RunRequest (or None) to the run method
        await self.run(request=request)
        return  # __call__ typically doesn't return a value directly in this pattern

    @abstractmethod
    async def _execute_step(
        self,
        step: StepRequest,
    ) -> AgentOutput | None:
        """
        Abstract method to execute a single step of the flow using a specific agent.

        Subclasses must implement this to handle the actual execution mechanism,
        which might involve sending messages via a runtime (Autogen) or directly
        calling an agent instance (Batch).

        Args:
            step: The role name of the agent to execute.
            input: The prepared AgentInput for the step.

        Returns:
            The AgentOutput from the agent, or None if execution failed to produce output.
        """
        raise NotImplementedError

    async def _evaluate_step(
        self,
        output: AgentOutput,
        ground_truth_record: Record | None,
        criteria: Any | None,
        weave_call: Any | None,  # For logging evaluation to the trace
    ) -> None:
        """
        Evaluates the output of a step if possible and logs the result.

        Concrete orchestrators should override this to implement scorer lookup and execution.
        """
        # Default implementation does nothing
        # Make log message more generic as role isn't directly on output
        logger.debug(f"Base _evaluate_step called for an agent output. No evaluation performed.")
        pass


class OrchestratorProtocol(BaseModel):
    bm: BM
    flows: Mapping[str, Orchestrator]
    ui: Literal["console", "slackbot"]
    flow: str
    record_id: str = ""
    uri: str = ""
    prompt: str = ""
