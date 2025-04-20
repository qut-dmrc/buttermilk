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

from buttermilk._core import AgentOutput, TaskProcessingComplete
from buttermilk._core.agent import ChatCompletionContext, FatalError, ProcessingError
from buttermilk._core.config import DataSourceConfig, SaveInfo
from buttermilk._core.contract import END, AgentInput, ManagerResponse, StepRequest
from buttermilk._core.flow import KeyValueCollector

# from buttermilk._core.job import Job # Job seems unused here
from buttermilk._core.types import Record, RunRequest  # Import RunRequest
from buttermilk._core.variants import AgentVariants
from buttermilk.agents.fetch import FetchRecord
from buttermilk.bm import BM, bm, logger

BASE_DIR = Path(__file__).absolute().parent


class Orchestrator(BaseModel, ABC):
    """Runs a single instance of a flow.

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
    history: list = Field(default=[])

    _flow_data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)
    _model_context: ChatCompletionContext
    _records: list[Record] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
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
                agent_dict[step_name.lower()] = defn
            else:
                agent_dict[step_name.lower()] = AgentVariants(**defn)

        self.agents = agent_dict

        # initialise the data cache
        self._flow_data.init(list(self.agents.keys()))

        self._model_context = UnboundedChatCompletionContext(initial_messages=self.history)

        return self

    async def run(self, request: RunRequest | None = None) -> None:
        """Starts a flow, given an incoming request."""
        client = bm.weave
        tracing_attributes = {**self.params, "session_id": self.session_id, "orchestrator": self.__repr_name__()}
        with weave.attributes(tracing_attributes):
            _traced = weave.op(
                self._run,
                call_display_name=f"{self.name} {self.params.get('criteria','')}",
            )
        output, call = await _traced.call(request=request)
        client.finish_call(call)
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
                        fetch = FetchRecord(role="fetch", description="fetch records and urls", data=self.data)
                        record = await fetch._run(uri=request.uri, record_id=request.record_id)
                        self._records = [record]
                else:
                    # Store records from the request
                    self._records = request.records

            # Main loop to get and execute subsequent steps
            while True:
                try:
                    # Loop until we receive an error or completion
                    await asyncio.sleep(1)  # Small delay to prevent busy-waiting

                    # Get next step from host
                    next_step_request = await self._get_host_suggestion()
                    if not next_step_request:
                        # No next step determined, maybe wait or check status?
                        # Depending on orchestrator logic, this might mean completion or idle state.
                        logger.debug("No next step determined by _get_next_step.")
                        await asyncio.sleep(5)  # Wait before checking again
                        continue

                    # Optional human-in-the-loop confirmation
                    if not await self._in_the_loop(next_step_request):  # Use correct variable
                        logger.info("User did not confirm plan. Waiting for new instructions.")
                        # Logic to handle user rejection/new input needed here
                        continue

                    # Prepare and execute the step
                    step_input = await self._prepare_step(next_step_request)
                    await self._execute_step(step=next_step_request, input=step_input)

                except ProcessingError as e:
                    # Non-fatal error, log and continue loop
                    logger.error(f"Processing error in orchestrator run: {e}")
                    # Optionally, inform user or trigger error handling agent
                    continue
                except StopAsyncIteration:
                    logger.info("Orchestrator loop stopped by StopAsyncIteration (likely flow completion).")
                    break  # Exit the main loop
                except KeyboardInterrupt:
                    logger.info("Orchestrator run interrupted by user (KeyboardInterrupt).")
                    raise  # Re-raise to allow clean exit
                except FatalError as e:
                    logger.exception(f"Fatal error encountered in orchestrator run: {e}")
                    raise  # Re-raise fatal errors
                except Exception as e:
                    logger.exception(f"Unexpected error in orchestrator run loop: {e}")
                    raise FatalError from e  # Wrap unexpected errors as Fatal

        except FatalError as e:
            # Catch fatal errors originating from setup or initial step
            logger.exception(f"Fatal error during orchestrator setup/initial step: {e}")
        finally:
            # Ensure cleanup runs regardless of how the loop exits
            logger.info("Orchestrator cleaning up resources...")
            await self._cleanup()  # Ensure _cleanup is defined or handled
            logger.info("Orchestrator cleanup complete.")

    @abstractmethod
    async def _setup(self):  # Ensure abstract _setup method is defined
        """Abstract method for setting up orchestrator resources (e.g., runtime)."""

        raise NotImplementedError

    @abstractmethod
    async def _cleanup(self):
        """Abstract method for cleaning up resources (e.g., stopping runtimes)."""
        raise NotImplementedError

    async def _in_the_loop(self, step: StepRequest | None = None, prompt: str = "") -> ManagerResponse:
        """Placeholder for human-in-the-loop confirmation. Default is True."""
        # Subclasses can override this to interact with a UI agent, etc.
        return ManagerResponse(confirm=True)

    async def execute(self, request: StepRequest) -> AgentOutput | None:
        """Execute a single step directly (potentially for testing or specific control flows).

        Args:
            request: The StepRequest defining the step to execute.

        Returns:
            AgentOutput | None: The output from the executed step, if any.

        """
        step_input = await self._prepare_step(request)
        # Assuming direct execution also needs the bm instance
        # Note: weave tracing might need adjustment if execute is used differently than _run
        # Weave tracing is now expected within the agent's handle_message or _process
        output = await self._execute_step(step=request.role, input=step_input)
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

    async def _prepare_step(
        self,
        request: StepRequest,
    ) -> AgentInput:
        """Create an AgentInput message for sending to an agent.

        Prepares a message with the appropriate inputs and context for the target agent.

        Resolves special keywords and mappings to provide the appropriate inputs
        for the given step.

        Special keywords include:
            - "participants": list of agents in the flow
            - "context": list of history messages
            - "records": list of InputRecords
            - "prompt": question from the user

        Args:
            step: Definition of inputs for the step

        Returns:
            AgentInput: A prepared message that can be sent to an agent

        """
        config = self.agents[request.role]

        input_map = dict(config.inputs)

        # Fill inputs based on input map
        inputs = self._flow_data._resolve_mappings(input_map)

        return AgentInput(
            inputs=inputs,
            context=await self._model_context.get_messages(),
            records=self._records,
            prompt=request.prompt,
        )

    @abstractmethod
    async def _execute_step(
        self,
        step: str,
        input: AgentInput,
    ) -> AgentOutput | None:
        """Abstract method to execute a single step using an agent."""
        raise NotImplementedError


class OrchestratorProtocol(BaseModel):
    bm: BM
    flows: Mapping[str, Orchestrator]
    ui: Literal["console", "slackbot"]
    flow: str
    record_id: str = ""
    uri: str = ""
    prompt: str = ""
