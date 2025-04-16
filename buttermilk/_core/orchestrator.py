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

from buttermilk._core import AgentOutput
from buttermilk._core.agent import ChatCompletionContext, FatalError, ProcessingError
from buttermilk._core.config import DataSourceConfig, SaveInfo
from buttermilk._core.contract import AgentInput, StepRequest
from buttermilk._core.flow import KeyValueCollector
from buttermilk._core.job import Job
from buttermilk._core.types import Record
from buttermilk._core.variants import AgentVariants
from buttermilk.bm import BM, logger

BASE_DIR = Path(__file__).absolute().parent


class Orchestrator(BaseModel, ABC):
    """Runs a single instance of a flow.

    The Orchestrator is responsible for coordinating the execution of steps in a flow,
    managing agent interactions, handling data flow between components, and collecting results.

    Attributes:
        session_id (str): A unique identifier for this flow execution session
        flow_name (str): The name of the flow being executed
        description (str): Short description of the flow's purpose
        save (SaveInfo | None): Configuration for saving flow results
        data (Sequence[DataSource]): Data sources available to the flow
        agents (Mapping[str, AgentVariants]): Agent variants available to run in the flow
        params (dict): Flow-level parameters that can be used by agents

    """
    bm: BM = Field(...)
    session_id: str = Field(
        default_factory=lambda: shortuuid.uuid()[:8],
        description="A unique session id for this set of flow runs.",
    )
    flow_name: str
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
        self._flow_data.init(self.agents.keys())

        self._model_context = UnboundedChatCompletionContext(initial_messages=self.history)

        return self

    async def _get_next_step(self) -> AsyncGenerator[StepRequest, None]:
        """Determine the next step based on the current flow data.

        This generator yields a series of steps to be executed in sequence,
        with each step containing the role and prompt information.

        Yields:
            StepRequest: An object containing:
                - 'role' (str): The agent role/step name to execute
                - 'prompt' (str): The prompt text to send to the agent
                - Additional key-value pairs that might be needed for agent execution

        Example:
            >>> async for step in self._get_next_step():
            >>>     await self._execute_step(**step)

        """
        raise NotImplementedError()

    async def run(self, request: StepRequest | None = None) -> None:
        """Starts a flow, given an incoming request."""

        client = self.bm.weave

        _traced = weave.op(
            self._run,
            call_display_name=f"{self.flow_name} {self.session_id}",
        )
        output, call = await _traced.call(request=request)
        client.finish_call(call)
        logger.info(f"Finished...")
        return

    async def _run(self, request: StepRequest | None = None) -> None:
        """Main execution method that sets up agents and manages the flow.

        By default, this runs through a sequence of pre-defined steps.
        """
        try:
            await self._setup()
            step_generator = self._get_next_step()
            while True:
                try:
                    if request:
                        step = await self._prepare_step(request)
                        await self._execute_step(step)
                    await asyncio.sleep(0.1)

                    # Get next step in the flow
                    request = await anext(step_generator)

                    if not await self._in_the_loop(request):
                        # User did not confirm plan; go back and get new instructions
                        continue

                except ProcessingError as e:
                    # non-fatal error
                    logger.error(f"Error in Orchestrator run: {e}")
                    continue
                except (StopAsyncIteration, KeyboardInterrupt):
                    raise
                except FatalError:
                    raise
                except Exception as e:  # This is only here for debugging for now.
                    logger.exception(f"Error in Orchestrator.run: {e}")
                    raise FatalError from e

        except (StopAsyncIteration, KeyboardInterrupt):
            logger.info("Orchestrator.run: Flow completed.")
        except FatalError as e:
            logger.exception(f"Error in Orchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()

    @abstractmethod
    async def _setup(self):
        raise NotImplementedError

    @abstractmethod
    async def _cleanup(self):
        raise NotImplementedError

    async def _in_the_loop(self, step: StepRequest) -> bool:
        """Just run."""
        return True

    async def execute(self, request: StepRequest) -> AgentOutput | None:
        """Execute a single step in the flow.

        Args:
            request: The step to execute

        Returns:
            Step outputs

        """
        step = self._prepare_step(request)
        with weave.attributes(dict(session_id=self.session_id, flow_name=self.flow_name, params=self.params)):
            _traced = weave.op(
                self._execute_step,
                call_display_name=f"{request.role} {self.session_id}",
            )
        return await _traced(step)

    async def __call__(self, request=None) -> None:
        """Makes the orchestrator callable, allowing it to be used as a function.

        Args:
            request: Optional input data for starting the flow

        Returns:
            Job: A job representing the flow execution

        """
        await self.run(request=request)
        return

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
            role=request.role,
            inputs=inputs,
            context=await self._model_context.get_messages(),
            records=self._records,
            prompt=request.prompt,
        )

    @abstractmethod
    async def _execute_step(
        self,
        step: AgentInput,
    ) -> AgentOutput | None:
        # Run step
        raise NotImplementedError


class OrchestratorProtocol(BaseModel):
    bm: BM
    flows: Mapping[str, Orchestrator]
    ui: Literal["console", "slackbot"]
    orchestrator: str | None = None
    flow: str | None = None
    criteria: str | None = None
    record: str | None = None
