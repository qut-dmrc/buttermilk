from abc import ABC, abstractmethod
from ast import arguments
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

from buttermilk._core.agent import ChatCompletionContext
from buttermilk._core.config import DataSourceConfig, SaveInfo
from buttermilk._core.contract import AgentInput, StepRequest
from buttermilk._core.flow import KeyValueCollector
from buttermilk._core.job import Job
from buttermilk._core.types import Record
from buttermilk._core.variants import AgentVariants
from buttermilk.bm import BM

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

    session_id: str = Field(
        default_factory=shortuuid.uuid,
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
        for step_name in self.agents.keys():
            yield StepRequest(role=step_name, source=self.flow_name)

    @abstractmethod
    async def run(self, request: Any = None) -> None:
        """Starts a flow, given an incoming request.

        This is the main entry point for flow execution that must be implemented
        by subclasses with their specific orchestration logic.

        Args:
            request: Optional input data for starting the flow

        """
        # loop

        # save the results
        # flow_data ...
        while True:
            # Get next step in the flow
            step = await anext(self._get_next_step())
            request = await self._prepare_step(step)
            await self._execute_step(step)
            # execute step

    @abstractmethod
    async def _execute_step(
        self,
        step: StepRequest,
    ) -> None:
        raise NotImplementedError()

    async def __call__(self, request=None) -> Job:
        """Makes the orchestrator callable, allowing it to be used as a function.

        Args:
            request: Optional input data for starting the flow

        Returns:
            Job: A job representing the flow execution

        """
        return await self.run(request=request)

    async def _prepare_step(
        self,
        step: StepRequest,
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
        config = self.agents[step.role]

        input_map = dict(config.inputs)

        # Fill inputs based on input map
        inputs = self._flow_data._resolve_mappings(input_map)

        # add reserved keyword inputs
        inputs.update(
            dict(
                participants = "\n".join([f"- {id}: {step.description}" for id, step in self.agents.items()])
                prompt=step.prompt,
            )
        )

        # add placeholder variables as llm messages
        placeholders = dict(records=[r.as_message() for r in self._records], context=await self._model_context.get_messages())

        return AgentInput(
            role=step.role,
            source=self.flow_name,
            inputs=inputs,
            placeholders=placeholders,
            parameters=step.arguments,
        )


class OrchestratorProtocol(BaseModel):
    bm: BM
    flows: Mapping[str, Orchestrator]
    ui: Literal["console", "slackbot"]
