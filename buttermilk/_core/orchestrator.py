import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

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

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.contract import AgentInput, StepRequest
from buttermilk._core.flow import FlowVariableRouter
from buttermilk._core.job import Job
from buttermilk._core.runner_types import Record
from buttermilk._core.variants import AgentVariants
from buttermilk.bm import BM

BASE_DIR = Path(__file__).absolute().parent


PLACEHOLDER_VARIABLES = ["participants", "content", "history", "context", "records"]


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
        history (list[str]): List of messages previously exchanged between agents

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
    data: Sequence[DataSource] = Field(default_factory=list)
    agents: Mapping[str, AgentVariants] = Field(
        default_factory=dict,
        description="Agent factories available to run.",
    )
    params: dict = Field(
        default={},
        description="Flow-level parameters available for use by agents.",
        exclude=True,
    )
    _flow_data: FlowVariableRouter = PrivateAttr(default_factory=FlowVariableRouter)
    _records: list[Record] = PrivateAttr(default_factory=list)
    _context: UnboundedChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    history: list[str] = Field(
        default_factory=list,
        description="List of messages previously exchanged between agents.",
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )

    @field_validator("history", mode="before")
    @classmethod
    def _parse_history(cls, value: Sequence[dict[str, str] | str]) -> list[str]:
        """Converts different history format types to a consistent string format.

        Args:
            value: A sequence of history items, either dictionaries or strings

        Returns:
            list[str]: History items in a consistent string format

        """
        history = []
        for item in value:
            if isinstance(item, str):
                history.append(item)
            elif isinstance(item, dict):
                history.append(f"{item['type']}: {item['content']}")

        return history

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Sequence[DataSource | dict]) -> list[DataSource]:
        """Ensures all data sources are proper DataSource objects.

        Args:
            value: A sequence of data sources, either DataSource objects or dictionaries

        Returns:
            list[DataSource]: List of DataSource objects

        """
        _data = []
        for source in value:
            if not isinstance(source, DataSource):
                source = DataSource(**source)
                _data.append(source)
        return _data

    @model_validator(mode="after")
    def validate_agents(self) -> Self:
        # Ensure that agents is a dict of AgentVariants specifications
        agent_dict = {}
        for step_name, defn in self.agents.items():
            if isinstance(defn, (AgentVariants)):
                agent_dict[step_name] = defn
            else:
                agent_dict[step_name] = AgentVariants(**defn)

        self.agents = agent_dict

        # initialise the data cache
        self._flow_data.init(self.agents.keys())
        return self

    @abstractmethod
    async def run(self, request: Any = None) -> None:
        """Starts a flow, given an incoming request.

        This is the main entry point for flow execution that must be implemented
        by subclasses with their specific orchestration logic.

        Args:
            request: Optional input data for starting the flow

        """
        self._flow_data = copy.deepcopy(self.data)  # process if needed
        # add request data
        # ...
        for step_name, step in self.agents.items():
            self._flow_data[step_name] = await step(self._flow_data)

        # save the results
        # flow_data ...

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
            - "content": list of string, fulltext from all records
            - "history": list of history messages in string format
            - "context": list of history messages in message format
            - "records": list of InputRecords"
            - "prompt": question from the user

        Args:
            step: Definition of inputs for the step

        Returns:
            AgentInput: A prepared message that can be sent to an agent

        """
        config = self.agents[step.role]

        input_dict = dict(config.inputs)

        # Overwrite any of the input dict values that are mappings to other data
        input_dict.update(self._flow_data._resolve_mappings(input_dict))

        # Add any arguments from the step request
        input_dict.update(step.arguments)

        # Special handling for named placeholder keywords
        input_dict["content"] = [
            f"{rec.record_id}: {rec.fulltext}" for rec in self._records
        ]
        input_dict["history"] = "\n".join(self.history)
        input_dict["participants"] = "\n".join([
            f"- {id}: {step.description}" for id, step in self.agents.items()
        ])
        input_dict["prompt"] = step.prompt

        return AgentInput(
            agent_role=step.role,
            agent_id=self.flow_name,
            content=step.prompt,
            context=await self._context.get_messages(),
            inputs=input_dict,
            records=self._records,
        )


class OrchestratorProtocol(BaseModel):
    bm: BM
    flows: Mapping[str, Orchestrator]
    ui: Literal["console", "slackbot"]
