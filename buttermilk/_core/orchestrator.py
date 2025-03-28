import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import shortuuid
from autogen_core.model_context import UnboundedChatCompletionContext
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.contract import AgentInput
from buttermilk._core.flow import FlowVariableRouter
from buttermilk._core.job import Job
from buttermilk._core.variants import AgentVariants

BASE_DIR = Path(__file__).absolute().parent


PLACEHOLDER_VARIABLES = ["participants", "content", "history", "context", "record"]


class StepDefinition(BaseModel):
    """Type definition for a step in the flow execution.

    A StepDefinition describes a single step within a flow, containing the agent role
    that should execute the step along with prompt and other execution parameters.

    Attributes:
        role (str): The agent role identifier to execute this step
        prompt (str): The prompt text to send to the agent
        description (str): Optional description of the step's purpose
        arguments (dict): Additional key-value pairs needed for step execution

    """

    role: str
    prompt: str = Field(default="")
    description: str = Field(default="")
    arguments: dict[str, Any] = Field(default={})


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
    _records: list = PrivateAttr(default_factory=list)
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

    @field_validator("agents", mode="before")
    @classmethod
    def validate_agents(cls, value: dict) -> dict[str, AgentVariants]:
        """Ensures all agent definitions are proper AgentVariants objects.

        Args:
            value: Dictionary of agent definitions

        Returns:
            dict[str, AgentVariants]: Dictionary of agent variants mapped by step name

        """
        # Ensure that agents is a dict of AgentVariants specifications
        agent_dict = {}
        for step_name, defn in value.items():
            if isinstance(defn, (AgentVariants)):
                agent_dict[step_name] = defn
            else:
                agent_dict[step_name] = AgentVariants(**defn)
        return agent_dict

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

    async def _prepare_inputs(self, step_name: str) -> dict[str, Any]:
        """Prepares input data for a specific step in the flow.

        Resolves special keywords and mappings to provide the appropriate inputs
        for the given step.

        Special keywords include:
            - "participants": list of agents in the flow
            - "content": list of string, fulltext from all records
            - "history": list of history messages in string format
            - "context": list of history messages in message format
            - "record": list of InputRecords"

        Args:
            step_name: The name of the step to prepare inputs for

        Returns:
            dict[str, Any]: Dictionary of input data ready for the step

        """
        config = self.agents[step_name]

        input_dict = dict(config.inputs)
        # Overwrite any of the input dict values that are mappings to other data
        input_dict.update(self._flow_data._resolve_mappings(input_dict))

        for value in PLACEHOLDER_VARIABLES:
            if value in config.inputs:
                if value == "content":
                    records = [
                        f"{rec.record_id}: {rec.fulltext}" for rec in self._records
                    ]
                    input_dict[value] = records
                elif value == "history":
                    input_dict[value] = "\n".join(self.history)
                elif value == "context":
                    # Get the chat context and records
                    input_dict[value] = await self._context.get_messages()
                elif value == "record":
                    input_dict[value] = self._records
                elif value == "participants":
                    participants = [
                        f"- {id}: {step.description}"
                        for id, step in self.agents.items()
                    ]
                    input_dict[value] = "\n".join(participants)

        return input_dict

    async def _prepare_step_message(
        self,
        step_name: str,
        prompt: str = "",
        source: str = "",
        **inputs,
    ) -> AgentInput:
        """Creates an AgentInput message for sending to an agent.

        Prepares a message with the appropriate inputs and context for the target agent.

        Args:
            step_name: The name of the step to prepare a message for
            prompt: Optional prompt text to include in the message
            source: Optional source identifier
            **inputs: Additional input parameters to include

        Returns:
            AgentInput: A prepared message that can be sent to an agent

        """
        # Send message with appropriate inputs for this step
        mapped_inputs = await self._prepare_inputs(step_name=step_name)
        mapped_inputs.update(**inputs, prompt=prompt)
        records = mapped_inputs.pop("record", [])

        return AgentInput(
            agent_id=self.flow_name,
            agent_name=source,
            content=prompt,
            inputs=mapped_inputs,
            records=records,
        )
