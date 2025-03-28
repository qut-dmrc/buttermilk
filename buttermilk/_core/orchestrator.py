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


class Orchestrator(BaseModel, ABC):
    """Runs a single instance of a flow."""

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
        _data = []
        for source in value:
            if not isinstance(source, DataSource):
                source = DataSource(**source)
                _data.append(source)
        return _data

    @field_validator("agents", mode="before")
    @classmethod
    def validate_agents(cls, value: dict) -> dict[str, AgentVariants]:
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
        """Starts a flow, given an incoming request"""
        self._flow_data = copy.deepcopy(self.data)  # process if needed
        # add request data
        # ...
        for step_name, step in self.agents.items():
            self._flow_data[step_name] = await step(self._flow_data)

        # save the results
        # flow_data ...

    async def __call__(self, request=None) -> Job:
        return await self.run(request=request)

    async def _prepare_inputs(self, step_name: str) -> dict[str, Any]:
        """Fill inputs according to specification.

        Includes several special case keywords:
            - "participants": list of agents in the flow
            - "content": list of string, fulltext from all records
            - "history": list of history messages in string format
            - "context": list of history messages in message format
            - "record": list of InputRecords"
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
        """Execute a step by sending requests to relevant agents and collecting responses"""
        # Send message with appropriate inputs for this step
        mapped_inputs = await self._prepare_inputs(step_name=step_name)
        mapped_inputs.update(**inputs)
        records = mapped_inputs.pop("record", [])

        return AgentInput(
            agent_id=self.flow_name,
            agent_name=source,
            content=prompt,
            inputs=mapped_inputs,
            records=records,
        )
