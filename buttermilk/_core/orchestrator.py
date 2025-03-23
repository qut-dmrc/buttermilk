import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
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
from buttermilk._core.flow import FlowVariableRouter
from buttermilk._core.job import Job
from buttermilk._core.variants import AgentVariants
from buttermilk.exceptions import ProcessingError

BASE_DIR = Path(__file__).absolute().parent


PLACEHOLDER_VARIABLES = ["participants", "content", "history", "context", "record"]


class Orchestrator(BaseModel, ABC):
    """Runs a single instance of a flow."""

    session_id: str = Field(
        default_factory=shortuuid.uuid,
        description="A unique session id for this set of flow runs.",
    )
    name: str
    save: SaveInfo | None = Field(default=None)
    data: Sequence[DataSource] = Field(default_factory=list)
    steps: Sequence[AgentVariants] = Field(
        default_factory=list,
        description="Agent factories available to run.",
    )

    _flow_data: FlowVariableRouter = PrivateAttr(default_factory=FlowVariableRouter)
    _records: list = PrivateAttr(default_factory=list)
    _context: UnboundedChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    _history: list[str] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )

    @field_validator("steps", mode="before")
    @classmethod
    def validate_steps(cls, value):
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [
                AgentVariants(**step) if not isinstance(step, AgentVariants) else step
                for step in value
            ]
        return value

    @abstractmethod
    async def run(self, request: Any = None) -> None:
        """Starts a flow, given an incoming request"""
        self._flow_data = copy.deepcopy(self.data)  # process if needed
        # add request data
        # ...
        for step in self.steps:
            self._flow_data[step.id] = await step(self._flow_data)

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
        config = None
        for config in self.steps:
            if config.id == step_name:
                break
        if not config:
            raise ProcessingError(f"Cannot find config for step {step_name}.")

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
                    input_dict[value] = "\n".join(self._history)
                elif value == "context":
                    # Get the chat context and records
                    input_dict[value] = await self._context.get_messages()
                elif value == "record":
                    input_dict[value] = self._records
                elif value == "participants":
                    participants = [
                        f"- {step.id}: {step.description}" for step in self.steps
                    ]
                    input_dict[value] = "\n".join(participants)

        input_dict.update(self._flow_data._resolve_mappings(config.inputs))

        return input_dict
