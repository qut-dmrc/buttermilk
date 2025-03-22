
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, Union

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
)

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.runner_types import Record
from buttermilk.utils.validators import make_list_validator

BASE_DIR = Path(__file__).absolute().parent


class FlowProtocol(Protocol):
    save: SaveInfo
    data: Sequence[DataSource]
    steps: Sequence["Agent"]

    async def run(self, job: "Job") -> "Job": ...

    async def __call__(self, job: "Job") -> "Job": ...


class OrchestratorProtocol(Protocol):
    bm: "BM"
    flows: Mapping[str, FlowProtocol]


######
# Communication between Agents
#
class FlowMessage(BaseModel):
    """A base class for all conversation messages."""

    _type = "GenericFlowMessage"
    content: str = Field(
        default="",
        description="The human-readable digest representation of the message.",
    )

    records: list[Record] = Field(
        default=[],
        description="Records relevant to this message",
    )

    error: list[str] = Field(
        default_factory=list,
        description="A list of errors that occurred during the agent's execution",
    )

    payload: dict[str, Any] = Field(
        default={},
        description="The data to or response from the agent",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the message."""

    _ensure_error_list = field_validator("error", mode="before")(
        make_list_validator(),
    )

    @computed_field
    @classmethod
    def type(cls) -> str:
        return cls._type


######
# Control communications
#
class ManagerMessage(FlowMessage):
    """OOB message to manage the flow.

    Usually involves an automated
    conductor or a human in the loop (or both).
    """

    _type = "ManagerMessage"


class UserConfirm(ManagerMessage):
    confirm: bool = Field(default=False, description="Ready to proceed?")
    stop: bool = Field(default=False, description="Whether to stop the flow")


class AgentInput(FlowMessage):
    """Base class for agent inputs with built-in validation"""

    _type = "InputRequest"
    context: list[Any] = Field(
        default=[],
        description="History or context to provide to the agent.",
    )
    _ensure_record_context_list = field_validator("records", "context", mode="before")(
        make_list_validator(),
    )


class AgentOutput(FlowMessage):
    """Base class for agent outputs with built-in validation"""

    _type = "Answer"


AgentMessages = Union[FlowMessage, AgentInput, AgentOutput]
