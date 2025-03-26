
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Union

from autogen_core.models import FunctionExecutionResult
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
)

from buttermilk._core.config import DataSource, Project, SaveInfo
from buttermilk._core.runner_types import Record
from buttermilk.utils.validators import make_list_validator

BASE_DIR = Path(__file__).absolute().parent


class FlowProtocol(Protocol):
    name: str  # flow name
    save: SaveInfo
    data: Sequence[DataSource]
    agents: Sequence["Agent"]
    orchestrator: str

    async def run(self, job: "Job") -> "Job": ...

    async def __call__(self, job: "Job") -> "Job": ...


class OrchestratorProtocol(Protocol):
    bm: Project
    flows: Mapping[str, FlowProtocol]
    ui: Literal["console", "slackbot"]


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

    inputs: dict[str, Any] = Field(
        default={},
        description="The data to provide the agent",
    )

    outputs: dict[str, Any] = Field(
        default={},
        description="The data returned from the agent",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the message."""

    _ensure_error_list = field_validator("error", mode="before")(
        make_list_validator(),
    )

    @computed_field
    @property
    def type(self) -> str:
        return self._type


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
    _type = "UserConfirm"
    confirm: bool = Field(default=False, description="Ready to proceed?")
    stop: bool = Field(default=False, description="Whether to stop the flow")


class UserInput(FlowMessage):
    _type = "User"
    """Instructions from the user."""


class AgentInput(FlowMessage):
    """Base class for agent inputs with built-in validation"""

    _type = "InputRequest"
    agent_id: str = Field(
        ...,
        description="The ID of the agent to which this request is made.",
    )
    context: list[Any] = Field(
        default=[],
        description="History or context to provide to the agent.",
    )
    _ensure_record_context_list = field_validator("records", "context", mode="before")(
        make_list_validator(),
    )


class AgentOutput(FlowMessage):
    """Base class for agent outputs with built-in validation"""

    _type = "Agent"

    agent_id: str = Field(
        ...,
        description="The ID of the agent that generated this output.",
    )
    agent_name: str = Field(
        ...,
        description="The name of the agent that generated this output.",
    )


class ToolOutput(FunctionExecutionResult):
    payload: Any = Field(
        ...,
        description="The output of the tool.",
    )


AgentMessages = Union[FlowMessage, AgentInput, AgentOutput]
