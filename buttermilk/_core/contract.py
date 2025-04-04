
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage  # noqa
from autogen_core.models import FunctionExecutionResult
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.runner_types import Record
from buttermilk.utils.validators import make_list_validator

BASE_DIR = Path(__file__).absolute().parent

CONDUCTOR = "host"
MANAGER = "manager"
CLOSURE = "collector"
CONFIRM = "confirm"


class FlowProtocol(BaseModel):
    flow_name: str  # flow name
    description: str
    save: SaveInfo | None = Field(default=None)
    data: list[DataSource] | None = Field(default=[])
    agents: Mapping[str, Any] = Field(default={})
    orchestrator: str


class StepRequest(BaseModel):
    """Type definition for a request to execute a step in the flow execution.

    A StepRequest describes a request for a step to run, containing the agent role
    that should execute the step along with prompt and other execution parameters.

    Attributes:
        role (str): The agent role identifier to execute this step
        prompt (str): The prompt text to send to the agent
        description (str): Optional description of the step's purpose
        tool (str): A tool request to make
        arguments (dict): Additional key-value pairs needed for step execution
        source (str): Caller name

    """

    role: str
    prompt: str | None = Field(default=None)
    description: str | None = Field(default=None)
    tool: str | None = Field(default=None)
    arguments: dict[str, Any] = Field(default={})
    source: str | None = Field(default=None)


######
# Communication between Agents
#
class FlowMessage(BaseChatMessage):
    """A base class for all conversation messages."""

    content: str | None = Field(
        default=None,
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


class AgentInput(BaseChatMessage):
    """Base class for agent inputs with built-in validation"""

    _type = "InputRequest"
    role: str = Field(
        ...,
        description="The role of the agent(s) to which this request is made.",
    )
    source: str = Field(
        ...,
        description="The name of the agent that requested this step.",
    )
    context: list[Any] = Field(
        default=[],
        description="History or context to provide to the agent.",
    )
    _ensure_record_context_list = field_validator("records", "context", mode="before")(
        make_list_validator(),
    )


######
# Control communications
#
class ManagerMessage(FlowMessage):
    """OOB message to manage the flow.

    Usually involves an automated
    conductor or a human in the loop (or both).
    """

    _type = "ManagerMessage"


class ConductorRequest(ManagerMessage, AgentInput):
    """Request for input from the conductor."""

    role: str = CONDUCTOR
    _type = "ConductorRequest"


class ManagerRequest(ManagerMessage, StepRequest):
    """Request for input from the user"""

    role: str = MANAGER
    _type = "RequestForManagerInput"
    options: bool | list[str] | None = Field(
        default=None,
        description="Require answer from a set of options",
    )
    confirm: bool | None = Field(
        default=None,
        description="Response from user: confirm y/n",
    )


class ManagerResponse(ManagerRequest):
    """Response from the manager."""

    role: str = CONDUCTOR
    _type = "ManagerResponse"


class UserInstructions(FlowMessage):
    """Instructions from the user."""

    _type = "UserInput"

    confirm: bool = Field(
        default=False,
        description="Response from user: confirm y/n",
    )
    stop: bool = Field(default=False, description="Whether to stop the flow")


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


GroupchatMessages = Union[
    AgentOutput,
    ToolOutput,
    UserInstructions,
]

AllMessages = Union[
    FlowMessage,
    AgentInput,
    AgentOutput,
    ToolOutput,
    UserInstructions,
    ConductorRequest,
    ManagerRequest,
    ManagerMessage,
    GroupchatMessages,
]
