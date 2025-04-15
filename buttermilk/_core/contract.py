from asyncio import streams
from collections.abc import Mapping
from enum import Enum
from math import e
from pathlib import Path
from typing import Any, Literal, Union
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_core.models import FunctionExecutionResult
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
)
import shortuuid


from .config import DataSourceConfig, SaveInfo
from .types import Record
from buttermilk.utils.validators import make_list_validator

from autogen_core.models import LLMMessage

BASE_DIR = Path(__file__).absolute().parent

CONDUCTOR = "host"
MANAGER = "manager"
CLOSURE = "collector"
CONFIRM = "confirm"
COMMAND_SYMBOL = "!"

class FlowProtocol(BaseModel):
    flow_name: str  # flow name
    description: str
    save: SaveInfo | None = Field(default=None)
    data: list[DataSourceConfig] | None = Field(default=[])
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

    """

    role: str
    prompt: str | None = Field(default=None)
    description: str | None = Field(default=None)
    tool: str | None = Field(default=None)
    arguments: dict[str, Any] = Field(default={})

    @field_validator("role")
    @classmethod
    def lowercase_fields(cls, v: str) -> str:
        """Ensure source and role are lowercase."""
        if v:
            return v.lower()
        return v


class FlowEvent(BaseModel):
    """For communication outside the groupchat."""
    _type = "FlowEvent"
    source: str
    role: str
    content: str

class ErrorEvent(FlowEvent):
    """Communicate errors to host and UI."""

######
# Communication between Agents
#
class FlowMessage(BaseModel):
    """A base class for all conversation messages."""

    _type = "FlowMessage"

    role: str = Field(
        ...,
        description="The role of the agent that generated this output.",
    )
    error: list[str] = Field(
        default_factory=list,
        description="A list of errors that occurred during the agent's execution",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the message."""

    @computed_field
    @property
    def type(self) -> str:
        return self._type

    @computed_field
    @property
    def is_error(self) -> bool:
        if self.error:
            return True
        return False


class AgentInput(FlowMessage):
    """Base class for agent inputs with built-in validation"""

    inputs: dict[str, Any] = Field(
        default={},
        description="The data to provide the agent",
    )
    parameters: dict[str, Any] = Field(
        default={},
        description="Task-specific settings to provide the agent",
    )

    context: list[LLMMessage] = Field(
        default=[],
        description="A list of messages to include in the prompt",
    )
    records: list[Record] = Field(
        default=[],
        description="A list of records to include in the prompt",
    )

    prompt: str = Field(
        default="",
        description="A prompt to include",
    )

    _type = "InputRequest"

    _ensure_input_list = field_validator("context", "records", mode="before")(
        make_list_validator(),
    )
class UserInstructions(FlowMessage):
    """Instructions from the user."""

    _type = "UserInput"
    content: str = Field(default="")

    confirm: bool = Field(
        default=False,
        description="Response from user: confirm y/n",
    )
    stop: bool = Field(default=False, description="Whether to stop the flow")


class AgentOutput(FlowMessage):
    """Base class for agent outputs with built-in validation"""

    _type = "Agent"
    call_id: str = Field(
        default_factory=lambda: shortuuid.uuid()[:8],
        description="A unique ID for this response.",
    )
    inputs: AgentInput | None = Field(default=None)
    params: dict[str, Any] = Field(
        default={},
        description="Invocation settings provided to the agent",
    )
    content: str | None = Field(
        default=None,
        description="The human-readable digest representation of the message.",
    )
    outputs: type[BaseModel] | dict[str, Any] = Field(
        default={},
        description="The data returned from the agent",
    )
    records: list[Record] = Field(
        default=[],
        description="A list of records to include in the prompt",
    )
    internal_messages: list[FlowMessage] = Field(
        default_factory=list,
        description="Messages generated along the way to the final response",
    )

    _ensure_error_list = field_validator("error", mode="before")(
        make_list_validator(),
    )

    _ensure_record_context_list = field_validator("internal_messages", "records", mode="before")(
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

    content: str | None = Field(
        default=None,
        description="The human-readable digest representation of the message.",
    )
    outputs: dict[str, Any] = Field(
        default={},
        description="Payload data",
    )
    agent_id: str = Field(default=CONDUCTOR, description="The ID of the agent that generated this output.",
    )
    role: str = Field(
        default=CONDUCTOR,
        description="The role of the agent that generated this output.",
    )
class ConductorRequest(ManagerMessage, AgentInput):
    """Request for input from the conductor."""

    _type = "ConductorRequest"

class ConductorResponse(ManagerMessage, AgentOutput):
    """Response to the conductor."""

    _type = "ConductorResponse"


class ManagerRequest(ManagerMessage, StepRequest):
    """Request for input from the user"""

    _type = "RequestForManagerInput"
    options: bool | list[str] | None = Field(
        default=None,
        description="Require answer from a set of options",
    )
    confirm: bool | None = Field(
        default=None,
        description="Response from user: confirm y/n",
    )
    halt: bool = Field(default=False, description="Whether to stop the flow")


class ManagerResponse(ManagerRequest):
    """Response from the manager."""

    _type = "ManagerResponse"

#########
# Function messages
class ToolOutput(FunctionExecutionResult):
    role: str = Field(..., description="The role the tool provides")

    results: Any =  Field(default_factory=dict)
    messages: list[UserMessage] = Field(default_factory=list)
    args: list[str] | list[dict[str,Any]] | dict[str, Any] = Field(default_factory=dict)

    content: str
    call_id: str = "unknown"
    is_error: bool |None = False

    send_to_ui: bool = False

#######
# Coordination Messages

class TaskProcessingComplete(BaseModel):
    """Sent by an agent after completing one sequential task."""
    role: str = Field(..., description="ID of the agent sending the notification")
    task_index: int = Field(..., description="Index of the task that was just completed")
    more_tasks_remain: bool = Field(..., description="True if the agent has more sequential tasks to process for the current input")
    is_error: bool = Field(default=False, description="True if the task resulted in an error")

class ProceedToNextTaskSignal(BaseModel):
    """Sent by a controller to signal an agent to process its next sequential task."""
    target_agent_id: str = Field(..., description="ID of the agent that should proceed")
    model_config = {"extra": "allow"}

class HeartBeat(BaseModel):
    go_next: bool = Field(..., description="True if the agent should proceed to the next task")

#######
# Unions

OOBMessages = Union[
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    
]

GroupchatMessageTypes = Union[
    AgentOutput,
    ToolOutput,
    UserInstructions,
    
]

AllMessages = Union[
    GroupchatMessageTypes,
    OOBMessages,
    AgentInput,ProceedToNextTaskSignal,
    ConductorRequest,ConductorResponse,HeartBeat
]
