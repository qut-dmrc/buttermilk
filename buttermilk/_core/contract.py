from asyncio import streams
from collections.abc import Mapping
from email.policy import strict
from enum import Enum
from math import e
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_core.models import FunctionExecutionResult
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)
import shortuuid

from buttermilk._core.exceptions import ProcessingError


from .config import DataSourceConfig, SaveInfo, Tracing
from .types import Record, _global_run_id
from buttermilk.utils.validators import make_list_validator

from autogen_core.models import LLMMessage

BASE_DIR = Path(__file__).absolute().parent

CONDUCTOR = "HOST"
MANAGER = "MANAGER"
CLOSURE = "COLLECTOR"
CONFIRM = "CONFIRM"
COMMAND_SYMBOL = "!"
END = "END"
WAIT = "WAIT"

class FlowProtocol(BaseModel):
    name: str  # friendly flow name
    description: str
    save: SaveInfo | None = Field(default=None)
    data: list[DataSourceConfig] | None = Field(default=[])
    agents: Mapping[str, Any] = Field(default={})
    orchestrator: str
    params: dict = Field(default={})


class StepRequest(BaseModel):
    """Type definition for a request to execute a step in the flow execution.

    A StepRequest describes a request for a step to run, containing the agent role
    that should execute the step along with prompt and other execution parameters.

    Attributes:
        role (str): The agent role identifier to execute this step
        prompt (str): The prompt text to send to the agent
        description (str): Optional description of the step's purpose

    """

    role: str = Field(..., description="the ROLE name (not the description) of the next expert to respond.")
    prompt: str = Field(..., description="The prompt text to send to the agent.")
    description: str = Field(..., description="Brief explanation of the next step.", exclude=True)
    # tool: str = Field(default="", description="The tool to invoke, if any.")
    # arguments: dict[str, Any] = Field(description="Arguments to provide to the tool, if any.")

    @field_validator("role")
    @classmethod
    def case_fields(cls, v: str) -> str:
        """Ensure role is uppercase."""
        if v:
            return v.upper()
        return v


class FlowEvent(BaseModel):
    """For communication outside the groupchat."""
    source: str
    content: str


class ErrorEvent(FlowEvent):
    """Communicate errors to host and UI."""


######
# Communication between Agents
#
class FlowMessage(BaseModel):
    """A base class for all conversation messages."""

    error: list[str] = Field(
        default_factory=list,
        description="A list of errors that occurred during the agent's execution",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata about the message.")

    _ensure_error_list = field_validator("error", mode="before")(
        make_list_validator(),
    )

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

    _ensure_input_list = field_validator("context", "records", mode="before")(
        make_list_validator(),
    )


class UserInstructions(FlowMessage):
    """Instructions from the user."""

    records: list[Record] = Field(
        default=[],
        description="A list of records to include in the prompt",
    )

    prompt: str = Field(
        default="",
        description="A prompt to include",
    )
    confirm: bool = Field(
        default=False,
        description="Response from user: confirm y/n",
    )
    stop: bool = Field(default=False, description="Whether to stop the flow")


class TracingDetails(BaseModel):
    weave: str = Field(..., validate_default=True, exclude=True)

    @field_validator("weave")
    @classmethod
    def _get_tracing_links(cls, value) -> str:
        import weave
        from buttermilk.bm import logger

        try:
            return weave.get_current_call().ref.id
        except Exception as e:
            msg = f"Unable to get weave call: {e}"
            logger.error(msg)
            raise ProcessingError(msg)


class AgentOutput(FlowMessage):
    """Base class for agent outputs with built-in validation"""

    agent_id: str = Field(..., description="The agent that generated this output.")
    run_id: str = Field(default=_global_run_id)
    call_id: str = Field(
        default_factory=lambda: shortuuid.uuid(),
        description="A unique ID for this response.",
    )

    inputs: AgentInput | None = Field(
        default=None,
        description="Agent inputs",
    )

    messages: list[LLMMessage] = Field(
        default=[],
        description="A list of message inputs",
    )

    params: dict[str, Any] = Field(
        default={},
        description="Invocation settings provided to the agent",
    )
    prompt: str = Field(default="")
    outputs: Union[BaseModel, Dict[str, Any]] = Field(
        default={},
        description="The data returned from the agent",
    )
    tracing: TracingDetails = Field(default_factory=TracingDetails)

    _ensure_error_list = field_validator("error", mode="before")(
        make_list_validator(),
    )

    # This is needed because for some reason pydantic doesn't serialise the AgentReasons
    # properly in the outputs: Basemodel field.
    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # If outputs is a BaseModel, ensure it's properly dumped
        if isinstance(self.outputs, BaseModel):
            data["outputs"] = self.outputs.model_dump()
        return data

    @property
    def contents(self) -> str:
        return str(self.outputs)


######
# Control communications
#


class ManagerMessage(FlowMessage):
    """OOB message to manage the flow.

    Usually involves an automated
    conductor or a human in the loop (or both).
    """

    content: str | None = Field(
        default=None,
        description="The human-readable digest representation of the message.",
    )
    outputs: BaseModel | dict[str, Any] = Field(  # Changed type to match AgentOutput
        default={},
        description="Payload data",
    )
    agent_id: str = Field(
        default=CONDUCTOR,
        description="The ID of the agent that generated this output.",
    )


class ConductorRequest(ManagerMessage, AgentInput):
    """Request for input from the conductor."""


class ConductorResponse(ManagerMessage, AgentOutput):
    """Response to the conductor."""


class ManagerRequest(ManagerMessage, StepRequest):
    """Request for input from the user"""

    description: str = Field(default="Request input from user.")

    options: bool | list[str] | None = Field(
        default=None,
        description="Require answer from a set of options",
    )
    confirm: bool | None = Field(
        default=None,
        description="Response from user: confirm y/n",
    )
    halt: bool = Field(default=False, description="Whether to stop the flow")


class ManagerResponse(FlowMessage):
    """Response from the manager with feedback and variant selection capabilities."""

    confirm: bool = True
    halt: bool = False
    prompt: Optional[str] = None
    selection: Optional[str] = None  # For multiple choice responses


#########
# Function messages
class ToolOutput(FunctionExecutionResult):
    role: str = Field(..., description="The role the tool provides")

    results: Any = Field(default_factory=dict)
    messages: list[LLMMessage] = Field(default_factory=list)  # Changed UserMessage to LLMMessage
    args: list[str] | list[dict[str, Any]] | dict[str, Any] = Field(default_factory=dict)

    content: str
    call_id: str = "unknown"
    is_error: bool | None = False

    send_to_ui: bool = False


#######
# Coordination Messages
class TaskProcessingStarted(BaseModel):
    """Sent by an agent after completing one sequential task."""

    agent_id: str = Field(..., description="ID of the agent sending the notification")
    role: str = Field(..., description="Task role name")
    task_index: int = Field(..., description="Index of the task that was just completed")


class TaskProcessingComplete(TaskProcessingStarted):
    """Sent by an agent after completing one sequential task."""

    agent_id: str = Field(..., description="ID of the agent sending the notification")
    role: str = Field(..., description="Task role name")
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
    ManagerMessage, ManagerRequest, ManagerResponse, TaskProcessingComplete, TaskProcessingStarted, ConductorResponse, ConductorRequest
]

GroupchatMessageTypes = Union[
    AgentOutput,
    ToolOutput,
    UserInstructions,
]

AllMessages = Union[GroupchatMessageTypes, OOBMessages, AgentInput, ProceedToNextTaskSignal, HeartBeat]
