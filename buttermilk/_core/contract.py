"""
Defines the Pydantic models and constants used for data contracts and communication
within the Buttermilk agent framework.

This includes message types for agent inputs, outputs, control flow, status updates,
and interactions with manager/conductor roles.
"""

import datetime
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import shortuuid  # For generating unique IDs

# Import Autogen types used as base or components
from autogen_core.models import FunctionExecutionResult, LLMMessage
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from buttermilk._core.job import SessionInfo
from buttermilk.utils.validators import convert_omegaconf_objects, convert_omegaconf_objects_recursive, make_list_validator  # Pydantic validators

from .config import AgentConfig, AgentVariants, DataSourceConfig, SaveInfo  # Core configuration models
from .log import logger
from .types import Record, _global_run_id, RunRequest  # Core data types

# --- Constants ---

# Standard Agent Roles
CONDUCTOR = "HOST"  # Role name often used for the agent directing the flow (e.g., LLMHostAgent). Configurable?
MANAGER = "MANAGER"  # Role name often used for the user interface or human-in-the-loop agent.
CLOSURE = "COLLECTOR"  # Role name (Collector?) - Usage context unclear from surrounding files. TODO: Verify usage.
CONFIRM = "CONFIRM"  # Special agent/topic name used by AutogenOrchestrator for handling ManagerResponse.

# Special Symbols / States
COMMAND_SYMBOL = "!"  # Prefix potentially used to identify command messages 
END = "END"  # Special role/signal used by Conductor/Host to indicate the flow should terminate.
WAIT = "WAIT"  # Special role/signal used by Conductor/Host to indicate pausing/waiting.


# --- General Communication & Base Messages ---

class FlowEvent(BaseModel):
    """Base model for simple event messages within the flow (potentially OOB)."""

    source: str = Field(..., description="Identifier of the message source (e.g., agent ID).")
    content: str = Field(..., description="The main content of the event message.")

    def __str__(self) -> str:
        return self.content
class ErrorEvent(FlowEvent):
    """Specific event type for broadcasting errors."""

    # Inherits source and content, content typically holds the error message string.
    
    def __str__(self) -> str:
        return "ERROR: " + self.content


class FlowMessage(BaseModel):
    """
    Base class for most messages exchanged between agents and orchestrators.
    Includes common fields like error tracking and metadata.
    """

    error: list[str] = Field(
        default_factory=list,
        description="List of error messages accumulated during processing.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the message.",
    )

    # Validator to ensure 'error' is always a list, even if initialized with None or str.
    _ensure_error_list = field_validator("error", mode="before")(make_list_validator())

    @computed_field
    @property
    def is_error(self) -> bool:
        """Convenience property to check if the error list is non-empty."""
        return bool(self.error)

    def set_error(self, error_msg: str) -> None:
        """Appends an error message to the error list."""
        if not isinstance(self.error, list):  # Ensure error is a list
            self.error = []
        if error_msg:
            self.error.append(error_msg)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
        exclude={"is_error"},
    )  # type: ignore

    def __str__(self) -> str:
        return self.model_dump_json()
class TracingDetails(BaseModel):
    """Holds tracing identifiers."""

    # Default factory attempts to get current Weave call ID when an AgentOutput is created.
    # This might happen outside a Weave trace context if AgentOutput is created manually.
    weave: str = Field(default="", validate_default=True, exclude=True)  # Exclude from standard serialization.

    @field_validator("weave", mode="before")
    @classmethod
    def _get_tracing_links(cls, value) -> str:
        """Attempts to get the current Weave call ID. Returns empty string if unavailable."""
        # Only try to get ID if no value was explicitly provided.
        if not value:
            try:
                import weave  # Import locally to avoid hard dependency?

                call = weave.get_current_call()
                if call and hasattr(call, "ref") and hasattr(call.ref, "id"):
                    return call.ref.id
                else:
                    logger.debug("Weave call or ref.id not found.")
                    return ""
            except ImportError:
                logger.warning("Weave library not installed. Cannot capture Weave trace ID.")
                return ""
            except Exception as e:
                # Catch errors if called outside a Weave context or other issues.
                logger.debug(f"Unable to get weave call ID: {e}")
                return ""
        return value  # Return provided value if exists


def _get_run_info() -> SessionInfo:
    from buttermilk.bm import bm

    return bm.run_info



# --- Core Step Execution ---

class AgentInput(FlowMessage):
    """
    Standard input structure for triggering an agent's primary processing logic (`_process`).

    Carries the necessary data, parameters, context (history), and records needed by the agent.
    The `Orchestrator` or another `Agent` typically constructs this before calling an agent.
    """

    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing input data resolved from mappings or passed directly.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific parameters overriding agent defaults (e.g., different template).",
    )
    context: list[LLMMessage] = Field(
        default_factory=list,
        description="Conversation history (list of SystemMessage, UserMessage, AssistantMessage).",
    )
    records: list[Record] = Field(
        default_factory=list,
        description="List of data records relevant to the current task.",
    )
    prompt: str | None = Field(
        default="",
        description="The primary prompt or instruction for the agent.",
    )

    # Validator to ensure context and records are always lists.
    _ensure_input_list = field_validator("context", "records", mode="before")(make_list_validator())


class StepRequest(AgentInput):
    """
    Represents a request, typically from a Conductor agent, to execute a specific step in the flow.
    Inherits fields from `AgentInput` (inputs, parameters, context, records, prompt).

    Attributes:
        role: The uppercase role name of the agent(s) that should execute this step.
        description: Human-readable description of why this step is being executed.
    """

    role: str = Field(..., description="The ROLE name (uppercase) of the agent to execute.")
    content: str = Field(default="", description="Brief explanation of this step's purpose.", exclude=True)  # Often excluded from LLM context

    @field_validator("role")
    @classmethod
    def _role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the role field is always uppercase for consistency."""
        if v:
            return v.upper()
        return v

    def __str__(self) -> str:
        return self.content
class AgentOutput(FlowMessage):
    """
    Standard output structure returned by an agent after processing an `AgentInput`.

    Contains the agent's results (`outputs`), along with metadata, original inputs,
    messages exchanged with LLMs, errors, and tracing information.
    """

    session_id: str = Field(default=_global_run_id, description="Session identifier.")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    run_info: SessionInfo = Field(default_factory=_get_run_info)
    agent_info: Optional["AgentConfig"] = Field(None, description="Configuration info from the agent")
    call_id: str = Field(
        default_factory=lambda: shortuuid.uuid(),
        description="A unique ID for this specific agent execution/response.",
    )
    parent_id: str | None = Field(default=None)
    # Parameters used to define this agent.
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for this agent execution.",
    )

    # The original inputs dict for this call.
    inputs: Mapping = Field(
        default={},
        description="The input for this call.",
    )

    # The data records that relate to this call.
    records: Sequence[Record] = Field(default=[], description="Data records passed to agent.")

    # Stores the list of messages sent to the LLM during processing.
    messages: list[LLMMessage] = Field(
        default_factory=list,
        description="Messages sent to the LLM for this execution step.",
    )
    prompt: str | None = Field(default=None, description="The prompt provided.")

    # The main result data from the agent, can be a Pydantic model or a dictionary.
    outputs: Any | None = Field(  # Allow str/None based on LLMAgent
        default=None,  # Default to None
        description="The primary output data generated by the agent (parsed JSON, Pydantic model, or raw string).",
    )

    # Tracing information, primarily Weave call ID.
    tracing: TracingDetails = Field(default_factory=TracingDetails, exclude=True)

    # Validator to ensure context and records are always lists.
    _ensure_input_list = field_validator("messages", "records", mode="before")(make_list_validator())

    # Ensure OmegaConf objects (like DictConfig) are converted to standard Python dicts before validation.
    _validate_parameters = field_validator("parameters", "inputs", mode="before")(convert_omegaconf_objects())

    # Validator inherited from FlowMessage handles 'error' field.

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        },
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    )  # type: ignore

    @computed_field
    @property
    def object_type(self) -> str:
        # add an extra field here that helps us rehydrate the object later on
        try:
            if self.outputs:
                return type(self.outputs).__name__
        except Exception as e:
            pass
        return "null"

    @property
    def content(self) -> str:
        """Provides a string representation of the 'outputs' field."""
        # TODO: Improve string representation for different output types (dict, list, model).
        try:
            if self.outputs:
                return str(self.outputs)
        except Exception as e:
            pass
        
        return ""
    
    def __str__(self) -> str:
        return self.content
    
    # def model_dump(self, **kwargs):
    #     """Custom serialization to handle Pydantic models and OmegaConf objects recursively."""
    #     data = super().model_dump(**kwargs)
        
    #     # Handle Pydantic models in 'outputs'
    #     if isinstance(self.outputs, BaseModel):
    #         data["outputs"] = self.outputs.model_dump()
        
    #     # Recursively convert OmegaConf objects in the output data
    #     data = convert_omegaconf_objects_recursive(data)
    #     return data

# --- Manager / Conductor / UI Interaction Messages ---
# These messages facilitate communication between the orchestrator, conductor agents,
# and the user interface agent (Manager).


class ManagerMessage(FlowMessage):
    """
    A generic message intended for the MANAGER (UI/User).
    Can be used for status updates, displaying results, or asking simple questions.
    """

    content: str | None = Field(
        default=None,
        description="Human-readable text content of the message.",
    )

    def __str__(self) -> str:
        return self.content
class ConductorRequest(AgentInput):
    """
    A request sent *to* a CONDUCTOR agent, asking for the next step or decision.
    Inherits fields from `AgentInput` (inputs, parameters, context, records, prompt).
    """
    participants: Mapping[str,str] = Field(..., description="Chat participant roles and descriptions of their purposes")
    
    # No additional fields specific to ConductorRequest currently defined.
    pass


class ConductorResponse(ManagerMessage, AgentOutput):
    """
    A response sent *from* a CONDUCTOR agent.
    Inherits fields from `ManagerMessage` (content?) and `AgentOutput` (outputs, metadata, etc.).
    The `outputs` field often contains a `StepRequest` object or special instructions (e.g., question for user).
    """

    # No additional fields specific to ConductorResponse currently defined.
    pass


class ManagerRequest(ManagerMessage, StepRequest):
    """
    A request sent *to* the MANAGER (UI/User), typically asking for confirmation,
    feedback, or selection.
    Inherits content/outputs from `ManagerMessage` and role/prompt/description from `StepRequest`.
    """

    # Override description from StepRequest for Manager context.
    description: str = Field(default="Requesting input or confirmation from the user.")
    # Options for multiple-choice questions presented to the user.
    options: bool | list[str] | None = Field(
        default=None,
        description="If list[str], presents options to the user. If bool, implies simple yes/no.",
    )
    # TODO: 'confirm' and 'halt' seem like response fields, not request fields. Verify placement.
    # confirm: bool | None = Field(
    #     default=None,
    #     description="Response from user: confirm y/n", # Description indicates response field.
    # )
    # halt: bool = Field(default=False, description="Whether to stop the flow")


class ManagerResponse(FlowMessage):
    """
    A response sent *from* the MANAGER (UI/User) back to the orchestrator/system.
    Communicates user decisions like confirmation, feedback, or selections.
    """
    confirm: bool = Field(default=False, description="Indicates user confirmation (True) or rejection (False).")
    halt: bool = Field(default=False, description="If True, signals the user wants to stop the entire flow.")
    interrupt: bool = Field(default=False, description="If True, signals the user wants to pause for conductor review of feedback.")
    prompt: Optional[str] = Field(default=None, description="Free-text feedback or instructions provided by the user.")
    selection: Optional[str] = Field(default=None, description="The option selected by the user (e.g., a specific variant ID).")
    params: Optional[Mapping[str, Any]] = Field(default=None)


# --- Tool / Function Call Messages ---


class ToolOutput(FunctionExecutionResult):
    """
    Represents the result of a tool (function) execution by an agent.
    Inherits fields from Autogen's `FunctionExecutionResult` (like `call_id`, `function_name`, `content`).
    Adds Buttermilk-specific context.
    """

    # TODO: 'results' seems duplicative of 'content' from base class. Consolidate or clarify.
    results: Any = Field(default=None)  # Default to None
    # TODO: 'messages' seems out of place for tool output. Clarify purpose.
    messages: list[LLMMessage] = Field(default_factory=list)
    # TODO: 'args' seems related to input, not output. Clarify purpose.
    args: list[str] | list[dict[str, Any]] | dict[str, Any] = Field(default_factory=dict)

    # Content inherited from FunctionExecutionResult holds the tool's return value (often stringified).
    content: str = Field(..., description="String representation of the tool's execution result.")
    # Call ID inherited? If not, uncomment. Ensure consistency.
    call_id: str = Field(default="unknown", description="ID of the corresponding tool call request.")
    # TODO: 'is_error' - does FunctionExecutionResult have this? If not, uncomment.
    # is_error: bool | None = Field(default=False)


# --- Status & Coordination Messages ---
# Primarily used internally by agents/orchestrators, especially within Autogen context.


class TaskProcessingStarted(BaseModel):
    """Signal indicating an agent has started processing a task."""

    agent_id: str = Field(..., description="ID of the agent starting the task.")
    role: str = Field(..., description="Role of the agent starting the task.")
    task_index: int = Field(..., description="Index of the task being started (for multi-task steps).")


class TaskProcessingComplete(TaskProcessingStarted):
    """Signal indicating an agent has completed processing a task."""

    # Inherits agent_id, role, task_index from TaskProcessingStarted.
    more_tasks_remain: bool = Field(..., description="True if the agent has more sequential tasks for the current input.")
    is_error: bool = Field(default=False, description="True if the task ended with an error.")


class ProceedToNextTaskSignal(BaseModel):
    """Control signal from a controller telling an agent to proceed with its next internal task."""

    # TODO: Usage context unclear. Is this used by current orchestrators?
    target_agent_id: str = Field(..., description="ID of the agent that should proceed.")
    model_config = {"extra": "allow"}


class HeartBeat(BaseModel):
    """Simple message potentially used for liveness checks or flow control."""

    # TODO: Usage context unclear. Is this actively used by Agent._check_heartbeat?
    go_next: bool = Field(..., description="Signal indicating if the recipient should proceed.")


# --- Message Union Types ---
# Convenience types for type hinting.

# Out-Of-Band (OOB) messages: Control, status, or manager interactions outside the main agent-to-agent data flow.
OOBMessages = Union[
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    TaskProcessingStarted,
    ConductorResponse,
    ConductorRequest,
    ErrorEvent, StepRequest,ProceedToNextTaskSignal, HeartBeat
]

# Group Chat messages: Standard outputs shared among participating agents.
GroupchatMessageTypes = Union[
    AgentOutput,
    ToolOutput,
    AgentInput,
    Record,
]

# All possible message types used within the system.
AllMessages = Union[GroupchatMessageTypes, OOBMessages, AgentInput, ]
