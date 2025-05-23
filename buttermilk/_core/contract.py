"""Defines the Pydantic models and constants used for data contracts and communication
within the Buttermilk agent framework.

This includes message types for agent inputs, outputs, control flow, status updates,
and interactions with manager/conductor roles.
"""

import datetime
import uuid
from collections.abc import Mapping
from typing import Any, Union

import numpy as np
import shortuuid  # For generating unique IDs
import weave

# Import Autogen types used as base or components
from autogen_core.models import FunctionExecutionResult, LLMMessage
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from buttermilk.utils.validators import (  # Pydantic validators
    convert_omegaconf_objects,
    make_list_validator,
)

from .config import AgentConfig
from .log import logger
from .types import Record  # Core data types

# --- General Communication & Base Messages ---


class FlowEvent(BaseModel):
    """Base model for simple event messages within the flow (potentially OOB)."""

    call_id: str = Field(default=uuid.uuid4().hex, description="Unique identifier for the event.")
    source: str = Field(default="server", description="Identifier of the message source (e.g., agent ID).")
    content: str = Field(..., description="The main content of the event message.")

    def __str__(self) -> str:
        return self.content

    @computed_field
    @property
    def is_error(self) -> bool:
        return False


class ErrorEvent(FlowEvent):
    """Specific event type for broadcasting errors."""

    # Inherits source and content, content typically holds the error message string.

    def __str__(self) -> str:
        return "ERROR: " + self.content

    @computed_field
    @property
    def is_error(self) -> bool:
        return True


class FlowMessage(BaseModel):
    """Base class for most messages exchanged between agents and orchestrators.
    Includes common fields like error tracking and metadata. Does not include
    envelope information, which will be added by the transport.
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
        """Append an error message to the error list."""
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

    # Default factory attempts to get current Weave call ID when an AgentTrace is created.
    # This might happen outside a Weave trace context if AgentTrace is created manually.
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
                logger.info("Weave call or ref.id not found.")
                return ""
            except ImportError:
                logger.warning("Weave library not installed. Cannot capture Weave trace ID.")
                return ""
            except Exception as e:
                # Catch errors if called outside a Weave context or other issues.
                logger.warning(f"Unable to get weave call ID: {e}")
                return ""
        return value  # Return provided value if exists


def _get_run_info():
    """Get the current session info from the global bm instance.

    Returns:
        A SessionInfo object with the current run information.

    """
    from buttermilk import buttermilk as bm

    return bm.run_info


# --- Core Step Execution ---


class AgentInput(FlowMessage):
    """Standard input structure for triggering an agent's primary processing logic (`_process`).

    Carries the necessary data, parameters, context (history), and records needed by the agent.
    The `Orchestrator` or another `Agent` typically constructs this before calling an agent.
    """

    parent_call_id: str | None = Field(default=None, description="ID of the parent call, if applicable.")

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

    # Validator to ensure context and records are always lists.
    _ensure_input_list = field_validator("context", "records", mode="before")(make_list_validator())

    def __str__(self) -> str:
        # Provide an abridged string representation of the inputs.
        parts = []
        if prompt := self.inputs.get("prompt"):
            prompt_summary = prompt[:50] + "..." if len(prompt) > 50 else prompt
            parts.append(f"Prompt: '{prompt_summary}'")
        if self.inputs:
            parts.append(f"{len(self.inputs)} inputs")
        if self.parameters:
            parts.append(f"{len(self.parameters)} parameters")
        if self.records:
            parts.append(", ".join([rec.record_id for rec in self.records]))

        if not parts:
            return "AgentInput (empty)"
        return f"AgentInput({', '.join(parts)})"


class StepRequest(AgentInput):
    """Represents a request, typically from a Conductor agent, to execute a specific step in the flow.

    Inherits fields from `AgentInput` (inputs, parameters, context, records, prompt).

    Attributes:
        role: The uppercase role name of the agent(s) that should execute this step.
        description: Human-readable description of why this step is being executed.

    """

    role: str = Field(..., description="The ROLE name (uppercase) of the agent to execute.")
    content: str = Field(
        default="", description="Brief explanation of this step's purpose.", exclude=True,
    )  # Often excluded from LLM context

    @field_validator("role")
    @classmethod
    def _role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the role field is always uppercase for consistency."""
        if v:
            return v.upper()
        return v

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class AgentOutput(BaseModel):
    """Standard response format for all Agent _process methods after processing an `AgentInput`.

    Contains the metadata and outputs from agent processing. Used by the Agent.__call__ method
    to create and manage the AgentTrace that will be logged and returned to the orchestrator.

    Attributes:
        metadata: Dictionary of metadata about the execution (e.g., timing, model info, etc.)
        outputs: The primary output data generated by the agent

    """

    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
    call_id: str = Field(default=uuid.uuid4().hex,
        description="A unique ID for this specific agent execution/response.",
    )
    agent_id: str = Field(..., description="Unique identifier for the agent instance.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata about the execution")
    outputs: Any | None = Field(default=None, description="The primary output data generated by the agent")

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

    @property
    def content(self) -> str:
        """Provides a string representation of the 'outputs' field."""
        # TODO: Improve string representation for different output types (dict, list, model).
        try:
            if self.outputs:
                return str(self.outputs)
        except Exception:
            pass

        return ""

    @field_validator("call_id", mode="before")
    @classmethod
    def _ensure_call_id(cls, v: str) -> str:
        """Look up call id if available."""
        if not v:
            # --- Weave Tracing ---
            # Get the current Weave call context if available.
            call = weave.get_current_call()
            if call and call.id:
                v = call.id
        if not v:
            v = str(shortuuid.uuid())
        return v

    def __str__(self) -> str:
        return self.content


class AgentTrace(FlowMessage, AgentOutput):
    """Full information about a single agent execution for tracing and logging.

    Combines the AgentInput parameters for the specific task execution, the run
    or session information, along with metadata, messages exchanged with LLMs,
    errors, and tracing information, as well as the main AgentOutput results.
    """

    run_info: Any = Field(default_factory=_get_run_info)
    agent_info: AgentConfig = Field(..., description="Configuration info from the agent (required)")
    session_id: str = Field(..., description="Unique identifier for client session")

    # Tracing information and call metadata.
    parent_call_id: str | None = Field(default=None)
    tracing_link: str | None = Field(default=None, description="Link to the tracing information, if available.")
    # The original inputs dict for this call.
    inputs: AgentInput = Field(
        ...,
        description="The input for this call.",
    )

    # Stores the list of messages sent to the LLM during processing.
    messages: list[LLMMessage] = Field(
        default_factory=list,
        description="Messages sent to the LLM for this execution step.",
    )

    # The main result data from the agent is inherited from AgentOutput

    # Validator to ensure context and records are always lists.
    _ensure_input_list = field_validator("messages", mode="before")(make_list_validator())

    # Ensure OmegaConf objects (like DictConfig) are converted to standard Python dicts before validation.
    _validate_parameters = field_validator("inputs", mode="before")(convert_omegaconf_objects())

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
        except Exception:
            pass
        return "null"

    def __str__(self) -> str:
        return self.content

    def model_dump(self, **kwargs):
        """Override model_dump to exclude empty collections."""
        exclude_keys = set(kwargs.get("exclude", set()))

        # Find keys with empty values
        for key, value in self.__dict__.items():
            if isinstance(value, (list, dict, set)) and len(value) == 0:
                exclude_keys.add(key)

        # Update exclude with our extended set
        kwargs["exclude"] = exclude_keys
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        """Override model_dump_json to exclude empty collections."""
        return super().model_dump_json(**kwargs, exclude_none=True, exclude_unset=True)

    # def model_dump(self, **kwargs):
    #     """Custom serialization to handle Pydantic models and OmegaConf objects recursively."""
    #     data = super().model_dump(**kwargs)

    #     # Handle Pydantic models in 'outputs'
    #     if isinstance(self.outputs, BaseModel):
    #         data["outputs"] = self.outputs.model_dump()

    #     return data


# --- Manager / Conductor / UI Interaction Messages ---
# These messages facilitate communication between the orchestrator, conductor agents,
# and the user interface agent (Manager).

class ConductorRequest(AgentInput):
    """A request sent *to* a CONDUCTOR agent, asking for the next step or decision.

    Inherits fields from `AgentInput` (inputs, parameters, context, records, prompt).
    """

    participants: Mapping[str, str] = Field(
        ..., description="Chat participant roles and descriptions of their purposes",
    )

    # No additional fields specific to ConductorRequest currently defined.


class UIMessage(FlowMessage):
    """A request sent *to* the MANAGER (UI/User), typically asking for confirmation,
    feedback, or selection.
    Inherits content/outputs from `ManagerMessage` and role/prompt/description from `StepRequest`.
    """

    content: str = Field(...)
    # Options for multiple-choice questions presented to the user.
    options: bool | list[str] | None = Field(
        default=None,
        description="If list[str], presents options to the user. If bool, implies simple yes/no.",
    )


class ManagerMessage(FlowMessage):
    """A response sent *from* the MANAGER (UI/User) back to the orchestrator/system.
    Communicates user decisions like confirmation, feedback, or selections.
    """

    confirm: bool | None = Field(default=False, description="Indicates user confirmation (True) or rejection (False).")
    halt: bool | None = Field(default=False, description="If True, signals the user wants to stop the entire flow.")
    interrupt: bool | None = Field(
        default=False, description="If True, signals the user wants to pause for conductor review of feedback.",
    )
    human_in_loop: bool | None = Field(default=None, description="If True, indicates the user is actively involved in the process.")
    content: str | None = Field(default=None, description="Free-text feedback or instructions provided by the user.")
    selection: str | None = Field(
        default=None, description="The option selected by the user (e.g., a specific variant ID).",
    )
    params: Mapping[str, Any] | None = Field(default=None)


# --- Task Progress Message ---
class TaskProgressUpdate(FlowMessage):
    """A message sent to provide information about the progress of a task or step in the workflow.

    This is primarily used to update the UI about the workflow's progress.
    """

    source: str = Field(..., description="The agent ID that is sending the progress update")
    step_name: str = Field(..., description="Name of the current step")
    status: str = Field(..., description="Current status (e.g., 'started', 'in_progress', 'completed', 'error')")
    message: str = Field(default="", description="Human-readable message explaining the current progress")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
    waiting_on: dict[str, Any] = Field(default={}, description="Agents with outstanding tasks")

    def __str__(self) -> str:
        status = f"[{self.status.upper()}] {self.message}"
        if self.waiting_on:
            waiting_on_str = ", ".join([f"{k}: {v}" for k, v in self.waiting_on.items()])
            status += f" - Waiting on: {waiting_on_str}"
        return status


# --- Tool / Function Call Messages ---


class ToolOutput(FunctionExecutionResult):
    """Represents the result of a tool (function) execution by an agent.
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
    task_index: int = Field(default=-1, description="Index of the task being started (for multi-task steps).")


class TaskProcessingComplete(TaskProcessingStarted):
    """Signal indicating an agent has completed processing a task."""

    # Inherits agent_id, role, task_index from TaskProcessingStarted.
    more_tasks_remain: bool = Field(
        default=False, description="True if the agent has more sequential tasks for the current input.",
    )
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

# Out-Of-Band (OOB) messages: Control, status, or other events outside the main agent-to-agent data flow.
OOBMessages = Union[UIMessage, TaskProcessingComplete, TaskProcessingStarted, TaskProgressUpdate, ConductorRequest, ErrorEvent, StepRequest, ProceedToNextTaskSignal, HeartBeat]

# Group Chat messages: Standard outputs shared among participating agents.
GroupchatMessageTypes = Union[
    AgentTrace,
    ToolOutput, AgentOutput, ManagerMessage,
    AgentInput,
    Record,
]

# All possible message types used within the system.
AllMessages = Union[GroupchatMessageTypes, OOBMessages, AgentInput]
