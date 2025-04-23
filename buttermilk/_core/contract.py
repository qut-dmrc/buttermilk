"""
Defines the Pydantic models and constants used for data contracts and communication
within the Buttermilk agent framework.

This includes message types for agent inputs, outputs, control flow, status updates,
and interactions with manager/conductor roles.
"""

from asyncio import streams  # TODO: Unused import?
from collections.abc import Mapping
from email.policy import strict  # TODO: Unused import?
from enum import Enum  # TODO: Unused import?
from math import e  # TODO: Unused import?
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

# Import Autogen types used as base or components
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_core.models import FunctionExecutionResult, LLMMessage
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)
import shortuuid  # For generating unique IDs

# Buttermilk core imports
from buttermilk._core.exceptions import ProcessingError
from .config import DataSourceConfig, SaveInfo, Tracing  # Core configuration models
from .types import Record, _global_run_id  # Core data types
from buttermilk.utils.validators import make_list_validator  # Pydantic validators

# TODO: BASE_DIR seems unused. Consider removing.
# BASE_DIR = Path(__file__).absolute().parent

# --- Constants ---

# Standard Agent Roles
CONDUCTOR = "HOST"  # Role name often used for the agent directing the flow (e.g., Sequencer, LLMHostAgent). Configurable?
MANAGER = "MANAGER"  # Role name often used for the user interface or human-in-the-loop agent.
CLOSURE = "COLLECTOR"  # Role name (Collector?) - Usage context unclear from surrounding files. TODO: Verify usage.
CONFIRM = "CONFIRM"  # Special agent/topic name used by AutogenOrchestrator for handling ManagerResponse.

# Special Symbols / States
COMMAND_SYMBOL = "!"  # Prefix potentially used to identify command messages (ignored by Sequencer._listen).
END = "END"  # Special role/signal used by Conductor/Host to indicate the flow should terminate.
WAIT = "WAIT"  # Special role/signal used by Conductor/Host to indicate pausing/waiting.


# --- Flow Protocol ---


class FlowProtocol(BaseModel):
    """
    Defines the overall structure expected for a flow configuration (e.g., loaded from YAML).
    Used primarily for validation or type hinting of the high-level flow definition.
    """

    name: str = Field(..., description="Human-friendly name for the flow.")
    description: str = Field(..., description="Description of the flow's purpose.")
    save: SaveInfo | None = Field(default=None, description="Configuration for saving results (optional).")
    data: list[DataSourceConfig] | None = Field(default_factory=list, description="List of data source configurations.")
    agents: Mapping[str, Any] = Field(default_factory=dict, description="Agent configurations (validated later by Orchestrator).")
    orchestrator: str = Field(..., description="Name or path of the Orchestrator class to use.")
    params: dict = Field(default_factory=dict, description="Flow-level parameters.")


# --- Core Step Execution ---


class StepRequest(BaseModel):
    """
    Represents a request, typically from a Conductor agent, to execute a specific step in the flow.

    Attributes:
        role: The uppercase role name of the agent(s) that should execute this step.
        prompt: The primary input prompt or instruction for the target agent(s).
        description: Human-readable description of why this step is being executed.
    """

    role: str = Field(..., description="The ROLE name (uppercase) of the agent to execute.")
    prompt: str = Field(default="", description="The prompt/instruction text for the agent.")
    description: str = Field(default="", description="Brief explanation of this step's purpose.", exclude=True)  # Often excluded from LLM context
    # TODO: Tool execution fields seem commented out. If tool calls are needed via StepRequest, uncomment and refine.
    # tool: str = Field(default="", description="The tool to invoke, if any.")
    # arguments: dict[str, Any] = Field(description="Arguments to provide to the tool, if any.")

    @field_validator("role")
    @classmethod
    def role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the role field is always uppercase for consistency."""
        if v:
            return v.upper()
        return v


# --- General Communication & Base Messages ---


class FlowEvent(BaseModel):
    """Base model for simple event messages within the flow (potentially OOB)."""

    source: str = Field(..., description="Identifier of the message source (e.g., agent ID).")
    content: str = Field(..., description="The main content of the event message.")


class ErrorEvent(FlowEvent):
    """Specific event type for broadcasting errors."""

    # Inherits source and content, content typically holds the error message string.
    pass


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
        self.error.append(error_msg)


class AgentInput(FlowMessage):
    """
    Standard input structure for triggering an agent's primary processing logic (`_process`).

    Carries the necessary data, parameters, context (history), and records needed by the agent.
    The `Orchestrator` or `Adapter` typically constructs this before calling an agent.
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
    prompt: str = Field(
        default="",
        description="The primary prompt or instruction for the agent.",
    )

    # Validator to ensure context and records are always lists.
    _ensure_input_list = field_validator("context", "records", mode="before")(make_list_validator())


class UserInstructions(FlowMessage):
    """Represents instructions or input originating directly from the user (e.g., via CLI)."""

    # TODO: Seems similar to AgentInput but simpler. Clarify distinction and usage.
    #       CLIUserAgent uses ManagerResponse, not this. When is UserInstructions used?
    records: list[Record] = Field(
        default_factory=list,
        description="List of records associated with the user's instruction.",
    )
    prompt: str = Field(
        default="",
        description="The user's prompt/instruction text.",
    )
    # TODO: 'confirm' and 'stop' fields seem out of place for *instructions*. They belong in a response. Verify usage.
    # confirm: bool = Field(
    #     default=False,
    #     description="Response from user: confirm y/n", # Description indicates response field.
    # )
    # stop: bool = Field(default=False, description="Whether to stop the flow")


class TracingDetails(BaseModel):
    """Holds tracing identifiers, primarily the Weave call ID."""

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


class AgentOutput(FlowMessage):
    """
    Standard output structure returned by an agent after processing an `AgentInput`.

    Contains the agent's results (`outputs`), along with metadata, original inputs,
    messages exchanged with LLMs, errors, and tracing information.
    """

    agent_id: str = Field(..., description="The unique ID of the agent instance that generated this output.")
    run_id: str = Field(default=_global_run_id, description="Global run identifier.")
    call_id: str = Field(
        default_factory=lambda: shortuuid.uuid(),
        description="A unique ID for this specific agent execution/response.",
    )
    # Stores the original inputs dict that led to this output.
    inputs: Optional[dict] = Field(  # Changed from AgentInput to dict based on LLMAgent.make_output
        default=None,
        description="The 'inputs' dictionary provided in the corresponding AgentInput.",
    )
    # Stores the list of messages sent to the LLM during processing.
    messages: list[LLMMessage] = Field(
        default_factory=list,
        description="Messages sent to the LLM for this execution step.",
    )
    # Parameters used for this specific execution.
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for this agent execution.",
    )
    prompt: str = Field(default="", description="The original prompt from AgentInput.")
    # The main result data from the agent, can be a Pydantic model or a dictionary.
    outputs: Union[BaseModel, Dict[str, Any], str, None] = Field(  # Allow str/None based on LLMAgent
        default=None,  # Default to None
        description="The primary output data generated by the agent (parsed JSON, Pydantic model, or raw string).",
    )
    # Tracing information, primarily Weave call ID.
    tracing: TracingDetails = Field(default_factory=TracingDetails)
    # LLM finish reason (e.g., 'stop', 'tool_calls', 'length')
    finish_reason: Optional[str] = Field(default=None, description="Finish reason from the LLM response.")

    # Validator inherited from FlowMessage handles 'error' field.

    def model_dump(self, **kwargs):
        """Custom serialization to handle Pydantic models within 'outputs'."""
        data = super().model_dump(**kwargs)
        # Ensure nested Pydantic models in 'outputs' are also serialized.
        if isinstance(self.outputs, BaseModel):
            data["outputs"] = self.outputs.model_dump()
        # TODO: Consider similar handling for 'inputs' if it could contain Pydantic models.
        return data

    @property
    def contents(self) -> str:
        """Provides a string representation of the 'outputs' field."""
        # TODO: Improve string representation for different output types (dict, list, model).
        return str(self.outputs)


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
    # TODO: 'outputs' field seems redundant if 'content' holds the main info. Clarify purpose.
    outputs: BaseModel | dict[str, Any] | None = Field(  # Allow None
        default=None,
        description="Optional structured data payload associated with the message.",
    )
    agent_id: str = Field(
        # Default likely overridden by sender context (e.g., orchestrator or specific agent).
        default="system",
        description="ID of the entity sending the message to the manager.",
    )
    role: str = Field(default="system", description="Role of the entity sending the message.")  # Added role for clarity


class ConductorRequest(ManagerMessage, AgentInput):
    """
    A request sent *to* a CONDUCTOR agent, asking for the next step or decision.
    Inherits fields from `ManagerMessage` (content, outputs?) and `AgentInput` (inputs, params, context, records, prompt).
    The `inputs` field typically contains context like workflow state, participant list, etc.
    """

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

    confirm: bool = Field(True, description="Indicates user confirmation (True) or rejection (False).")
    halt: bool = Field(False, description="If True, signals the user wants to stop the entire flow.")
    prompt: Optional[str] = Field(None, description="Free-text feedback or instructions provided by the user.")
    selection: Optional[str] = Field(None, description="The option selected by the user (e.g., a specific variant ID).")


# --- Tool / Function Call Messages ---


class ToolOutput(FunctionExecutionResult):
    """
    Represents the result of a tool (function) execution by an agent.
    Inherits fields from Autogen's `FunctionExecutionResult` (like `call_id`, `function_name`, `content`).
    Adds Buttermilk-specific context.
    """
    # TODO: 'role' seems redundant if associated with an agent. Clarify purpose.
    role: str = Field(..., description="The role the tool provides (?)")

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

    # TODO: 'send_to_ui' flag - clarify purpose and implementation.
    send_to_ui: bool = Field(False, description="Flag indicating if this tool output should be directly shown to the UI.")


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
    ErrorEvent,  # Added ErrorEvent
]

# Group Chat messages: Standard outputs shared among participating agents.
GroupchatMessageTypes = Union[
    AgentOutput,
    ToolOutput,
    UserInstructions,  # Why UserInstructions here? Seems misplaced.
    # Add other types that are typically broadcast in the chat?
]

# All possible message types used within the system.
AllMessages = Union[GroupchatMessageTypes, OOBMessages, AgentInput, ProceedToNextTaskSignal, HeartBeat]
