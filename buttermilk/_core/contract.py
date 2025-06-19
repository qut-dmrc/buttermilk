"""Defines the Pydantic models and constants used for data contracts and communication
within the Buttermilk agent framework.

This module establishes the structure for messages exchanged between various
components of the Buttermilk system, such as agents, orchestrators, and UI
elements. These models ensure consistent data handling and facilitate clear
communication patterns. Key message types include inputs to agents (`AgentInput`),
outputs from agents (`AgentOutput`, `AgentTrace`), control flow messages
(`StepRequest`, `ConductorRequest`), status updates (`TaskProgressUpdate`), and
messages for user interaction (`UIMessage`, `ManagerMessage`).
"""

import datetime
import uuid
from collections.abc import Mapping
from typing import Any, Union

import numpy as np
import shortuuid  # For generating unique IDs

# Import Autogen types used as base or components - conditional import
try:
    from autogen_core.models import FunctionExecutionResult, LLMMessage
except ImportError:
    FunctionExecutionResult = None
    LLMMessage = None
from omegaconf import DictConfig, ListConfig  # For OmegaConf integration
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from buttermilk._core.context import session_id_var
from buttermilk.utils.utils import clean_empty_values
from buttermilk.utils.validators import convert_omegaconf_objects, make_list_validator  # Pydantic validators

from .config import AgentConfig  # Core agent configuration model
from .log import logger  # Centralized logger
from .types import Record  # Core data types like Record

# --- General Communication & Base Messages ---


class FlowEvent(BaseModel):
    """Base model for simple, general-purpose event messages within the flow.

    These events are often used for Out-Of-Band (OOB) communication or simple
    notifications that don't fit more specialized message types.

    Attributes:
        call_id (str): A unique identifier for this specific event instance.
            Defaults to a new UUID4 hex string.
        source (str): Identifier of the message source, e.g., an agent's ID
            or a system component name. Defaults to "server".
        content (str): The main textual content or payload of the event message.

    """

    call_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,  # Use factory for dynamic default
        description="Unique identifier for the event.",
    )
    source: str = Field(
        default="server",
        description="Identifier of the message source (e.g., agent ID, system component).",
    )
    content: str = Field(..., description="The main content or payload of the event message.")
    agent_info: AgentConfig = Field(default=None, description="Configuration info from the agent if applicable.")

    def __str__(self) -> str:
        """Returns the content of the event as its string representation."""
        return self.content

    @computed_field
    @property
    def is_error(self) -> bool:
        """Indicates if this event represents an error. Always False for base FlowEvent."""
        return False

    @field_validator("agent_info", mode="before")
    @classmethod
    def _validate_agent_info(cls, value) -> AgentConfig | None:
        if not isinstance(value, AgentConfig):
            # if agent_config := get_agent_config():
            #     return agent_config
            # Attempt to get the agent info from the current session context.
            # get_agent_info = agent_id_var.get()
            # if get_agent_info:
                # TODO: We maintain a registry of sessions by session_id somewhere. Let's find
                # a way to maintain a registry of agents by agent_id in that registry, and
                # use it to automatically populate the agent_info field here and in other
                # relevant classes.
                # This will help us avoid having to pass agent_info around in the message
                # and make it easier to access agent-specific information.
            return AgentConfig(**value)
        return value


class ErrorEvent(FlowEvent):
    """A specific event type for broadcasting error information within the flow.

    Inherits `call_id`, `source`, and `content` from `FlowEvent`. The `content`
    field typically holds the error message string.

    Attributes:
        is_error (bool): Computed field, always True for `ErrorEvent`.

    """

    # content field inherited from FlowEvent will store the error message.

    def __str__(self) -> str:
        """Prepends "ERROR: " to the content for string representation."""
        return "ERROR: " + self.content

    @computed_field
    @property
    def is_error(self) -> bool:
        """Indicates if this event represents an error. Always True for ErrorEvent."""
        return True


class FlowMessage(BaseModel):
    """Base class for most structured messages exchanged between agents and orchestrators.

    It includes common fields for error tracking and attaching arbitrary metadata.
    This class does not include envelope information (like sender/receiver),
    which is typically handled by the transport layer.

    Attributes:
        error (list[str]): A list of error messages accumulated during the
            processing related to this message. Defaults to an empty list.
        metadata (dict[str, Any]): A dictionary for arbitrary metadata that
            can be associated with the message (e.g., timestamps, tags).
            Defaults to an empty dict.
        is_error (bool): A computed property that is True if the `error` list
            is non-empty.
        model_config (ConfigDict): Pydantic model configuration.
            - `extra`: "forbid" - Disallows extra fields not defined in the model.
            - `arbitrary_types_allowed`: False.
            - `validate_assignment`: True - Validates fields on assignment.
            - `exclude_unset`: True - Excludes fields not explicitly set during serialization.
            - `exclude_none`: True - Excludes fields with None values during serialization.
            - `exclude`: {"is_error"} - Excludes the computed `is_error` property from serialization.

    """

    error: list[str] = Field(
        default_factory=list,
        description="List of error messages accumulated during processing related to this message.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the message (e.g., timestamps, tags).",
    )

    _ensure_error_list: classmethod = field_validator("error", mode="before")(make_list_validator())  # type: ignore

    @computed_field
    @property
    def is_error(self) -> bool:
        """Checks if the `error` list contains any messages.

        Returns:
            bool: True if there are errors, False otherwise.

        """
        return bool(self.error)

    def set_error(self, error_msg: str) -> None:
        """Appends an error message to the `error` list.

        Args:
            error_msg: The error message string to add.

        """
        if not isinstance(self.error, list):  # Should be handled by validator, but defensive
            self.error = []
        if error_msg:  # Only append if the message is not empty
            self.error.append(error_msg)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
        exclude={"is_error"},  # Exclude computed property from model_dump
    )

    def __str__(self) -> str:
        """Returns a JSON string representation of the model."""
        return self.model_dump_json()


class TracingDetails(BaseModel):
    """Holds tracing-related identifiers, particularly for Weave.

    Attributes:
        weave (str): The Weave call ID. This field attempts to auto-populate
            from the current Weave context when an instance is created, but
            only if no explicit value is provided. Excluded from standard serialization.

    """

    # Default factory attempts to get current Weave call ID.
    # This might occur outside a Weave trace context if created manually.
    weave: str = Field(
        default="",  # Default to empty string if not found or not in context
        validate_default=True,  # Ensures validator runs even if default is used
        exclude=True,  # Exclude from standard model serialization (model_dump).
        description="Weave call ID, auto-populated if in a Weave trace context and not explicitly set.",
    )

    @field_validator("weave", mode="before")
    @classmethod
    def _get_tracing_links(cls, value: str | None) -> str:
        """Attempts to get the current Weave call ID if no value is provided.

        Args:
            value: The explicitly provided value for the Weave ID.

        Returns:
            str: The Weave call ID if found or explicitly provided, otherwise an empty string.

        """
        if not value:  # Only attempt to auto-populate if no value was given
            try:
                # Local import to keep Weave as a soft dependency at module level if needed
                import weave as wv_internal
                call = wv_internal.get_current_call()
                # Check if call and its ref attribute and ref.id exist
                if call and hasattr(call, "ref") and call.ref and hasattr(call.ref, "id"):
                    return str(call.ref.id)  # Ensure it's a string
                logger.debug("Weave call or ref.id not found during TracingDetails creation.")
            except ImportError:
                logger.warning("Weave library not installed. Cannot capture Weave trace ID for TracingDetails.")
            except Exception as e:
                logger.warning(f"Unable to get Weave call ID for TracingDetails: {e!s}")
            return ""  # Return empty string if auto-population fails
        return value  # Return the explicitly provided value


def _get_run_info() -> Any:
    """Retrieves the current run/session information from the global Buttermilk instance.

    This function is used as a default factory for fields that need to be populated
    with context-specific run information at the time of model instantiation.

    Returns:
        Any: The current run information object (e.g., `SessionInfo`) from `bm.run_info`.
             The exact type depends on what `bm.run_info` holds. Returns None if
             `bm` or `bm.run_info` is not available.

    """
    try:
        from buttermilk import buttermilk as bm
        return bm.run_info
    except ImportError:
        logger.warning("Buttermilk global instance (bm) not available to get run_info.")
        return None
    except AttributeError:
        logger.warning("bm.run_info not available to get run_info.")
        return None


# --- Core Step Execution ---


class AgentInput(FlowMessage):
    """Standard input structure for triggering an agent's primary processing logic (`_process`).

    This message carries all necessary data for an agent to perform its task,
    including resolved input data, specific parameters for the current task,
    conversation history (context), and relevant data records. It's typically
    constructed by an `Orchestrator` or another `Agent` before invoking a target agent.

    Attributes:
        parent_call_id (str | None): The ID of the parent Weave call, if this input
            is part of a nested trace. Defaults to None.
        inputs (dict[str, Any]): A dictionary containing input data that has been
            resolved from mappings (defined in `AgentConfig.inputs`) or passed
            directly. This is where an agent finds its primary data arguments.
        parameters (dict[str, Any]): Task-specific parameters that can override
            the agent's default configurations (e.g., a different LLM model,
            a specific prompt template name).
        context (list[LLMMessage]): A list of messages representing the conversation
            history (e.g., `SystemMessage`, `UserMessage`, `AssistantMessage` from Autogen).
            This provides conversational context, especially for LLM-based agents.
        records (list[Record]): A list of `Record` objects relevant to the current
            task. These are typically the primary data items the agent will process.

    """

    parent_call_id: str | None = Field(
        default=None,
        description="ID of the parent Weave call in a trace, if applicable.",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing primary input data for the agent, often resolved from mappings.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific parameters overriding agent defaults (e.g., different LLM model, template name).",
    )
    context: list[LLMMessage] = Field(
        default_factory=list,
        description="Conversation history (list of Autogen `LLMMessage` objects like SystemMessage, UserMessage, AssistantMessage).",
    )
    records: list[Record] = Field(
        default_factory=list,
        description="List of `Record` objects relevant to the current task for the agent to process.",
    )

    _ensure_input_list: classmethod = field_validator("context", "records", mode="before")(make_list_validator())  # type: ignore

    def __str__(self) -> str:
        """Provides a concise string representation of the AgentInput instance."""
        parts = []
        if prompt := self.inputs.get("prompt"):  # Check specifically for a 'prompt' key
            prompt_summary = str(prompt)[:50] + "..." if len(str(prompt)) > 50 else str(prompt)
            parts.append(f"Prompt: '{prompt_summary}'")
        if self.inputs:
            parts.append(f"{len(self.inputs)} inputs keys")
        if self.parameters:
            parts.append(f"{len(self.parameters)} parameters")
        if self.records:
            record_ids = [rec.record_id for rec in self.records if hasattr(rec, "record_id")]
            parts.append(f"Records: [{', '.join(record_ids)}]")

        if not parts:
            return "AgentInput (empty)"
        return f"AgentInput({'; '.join(parts)})"


class StepRequest(AgentInput):
    """Represents a request, typically from a Conductor agent, to execute a specific step in a flow.

    This message instructs an agent (identified by `role`) to perform a task.
    It inherits all fields from `AgentInput` (like `inputs`, `parameters`, `context`,
    `records`) to provide the necessary data for the step.

    Attributes:
        role (str): The role name (typically uppercase) of the agent or group of
            agents that should execute this step. This is a mandatory field.
        content (str): A brief explanation or description of this step's purpose.
            This field is often excluded from being passed into an LLM's context
            to avoid clutter, serving more as a human-readable note or for logging.
            Defaults to an empty string.

    """

    role: str = Field(
        ...,  # Mandatory field
        description="The ROLE name (typically uppercase) of the agent designated to execute this step.",
    )
    content: str = Field(
        default="",
        description="Brief human-readable explanation of this step's purpose. Often excluded from LLM context.",
        exclude=True,  # Exclude from model_dump by default if not needed for processing
    )

    @field_validator("role")
    @classmethod
    def _role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the `role` field is always in uppercase for consistency."""
        if v:
            return v.upper()
        # Consider raising ValueError if role is empty, as it's a required field.
        # However, Pydantic handles required fields by default.
        return v

    def __str__(self) -> str:
        """Returns a string representation in the format "ROLE: content"."""
        return f"{self.role}: {self.content}"


class AgentOutput(BaseModel):
    """Standard response format from an agent's `_process` method.

    This model encapsulates the results of an agent's execution, including a timestamp,
    a unique call ID for this specific execution, the ID of the agent that produced
    the output, any metadata, and the primary output data itself.

    Attributes:
        timestamp (datetime.datetime): Timestamp of when the output was generated.
            Defaults to the current UTC time.
        call_id (str): A unique identifier for this specific agent execution/response.
            Defaults to a new UUID4 hex string. It attempts to use the current
            Weave call ID if available and no ID is provided.
        agent_id (str): The unique identifier of the agent instance that produced this output.
        metadata (dict[str, Any]): A dictionary for metadata about the execution
            (e.g., timing information, model used, token counts).
        outputs (Any | None): The primary output data generated by the agent. This can
            be of any type (e.g., a string, a list, a dictionary, another Pydantic model).
        content (str): A computed property providing a string representation of `outputs`.
        model_config (ConfigDict): Pydantic model configuration.

    """

    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="Timestamp of when the output was generated (UTC).",
    )
    call_id: str = Field(
        default_factory=lambda: str(shortuuid.uuid()),  # Use shortuuid for brevity
        description="A unique ID for this specific agent execution/response. Attempts to use Weave call ID if available.",
    )
    tracing_link: str | None = Field(default=None, description="Link to the tracing information, if available.", exclude=True)
    agent_id: str = Field(
        ...,  # Mandatory field
        description="Unique identifier of the agent instance that produced this output.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the execution (e.g., timing, model info, token counts).",
    )
    outputs: Any | None = Field(
        default=None,
        description="The primary output data generated by the agent (can be any type).",
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,  # Be strict by default
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            np.bool_: bool,  # Handle numpy bools
            datetime.datetime: lambda v: v.isoformat(),  # Standard ISO format for datetimes
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),  # OmegaConf compatibility
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        },
        validate_assignment=True,
        exclude_unset=True,  # Exclude fields not explicitly set
        exclude_none=True,  # Exclude fields with None values
    )

    @property
    def content(self) -> str:
        """Provides a string representation of the `outputs` field.

        Returns:
            str: String representation of `outputs`, or an empty string if
                 `outputs` is None or string conversion fails.

        """
        try:
            if self.outputs is not None:
                return str(self.outputs)
        except Exception as e:
            logger.warning(f"Could not convert AgentOutput.outputs to string: {e!s}")
        return ""

    def __str__(self) -> str:
        """Returns the `content` (string representation of `outputs`) of the agent output."""
        return self.content


class AgentTrace(FlowMessage, AgentOutput):
    """Comprehensive record of a single agent execution, for tracing and logging.

    This model captures the full context of an agent's operation. It combines:
    - Information from `FlowMessage` (like `error`, `metadata`).
    - Information from `AgentOutput` (like `timestamp`, `call_id`, `agent_id`, `outputs`).
    - The specific `AgentInput` that triggered the execution.
    - Configuration of the agent (`agent_info`).
    - Session and run information.
    - Messages exchanged with an LLM during processing (if any).
    - Tracing linkage (parent call ID, link to trace).

    Attributes:
        run_info (Any): Information about the current run or session, typically
            obtained from the global Buttermilk instance (`bm.run_info`).
        agent_info (AgentConfig): The configuration of the agent that performed
            the execution. This is a mandatory field.
        session_id (str): Unique identifier for the client session or overall flow execution.
        parent_call_id (str | None): ID of the parent Weave call, for tracing nested operations.
        tracing_link (str | None): A direct URL or link to the trace information in a
            tracing system (e.g., Langfuse, Weave UI).
        inputs (AgentInput): The exact `AgentInput` object that was processed by the agent.
        messages (list[LLMMessage]): A list of messages (e.g., prompts, responses)
            exchanged with an LLM during this execution step.
        object_type (str): A computed field that attempts to determine the Python
            type name of the `outputs` field, useful for later rehydration or analysis.

    """

    run_info: Any = Field(
        default_factory=_get_run_info,
        description="Information about the current run/session, from `bm.run_info`.",
    )
    agent_info: AgentConfig = Field(
        ...,  # Mandatory field
        description="Configuration of the agent that performed this execution.",
    )
    session_id: str = Field(
        default_factory=session_id_var.get,  # Mandatory field
        description="Unique identifier for the client session or overall flow execution.",
    )

    # Tracing information and call metadata.
    parent_call_id: str | None = Field(
        default=None,
        description="ID of the parent Weave call for tracing nested operations.",
    )
    tracing_link: str | None = Field(
        default=None,
        description="Direct URL/link to the trace in a tracing system (e.g., Langfuse, Weave UI).",
    )
    inputs: AgentInput = Field(
        ...,  # Mandatory field
        description="The exact AgentInput object that was processed by the agent for this trace.",
    )
    messages: list[LLMMessage] = Field(
        default_factory=list,
        description="List of messages (prompts, responses) exchanged with an LLM during this execution step.",
    )

    _ensure_messages_list: classmethod = field_validator("messages", mode="before")(make_list_validator())  # type: ignore
    _validate_input_params: classmethod = field_validator("inputs", mode="before")(convert_omegaconf_objects)  # type: ignore

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
        # Exclude computed fields from FlowMessage and AgentOutput if they are also here
        exclude={"is_error", "content", "object_type"},
    )

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Override model_dump to exclude empty collections."""
        raw_dump = super().model_dump(*args, **kwargs)
        # Apply the cleaning function to the result
        return clean_empty_values(raw_dump)

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Override model_dump to exclude empty collections."""
        raw_dump = super().model_dump(*args, **kwargs)
        # Apply the cleaning function to the result
        return clean_empty_values(raw_dump)

    @computed_field
    @property
    def object_type(self) -> str:
        """Attempts to determine the Python type name of the `outputs` field.

        Useful for later rehydration or analysis of the output data.

        Returns:
            str: The type name of `self.outputs` if available, otherwise "null".

        """
        try:
            if self.outputs is not None:
                return type(self.outputs).__name__
        except Exception:  # Broad catch as type determination can be tricky
            pass  # Fall through to return "null"
        return "null"  # Default if outputs is None or type cannot be determined

    def __str__(self) -> str:
        """Returns the `content` (string representation of `outputs`) of the agent trace."""
        # Inherits content property from AgentOutput
        return super().content


# --- Manager / Conductor / UI Interaction Messages ---

class ConductorRequest(AgentInput):
    """A request message sent *to* a CONDUCTOR agent.

    This message typically asks the Conductor agent to make a decision about the
    next step in a workflow, select an agent, or manage the overall flow.
    It inherits all fields from `AgentInput` to provide the Conductor with the
    necessary context (current inputs, parameters, conversation history, records).

    Attributes:
        participants (Mapping[str, str]): A mapping where keys are participant
            roles (e.g., "SUMMARIZER", "REVIEWER") and values are descriptions
            of their purposes or capabilities. This helps the Conductor understand
            the available agents and their functions. This is a mandatory field.
        participant_tools (Mapping[str, list[dict[str, Any]]]): Optional mapping of
            participant roles to their tool definitions. Each tool definition includes
            name, description, and schema information.

    """

    participants: Mapping[str, str] = Field(
        ...,  # Mandatory field
        description="Mapping of chat participant roles to their purpose descriptions (e.g., {'SUMMARIZER': 'Summarizes text'}).",
    )
    participant_tools: Mapping[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Mapping of participant roles to their tool definitions (e.g., {'RESEARCHER': [{'name': 'search', 'description': '...'}]}).",
    )


class UIMessage(FlowMessage):
    """A message sent *to* the MANAGER (representing the UI/User).

    This message is typically used to ask the user for confirmation, feedback,
    or to make a selection from a list of options. It inherits from `FlowMessage`
    for error and metadata handling.

    Attributes:
        content (str): The question or information to present to the user. This
            is a mandatory field.
        options (bool | list[str] | None): Defines the type of response expected:
            - If `list[str]`: Presents these strings as multiple-choice options to the user.
            - If `bool` (typically `True`): Implies a simple Yes/No or confirmation request.
            - If `None` (default): No specific options are presented; user might provide free text.

    """

    content: str = Field(..., description="The question or information to present to the user.")
    options: bool | list[str] | None = Field(
        default=None,
        description="Options for user response: list of strings for choices, bool for Yes/No, None for free text.",
    )
    thought: str | None = Field(
        default=None,
        description="The reasoning text for the completion if available. Used for reasoning model and additional text content besides function calls.",
    )
    thought: str | None = Field(
        default=None,
        description="The reasoning text for the completion if available. Used for reasoning model and additional text content besides function calls.",
    )


class ManagerMessage(FlowMessage):
    """A response message sent *from* the MANAGER (UI/User) back to the system.

    This message communicates user decisions, feedback, or instructions resulting
    from a `UIMessage` or other user interaction. It inherits from `FlowMessage`.

    Attributes:
        confirm (bool | None): Indicates user confirmation (`True`) or rejection (`False`).
            `None` if not a confirmation-style interaction. Defaults to `False`.
        halt (bool | None): If `True`, signals that the user wants to stop the
            entire workflow. Defaults to `False`.
        interrupt (bool | None): If `True`, signals that the user wants to pause
            the current operation, often for a conductor or human to review feedback
            before proceeding. Defaults to `False`.
        human_in_loop (bool | None): If `True`, indicates that the user is actively
            involved or wishes to remain involved in the process. `None` if not specified.
        content (str | None): Free-text feedback, instructions, or responses
            provided by the user.
        selection (str | None): The option selected by the user if they were
            presented with choices (e.g., a specific variant ID, one of the
            strings from `UIMessage.options`).
        params (Mapping[str, Any] | None): Additional parameters or data provided
            by the user, potentially for configuring the next step.

    """

    confirm: bool | None = Field(
        default=False, description="User confirmation (True) or rejection (False).",
    )
    halt: bool | None = Field(
        default=False, description="If True, user signals to stop the entire flow.",
    )
    interrupt: bool | None = Field(
        default=False, description="If True, user signals a pause for review of feedback.",
    )
    human_in_loop: bool | None = Field(
        default=None, description="If True, indicates active user involvement.",
    )
    content: str | None = Field(
        default=None, description="Free-text feedback or instructions from the user.",
    )
    selection: str | None = Field(
        default=None, description="The option selected by the user from choices provided.",
    )
    params: Mapping[str, Any] | None = Field(
        default=None, description="Additional parameters or data provided by the user.",
    )


# --- Task Progress Message ---
class FlowProgressUpdate(FlowMessage):
    """A message used to provide real-time information about the progress of a task or step.

    This is primarily intended for updating a user interface or logging system
    about the current state of a potentially long-running operation. It inherits
    from `FlowMessage`.

    Attributes:
        source (str): The ID of the agent or component sending the progress update.
            This is a mandatory field.
        step_name (str): The name or identifier of the current step being processed.
            This is a mandatory field.
        status (str): The current status of the step (e.g., "STARTED",
            "IN_PROGRESS", "COMPLETED", "ERROR"). This is a mandatory field.
        message (str): A human-readable message describing the current progress
            or status. Defaults to an empty string.
        timestamp (datetime.datetime): The time at which the progress update was
            generated. Defaults to the current UTC time.
        waiting_on (dict[str, Any]): A dictionary indicating if this step is
            waiting on other agents or tasks, and potentially details about them.
            Keys might be agent IDs, values could be status or task names.
            Defaults to an empty dict.

    """

    source: str = Field(..., description="ID of the agent or component sending the progress update.")
    step_name: str = Field(..., description="Name or identifier of the current step being processed.")
    status: str = Field(..., description="Current status (e.g., 'STARTED', 'COMPLETED', 'ERROR').")
    message: str = Field(default="", description="Human-readable message describing current progress.")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="Timestamp of when the progress update was generated (UTC).",
    )
    waiting_on: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary indicating other agents/tasks this step is waiting on (e.g., {'agent_id': 'task_name'}).",
    )

    def __str__(self) -> str:
        """Returns a formatted string representation of the task progress."""
        status_msg = f"[{self.status.upper()}] {self.message}"
        if self.waiting_on:
            waiting_on_str = ", ".join([f"{k}: {v}" for k, v in self.waiting_on.items()])
            status_msg += f" - Waiting on: {waiting_on_str}"
        return status_msg


# --- Tool / Function Call Messages ---


if FunctionExecutionResult is not None:
    class ToolOutput(FunctionExecutionResult):
        """Represents the result of a tool (function) execution performed by an agent.

        This class inherits from Autogen's `FunctionExecutionResult`, which typically
        includes fields like `call_id` (for the function call), `function_name`, and
        `content` (the stringified result of the function). It adds Buttermilk-specific
        context or alternative ways to structure results.

    Attributes:
        results (Any): The raw or structured result from the tool execution. This
            can be of any type and might be preferred over `content` if the result
            is not naturally a string. Defaults to None.
        messages (list[LLMMessage]): A list of LLM messages that might have been
            generated or consumed during the tool's execution, if the tool itself
            interacts with an LLM. Defaults to an empty list.
        args (list[str] | list[dict[str, Any]] | dict[str, Any]): The arguments
            that were passed to the tool when it was called. Defaults to an empty dict.
        content (str): Inherited from `FunctionExecutionResult`. Should be the
            string representation of the tool's execution result. This is a
            mandatory field.
        call_id (str): Inherited (or overridden) field for the ID of the
            corresponding tool call request. Defaults to "unknown".

    """

    # `function_name` and `tool_call_id` are inherited from FunctionExecutionResult.
    # `content` (str) is also inherited and should hold the stringified tool output.

    results: Any = Field(
        default=None,
        description="Raw or structured result from the tool execution. Can be any type.",
    )
    messages: list[LLMMessage] = Field(
        default_factory=list,
        description="LLM messages generated or consumed during tool execution, if applicable.",
    )
    args: list[str] | list[dict[str, Any]] | dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool when it was called.",
    )
    content: str = Field(  # Overriding to ensure it's always present in schema if needed
        ...,
        description="String representation of the tool's execution result.",
    )
    call_id: str = Field(  # Overriding to ensure consistent description and default
        default="unknown",
        description="ID of the corresponding tool call request that this output answers.",
    )
        # is_error might be a useful addition if not in base class, e.g.:
        # is_error: bool = Field(default=False, description="True if the tool execution resulted in an error.")
else:
    # Fallback when autogen is not available
    class ToolOutput(BaseModel):
        """Fallback ToolOutput when autogen is not available."""
        call_id: str = ""
        function_name: str = ""
        content: str = ""
        results: Any = None


# --- Status & Coordination Messages ---

class TaskProcessingStarted(BaseModel):
    """A signal message indicating that an agent has started processing a task.

    This is typically used for monitoring and UI updates.

    Attributes:
        agent_id (str): The unique identifier of the agent that has started the task.
        role (str): The role of the agent starting the task.
        task_index (int): An optional index for the task, particularly if an agent
            is performing multiple sequential tasks for a single input. Defaults to -1.

    """

    agent_id: str = Field(..., description="ID of the agent that has started the task.")
    role: str = Field(..., description="Role of the agent starting the task.")
    task_index: int = Field(
        default=-1,
        description="Index of the task being started (for multi-task steps by a single agent).",
    )


class TaskProcessingComplete(TaskProcessingStarted):
    """A signal message indicating that an agent has completed processing a task.

    Inherits `agent_id`, `role`, and `task_index` from `TaskProcessingStarted`.

    Attributes:
        more_tasks_remain (bool): If `True`, indicates that the agent has more
            sequential tasks to perform for the current input/context. Defaults to `False`.
        is_error (bool): If `True`, indicates that the task ended with an error.
            Defaults to `False`.

    """

    more_tasks_remain: bool = Field(
        default=False,
        description="True if the agent has more sequential tasks for the current input.",
    )
    is_error: bool = Field(
        default=False, description="True if the task completed with an error.",
    )


class ProceedToNextTaskSignal(BaseModel):
    """A control signal, typically from a controller or orchestrator, instructing
    an agent to proceed with its next internal task or step.

    The exact usage context for this signal might depend on specific orchestrator
    implementations.

    Attributes:
        target_agent_id (str): The unique identifier of the agent that should
            proceed to its next task.
        model_config (dict): Pydantic model configuration allowing extra fields.

    """

    # TODO: Clarify usage context if this is actively used.
    target_agent_id: str = Field(..., description="ID of the agent that should proceed to its next task.")
    model_config = {"extra": "allow"}  # Allows extra fields if needed


class HeartBeat(BaseModel):
    """A simple message potentially used for liveness checks or basic flow control.

    The exact usage context might depend on specific agent or orchestrator
    implementations (e.g., related to `Agent._check_heartbeat`).

    Attributes:
        go_next (bool): A boolean signal, possibly indicating if the recipient
            should proceed with an action or if a condition is met.

    """

    # TODO: Clarify usage context if this is actively used by Agent._check_heartbeat.
    go_next: bool = Field(..., description="Signal indicating if the recipient should proceed or a condition is met.")


# --- Message Union Types ---

OOBMessages = Union[
    UIMessage,
    TaskProcessingComplete,
    TaskProcessingStarted,
    FlowProgressUpdate,
    ConductorRequest,
    ErrorEvent,
    StepRequest,
    ProceedToNextTaskSignal,
    HeartBeat,
]
"""A type alias for messages considered Out-Of-Band (OOB).

These include UI interactions, task status updates, control signals, and errors
that occur outside the primary data flow between processing agents.
"""

GroupchatMessageTypes = Union[
    AgentTrace,
    ToolOutput,
    AgentOutput,  # AgentOutput might be too generic here if AgentTrace is preferred for group chat
    ManagerMessage,  # If manager responses are broadcasted
    # AgentInput, # AgentInput is usually direct, not a "group chat" message
    Record,  # If raw records are shared
]
"""A type alias for messages typically shared among participating agents in a group chat or multi-agent setup.

This usually includes traces of agent actions, tool outputs, and potentially user/manager messages
if they are broadcast to the group.
"""

AllMessages = Union[GroupchatMessageTypes, OOBMessages, AgentInput]
"""A comprehensive type alias representing all possible message types used within the Buttermilk system.

This is useful for type hinting in components that can handle or route any kind of message.
"""
