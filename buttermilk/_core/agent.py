from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Annotated, Any, AsyncGenerator, Callable, Self, Sequence, Union, TYPE_CHECKING

from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
import jmespath
import pydantic
from shortuuid import ShortUUID, uuid
import weave
from autogen_core import CancellationToken, MessageContext
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from autogen_core.tools import BaseTool, FunctionTool
from buttermilk._core.log import logger
from buttermilk._core.config import DataSourceConfig, ToolConfig
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    AgentInput,
    AgentOutput,
    AllMessages,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerResponse,
    OOBMessages,
    TaskProcessingComplete,  # Keep for type hints if needed elsewhere
    TaskProcessingStarted,  # Keep for type hints if needed elsewhere
    ToolOutput,
    UserInstructions,
    UserMessage,
)

# Removed import: from buttermilk._core.evaluate import evaluate
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.flow import KeyValueCollector
from buttermilk._core.types import Record
from buttermilk.utils.validators import convert_omegaconf_objects, lowercase_validator
from functools import wraps  # Import wraps for decorator


# Forward reference for type hint
if TYPE_CHECKING:
    from buttermilk.bm import BM


#########
# Agent
#
# A simple class with a function that process input.
#
#
##########


# --- Custom Decorator for Autogen Handlers ---
def buttermilk_handler(message_types: type):
    """
    Decorator to mark methods within a Buttermilk Agent as handlers
    for specific Autogen message types.
    The AutogenAgentAdapter will look for this marker.

    Args:
        message_type: The type of message this handler should process.
    """

    def decorator(func):
        # Attach the message type information to the function object
        setattr(func, "_buttermilk_handler_message_type", message_types)
        setattr(func, "_is_buttermilk_handler", True)  # Marker attribute

        @wraps(func)  # Preserve original function metadata
        async def wrapper(*args, **kwargs):  # The wrapper itself doesn't need to do much here
            return await func(*args, **kwargs)

        # Attach marker also to the wrapper if needed, but primarily to original func
        setattr(wrapper, "_buttermilk_handler_message_type", message_types)
        setattr(wrapper, "_is_buttermilk_handler", True)  # Marker attribute
        return wrapper

    return decorator


# --- End Custom Decorator ---


class AgentConfig(BaseModel):  # Restore class definition
    """Base configuration for all agents."""

    id: str = Field(default="")
    agent_obj: str = Field(  # Keep this if used for dynamic loading
        default="",
        description="The object name to instantiate",
        exclude=True,
    )
    role: Annotated[str, AfterValidator(lowercase_validator)] = Field(
        ...,
        description="The role type that this agent fulfils.",
    )

    name: str = Field(default="", description="The human friendly name of this agent type.")

    description: str = Field(
        ...,
        description="Short explanation of what this agent type does",
    )
    tools: list[ToolConfig] = Field(
        default=[],
        description="Tools the agent can invoke",
    )
    data: list[DataSourceConfig] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}
    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    _id: str = PrivateAttr(default_factory=lambda: uuid()[:6])
    base_name: str | None = Field(default=None, description="Base component of friendly name, derived from name field on init.")
    _name_components: list[str] = ["base_name", "_id"]
    _validate_parameters = field_validator("parameters", mode="before")(convert_omegaconf_objects())

    # @model_validator(mode="before")
    # def _generate_name(cls, values):
    #     # Take name out from the human friendly name passed in by config
    #     # and replace it with our generated unique
    #     values["id"] = uuid()[:6]
    #     values["name"] = f"{values['name']} #{values['id']}"
    #     values["id"] = f"{values['role']}-{values['id']}"

    @model_validator(mode="after")
    def _generate_name(self):
        # add a unique ID to a variant's name and role
        self.id = f"{self.role}-{self._id}"
        if not self.base_name:
            if not self.name:
                raise ValueError("You must provide a human friendly 'name' for agents.")
            self.base_name = self.name
        components = [getattr(self, x, None) for x in self._name_components]
        self.name = " ".join([x for x in components if x])
        return self


class Agent(AgentConfig):  # Agent inherits the restored fields
    """Base Agent interface for all processing units.

    Agents are stateful. Context is stored internally by the agent and merged
    with data passed via AgentInput. _reset() clears state.
    """

    _trace_this = True  # Flag for potential future use, currently weave is explicit

    _records: list[Record] = PrivateAttr(default_factory=list)
    _model_context: ChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    _message_types_handled: type[FlowMessage] = PrivateAttr(default=AgentInput)  # TODO: Revisit if needed
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))  # Keep for potential Autogen compatibility layer
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)

    async def _check_heartbeat(self, timeout=60) -> bool:
        """Check if the heartbeat queue is empty."""
        try:
            await asyncio.wait_for(self._heartbeat.get(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """Add local agent state to inputs before processing."""
        # add parameters
        inputs.parameters.update(self.parameters)

        # Fill inputs based on input map defined in config
        inputs.inputs.update(self._data._resolve_mappings(self.inputs))

        # Add context and records from agent's memory
        inputs.context.extend(await self._model_context.get_messages())
        inputs.records.extend(self._records)

        return inputs

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> None:
        """Save incoming messages from *other* agents to update internal state."""
        # Look for matching roles in our inputs mapping
        if isinstance(message, (AgentOutput, ConductorResponse)):
            datadict = {source.split("-", maxsplit=1)[0]: message.model_dump()}

            for key, mapping in self.inputs.items():
                if mapping and isinstance(mapping, str):
                    if value := jmespath.search(mapping, datadict):
                        if key == "records":
                            # records are stored separately in our memory cache
                            self._records.extend(value)
                            continue
                        self._data.add(key, value)
                else:
                    logger.warning(
                        f"Input mapping for {self.id} is too sophisticated and we haven't written the code to interpret it yet: {key} to {mapping}"
                    )
                    continue

        # Add message content to model context if appropriate
        if isinstance(message, (AgentOutput, ConductorResponse)):
            # Use content if available
            if message.content:
                await self._model_context.add_message(AssistantMessage(content=str(message.content), source=source))
        elif isinstance(message, UserInstructions):
            # Use prompt for UserInstructions, check if content exists (it's on ToolOutput)
            prompt_content = getattr(message, "prompt", None)
            if prompt_content and not str(prompt_content).startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(prompt_content), source=source))
        elif isinstance(message, ToolOutput):
            # Use content for ToolOutput
            tool_content = getattr(message, "content", None)  # Use getattr for safety
            if tool_content and not str(tool_content).startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(tool_content), source=source))
        else:
            # don't log other types of messages to history
            pass

    async def _process(
        self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs
    ) -> AgentOutput | ToolOutput | None:
        """Internal process function. Implement core agent logic here. Traced by Weave."""
        # Example:
        # logger.info(f"Agent {self.role} processing inputs: {inputs.inputs}")
        # await asyncio.sleep(1) # Simulate work
        # output_content = f"Processed: {inputs.prompt}"
        # return AgentOutput(role=self.role, content=output_content, inputs=inputs)
        raise NotImplementedError("Subclasses must implement the _process method.")

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> OOBMessages | None:
        """Handle out-of-band control messages if needed."""
        logger.debug(f"Agent {self.role} {self.name} dropping control message: {message}")
        return None

    async def __call__(
        self,
        message: FlowMessage,
        cancellation_token: CancellationToken | None = None,
        source: str = "unknown",  # Identifier of the sender, if known
        **kwargs,
    ) -> AgentOutput | ToolOutput | OOBMessages | None:
        """Primary entry point called by the Orchestrator."""
        result = None
        call = None  # For weave logging

        try:
            if isinstance(message, AgentInput):
                # Prepare final input by adding agent state
                final_input = await self._add_state_to_input(message)

                # Execute core logic and trace
                _traced_process = weave.op(self._process, call_display_name=self.name)
                result, call = await _traced_process.call(message=final_input, cancellation_token=cancellation_token, **kwargs)
                # Weave swallows errors. They should be reported in code below, so don't raise again.
                if call.exception:
                    logger.error(f"Agent {self.id} hit error processing request: {call.exception}")

                # --- Evaluation Logic ---
                if isinstance(result, AgentOutput) and not result.is_error and final_input.records:
                    # Check for ground truth in records
                    ground_truth_record = next((r for r in final_input.records if getattr(r, "ground_truth", None) is not None), None)
                    # if ground_truth_record:
                    #     # Evaluation logic moved to Orchestrator
                    #     pass
                    #     # evaluation_score = await evaluate(
                    #     #     output=result,
                    #     #     ground_truth=ground_truth_record.ground_truth,
                    #     #     criteria=final_input.parameters.get("criteria"),  # Or get from self.params
                    #     # )
                    #     # if evaluation_score and call:
                    #     #     # Log evaluation to Weave trace associated with the agent's call
                    #     #     call.log({"evaluation": evaluation_score.model_dump()})
                # --- End Evaluation Logic ---

            elif isinstance(message, GroupchatMessageTypes):
                # Listen to messages from other agents to update state
                result = await self._listen(message=message, cancellation_token=cancellation_token, source=source, **kwargs)

            elif isinstance(message, OOBMessages):
                # Handle control messages
                result = await self._handle_events(message=message, cancellation_token=cancellation_token, **kwargs)

            else:
                logger.warning(f"Agent {self.role} received unhandled message type: {type(message)}")
                result = None

        except Exception as e:
            logger.error(f"Error during agent {self.role} handle_message: {e}", exc_info=True)
            # Create an error output
            error_input = message if isinstance(message, AgentInput) else None
            result = AgentOutput(error=[str(e)], inputs=error_input)

        # Orchestrator is responsible for handling the result (e.g., routing AgentOutput)
        return result

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent (e.g., load resources). Called by Orchestrator."""
        pass  # Default implementation

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state. Called by Orchestrator."""
        self._records = []
        self._model_context = UnboundedChatCompletionContext()
        self._data = KeyValueCollector()
        # Clear heartbeat queue if necessary
        while not self._heartbeat.empty():
            try:
                self._heartbeat.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"Agent {self.role} ({self.id}) reset.")
        pass  # Allow subclasses to add more reset logic
