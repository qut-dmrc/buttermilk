"""Defines the core Agent base class, configuration, and handler decorator for Buttermilk.
"""

import asyncio
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from functools import wraps  # Import wraps for decorator
from typing import TYPE_CHECKING, Any

import weave  # For tracing

if TYPE_CHECKING:
    from weave.trace.weave_client import Call, WeaveObject
# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
from autogen_core import CancellationToken
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
)

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core.config import AgentConfig

# Buttermilk core imports
from buttermilk._core.constants import COMMAND_SYMBOL  # Constant for command messages,
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,  # Standard input message structure
    AgentTrace,
    ErrorEvent,
    GroupchatMessageTypes,  # Union of types expected in group chat listening
    ManagerMessage,  # Messages from the user
    OOBMessages,  # Union of Out-Of-Band control messages
    TaskProcessingComplete,
    TaskProcessingStarted,  # Request to execute a specific step
)
from buttermilk._core.exceptions import ProcessingError  # Custom exceptions
from buttermilk._core.log import logger  # Buttermilk logger instance
from buttermilk._core.message_data import extract_message_data
from buttermilk._core.types import Record  # Data record structure
from buttermilk.utils.templating import KeyValueCollector  # Utility for managing state data

# --- Buttermilk Handler Decorator ---


def buttermilk_handler(message_types: type):
    """Decorator to mark methods within a Buttermilk `Agent` subclass as handlers
    for specific message types (typically originating from Autogen via `AutogenAgentAdapter`).

    The `AutogenAgentAdapter` inspects agent methods for the `_is_buttermilk_handler`
    attribute and the `_buttermilk_handler_message_type` attribute set by this decorator
    to determine which method should process an incoming message.

    Args:
        message_types: The specific message type (or tuple of types) that the
                       decorated method is intended to handle.

    """

    def decorator(func: Callable) -> Callable:
        # Attach metadata attributes to the original function object.
        # The adapter will look for these attributes.
        func._buttermilk_handler_message_type = message_types
        func._is_buttermilk_handler = True  # Marker attribute

        # Use functools.wraps to preserve the original function's metadata (name, docstring, etc.)
        # The wrapper itself currently doesn't add extra logic, but provides the structure.
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Simply await the original decorated function.
            return await func(*args, **kwargs)

        # It might be useful for introspection if the wrapper also had the markers,
        # but the primary mechanism relies on the markers being on the original function.
        # setattr(wrapper, "_buttermilk_handler_message_type", message_types)
        # setattr(wrapper, "_is_buttermilk_handler", True)
        return wrapper

    return decorator


# --- Base Agent Class ---


class Agent(AgentConfig):
    """Abstract Base Class for all Buttermilk agents.

    Inherits configuration from `AgentConfig` and defines the core execution
    interface and state management logic. Subclasses must implement the
    `_process` method for their primary logic.
    """

    session_id: str = Field(...,
        description="A unique session id for this specific flow execution.",
    )

    # --- Internal State ---
    _records: list[Record] = PrivateAttr(default_factory=list)  # Stores data records relevant to the agent.
    _model_context: ChatCompletionContext = PrivateAttr(default_factory=UnboundedChatCompletionContext)  # Stores conversation history.
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)  # Stores key-value data extracted/passed via inputs mapping.
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))

    model_config = {    # Pydantic Model Configuration
        "extra": "ignore",
        "arbitrary_types_allowed": False,  # Disallow arbitrary types unless explicitly handled.
        "populate_by_name": True,  # Allow population by field name.
        "validate_assignment": True,
    }

    @computed_field()
    @property
    def _cfg(self) -> AgentConfig:
        """Extract AgentConfig parameters by creating a new AgentConfig from an Agent instance.

        Args:
            agent: An instance of Agent or a subclass

        Returns:
            AgentConfig: A clean AgentConfig object with only the config fields

        """
        # Get all field names from AgentConfig
        agent_config_fields = set(AgentConfig.model_fields.keys())

        # Extract only the fields that are part of AgentConfig
        config_data = {field: getattr(self, field) for field in agent_config_fields if hasattr(self, field)}

        # Create a new AgentConfig instance with the extracted data
        return AgentConfig(**config_data)

    # --- Core Methods (Lifecycle & Interaction) ---

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent state or resources. Called once by the orchestrator.
        Subclasses can override this to perform setup tasks (e.g., loading models, connecting to services).
        """
        logger.debug(f"Agent {self.agent_name}: Base initialize.")
        # Default implementation does nothing.

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state to its initial condition.
        Called by the orchestrator, e.g., between different runs using the same agent instance.
        """
        logger.info(f"Agent {self.agent_name}: Resetting state.")
        self._records = []
        self._model_context = UnboundedChatCompletionContext()
        self._data = KeyValueCollector()
        # Clear heartbeat queue.
        while not self._heartbeat.empty():
            try:
                self._heartbeat.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Allow subclasses to add more reset logic.

    async def __call__(
        self,
        message: AgentInput,
        **kwargs,  # Allows for additional context/callbacks from caller (e.g., adapter)
    ) -> AgentOutput:
        """Primary execution method for the agent with standard inputs and outputs.
        Handles tracing as well as the core agent logic.
        entry point for agent execution, called by the orchestrator/adapter.

        Handles preparing input with agent state, tracing via Weave, calling the
        subclass's core `_process` logic, and returning the result.

        Args:
            message: The input message triggering the execution.
            **kwargs: Additional keyword arguments (e.g., callbacks from adapter).

        Returns:
            AgentOutput type.

        Raises:
            ProcessingError: If an error occurs during execution.

        """
        result = None
        logger.debug(f"Agent {self.agent_name} received input via __call__.")

        # --- Tracing ---
        trace_params = {"name": self.agent_name, "model": self._cfg.parameters.get("model"), **message.parameters, **message.metadata, **self.parameters}

        # Get the trace context if necessary
        parent_call: Call | WeaveObject = weave.get_current_call()
        if message.parent_call_id:
            try:
                parent_call = bm.weave.get_call(message.parent_call_id)
            except:
                pass
        child_call = None

        op = weave.op(self._process, call_display_name=self.agent_name)

        # --- Execute Core Logic ---
        try:
            child_call = bm.weave.create_call(op, inputs=message.model_dump(mode="json"),
                                              parent=parent_call, display_name=self.agent_name, attributes=trace_params)

            parent_call._children.append(child_call)  # Nest this call for tracing
            result = await self._process(message=message)
            result.call_id = child_call.id
            result.tracing_link = child_call.ui_url

        finally:
            if child_call:
                # Mark the child call as complete, regardless of success or failure.
                bm.weave.finish_call(child_call, output=result, op=op)

    async def invoke(
        self,
        message: AgentInput,
        public_callback: Callable[[Any], Awaitable[None]],
        message_callback: Callable[[Any], Awaitable[None]],
        cancellation_token: CancellationToken | None = None,
        **kwargs,  # Allows for additional context/callbacks from caller (e.g., adapter)
    ) -> AgentTrace:
        """Prepare input, call agent, and determine what to do with the result."""
        is_error = False

        # Prepare the final input by merging message data with internal agent state.
        try:
            final_input = await self._add_state_to_input(message)
        except Exception as e:
            raise ProcessingError(f"Agent {self.agent_name}: Error preparing input state: {e}")

        # Publish status update: Task Processing Started
        await public_callback(TaskProcessingStarted(agent_id=self.agent_id, role=self.role, task_index=0))

        try:
            result = await self.__call__(message=final_input)
        except Exception as e:
            logger.error(f"Agent {self.agent_name} error during __call__: {e}")
            result = ErrorEvent(source=self.agent_name, content=f"Failed to call agent: {e}")
            is_error = True

        # Publish status update: Task Complete (including error if error)
        await public_callback(
            TaskProcessingComplete(agent_id=self.agent_id, role=self.role, task_index=0, more_tasks_remain=False, is_error=is_error),
        )

        # Create the trace here with required values
        trace = AgentTrace(call_id=result.call_id, session_id=self.session_id, agent_id=self.agent_id,
            agent_info=self._cfg, tracing_link=result.tracing_link,
            inputs=final_input, parent_call_id=final_input.parent_call_id,
        )
        trace.outputs = getattr(result, "outputs", None)  # Extract outputs if available

        logger.debug(f"Agent {self.agent_name} {self.agent_name} finished task {message}.")

        # Publish the trace to the public callback
        await public_callback(trace)
        return trace

    @abstractmethod
    async def _process(self, *, message: AgentInput,
        **kwargs) -> AgentOutput:
        """Abstract method for core agent logic. Subclasses MUST implement this.

        This method receives the `AgentInput` (potentially augmented with internal state
        by `_add_state_to_input`) and should perform the agent's primary task.
        
        LLM agent implementations will typically return AgentTrace.
        Flow control agents (like host agents) will typically return StepRequest directly.
        Interface agents may return ManagerMessage.
        Tool agents typically return ToolOutput.

        Args:
            message: The fully prepared input message.
            **kwargs: Additional arguments passed from `__call__`.

        Returns:
            An AgentOutput object

        """
        raise NotImplementedError("Subclasses must implement the _process method.")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken,
        source: str = "",
        public_callback: Callable,
        message_callback: Callable,
        **kwargs: Any,
    ) -> None:
        """Handles passively received messages from other agents in the group chat.

        Updates internal state (`_data`, `_records`, `_model_context`) based on the
        message content and the agent's `inputs` mapping configuration. Does not
        typically generate a direct response.

        Args:
            message: The incoming message object.
            cancellation_token: Token for cancellation.
            source: Identifier of the message sender.
            public_callback: Callback to publish messages to the default topic.
            message_callback: Callback to publish messages to the incoming message's topic.
            **kwargs: Additional arguments.

        """
        logger.debug(f"Agent {self.agent_name} received message from {source} via _listen.")

        if isinstance(message, Record):
            self._records.append(message)
        elif isinstance(message, (AgentOutput, AgentTrace)) and isinstance(message.outputs, Record):
            self._records.append(message.outputs)
        else:
            # Extract data from the message using the utility function
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=self.inputs,
            )

            self._records.extend(extracted.pop("records", []))

            # Update internal state (_data, _records)
            found = []
            for key, value in extracted.items():
                if value and value != [] and value != {}:
                    self._data.add(key, value)
                    found.append(key)

            if found:
                logger.debug(f"Agent {self.agent_name} extracted keys [{found}] from {source}.")

        # Add relevant message content to the conversation history (_model_context).
        # Exclude command messages and potentially filter based on message type.
        # TODO: Refine logic for which message types/content get added to history.
        if isinstance(message, AgentTrace):
            # Use 'contents' if available (likely parsed output)
            content_to_add = getattr(message, "contents", None)
            if content_to_add:
                await self._model_context.add_message(
                    AssistantMessage(content=str(content_to_add), source=source),
                )  # Assume Assistant role for outputs

        elif isinstance(message, ManagerMessage) and message.content:
            # ManagerMessage and subclasses have content field
            content_str = str(message.content) if message.content else ""
            if not content_str.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=content_str, source=source))
        else:
            # don't log other types of messages to history by default
            logger.debug(f"Agent {self.agent_name} ignored message type {type(message)} for context history.")

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,  # Callback provided by adapter
        message_callback: Callable | None = None,  # Callback provided by adapter
        **kwargs,
    ) -> OOBMessages | None:
        """Handles Out-Of-Band (OOB) control messages.

        Subclasses can override this to react to specific control signals
        (e.g., reset requests, status queries) outside the main processing flow.
        Default implementation ignores most OOB messages.

        Args:
            message: The OOB message received.
            cancellation_token: Token for cancellation.
            **kwargs: Additional arguments.

        Returns:
            An OOB message as a response, or None if no direct response is needed.

        """
        # logger.debug(f"Agent {self.id} received OOB event: {type(message).__name__}. Default handler dropping.")
        # Example: Handle a specific reset request
        # if isinstance(message, ResetSignal):
        #    await self.on_reset(cancellation_token)
        #    return ResetAck(...) # Acknowledge reset
        return None  # Default: no response

    # --- Helper Methods ---

    async def _check_heartbeat(self, timeout=240) -> bool:
        """Waits for a heartbeat signal on the internal queue.

        Potentially used by orchestrators to check agent responsiveness or readiness.

        Args:
            timeout: Maximum seconds to wait for a heartbeat signal.

        Returns:
            True if a heartbeat was received within the timeout, False otherwise.

        """
        try:
            logger.debug(f"Agent {self.agent_name} waiting for heartbeat (timeout: {timeout}s)...")
            await asyncio.wait_for(self._heartbeat.get(), timeout=timeout)
            logger.debug(f"Agent {self.agent_name} received heartbeat.")
            self._heartbeat.task_done()  # Mark as processed
            return True
        except TimeoutError:
            logger.warning(f"Agent {self.agent_name} timed out waiting for heartbeat.")
            return False
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error checking heartbeat: {e}")
            return False

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """Augments the incoming `AgentInput` message with the agent's internal state.

        Merges configured parameters, resolved input mappings from `_data`,
        conversation history from `_model_context`, and stored `_records`.

        Args:
            inputs: The original `AgentInput` message.

        Returns:
            A new `AgentInput` instance with state merged into its fields.

        """
        # Create a copy to avoid modifying the original message directly.
        updated_inputs = inputs.model_copy(deep=True)

        # Merge agent's default parameters, letting message parameters override.
        # Ensure parameters dict exists.
        if updated_inputs.parameters is None:
            updated_inputs.parameters = {}
        merged_params = {**self.parameters, **updated_inputs.parameters}
        updated_inputs.parameters = merged_params

        # Resolve input mappings using data stored in self._data.
        # Ensure inputs dict exists.
        if updated_inputs.inputs is None:
            updated_inputs.inputs = {}
        try:
            extracted_data = {}
            for key in self.inputs.keys():
                data = self._data.get(key, [])
                # the _data collector object always uses lists. Get the non-empty values
                data = [x for x in data if x is not None and x != [] and x != {}]
                if data != []:
                    extracted_data[key] = data

            # Merge resolved mappings, letting message inputs override.
            merged_inputs_dict = {**extracted_data, **updated_inputs.inputs}
            updated_inputs.inputs = merged_inputs_dict
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error resolving input mappings: {e}")
            # Continue without resolved mappings? Or raise? Raising for clarity.
            raise ProcessingError(f"Error resolving input mappings for agent {self.agent_id}") from e

        # Prepend conversation history from agent's context.
        # Ensure context list exists.
        if updated_inputs.context is None:
            updated_inputs.context = []
        try:
            history = await self._model_context.get_messages()
            # Prepend history so message.context comes after agent's context
            updated_inputs.context = history + updated_inputs.context
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error retrieving model context: {e}")
            # Decide how to handle context retrieval failure. Continue without history?

        # Ensure records list exists. Use the last saved one if it doesn't.
        if not updated_inputs.records:
            updated_inputs.records = self._records[-1:] if self._records else []

        logger.debug(
            f"Agent {self.agent_id}: Added state to input. Final input keys: {list(updated_inputs.inputs.keys())}, Context length: {len(updated_inputs.context)}, Records count: {len(updated_inputs.records)}",
        )
        return updated_inputs
