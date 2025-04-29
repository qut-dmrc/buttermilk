"""Defines the core Agent base class, configuration, and handler decorator for Buttermilk.
"""

import asyncio
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from functools import wraps  # Import wraps for decorator
from typing import Any

import weave  # For tracing

# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
from autogen_core import CancellationToken
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, UserMessage
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai  # OpenAI integration  # noqa
from pydantic import (
    PrivateAttr,
    computed_field,
)

from buttermilk._core.config import AgentConfig

# Buttermilk core imports
from buttermilk._core.contract import (
    COMMAND_SYMBOL,  # Constant for command messages,
    AgentInput,  # Standard input message structure
    AgentOutput,
    ErrorEvent,
    GroupchatMessageTypes,  # Union of types expected in group chat listening
    ManagerMessage,  # Messages for display to the user
    ManagerRequest,  # Request sent to the manager
    OOBMessages,  # Union of Out-Of-Band control messages
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,  # Request to execute a specific step
    ToolOutput,  # The result of a tool execution
)
from buttermilk._core.exceptions import ProcessingError  # Custom exceptions
from buttermilk._core.log import logger  # Buttermilk logger instance
from buttermilk._core.message_data import extract_message_data, extract_records_from_data
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

    # --- Internal State ---
    _records: list[Record] = PrivateAttr(default_factory=list)  # Stores data records relevant to the agent.
    _model_context: ChatCompletionContext = PrivateAttr(default_factory=UnboundedChatCompletionContext)  # Stores conversation history.
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)  # Stores key-value data extracted/passed via inputs mapping.
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))

    model_config = {    # Pydantic Model Configuration
        "extra": "forbid",
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
        logger.debug(f"Agent {self.id}: Base initialize.")
        # Default implementation does nothing.

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state to its initial condition.
        Called by the orchestrator, e.g., between different runs using the same agent instance.
        """
        logger.info(f"Agent {self.id}: Resetting state.")
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

    @weave.op()  # Mark the primary execution method for tracing.
    @observe()
    async def __call__(
        self,
        message: AgentInput,
        public_callback: Callable[[Any], Awaitable[None]],
        message_callback: Callable[[Any], Awaitable[None]],
        cancellation_token: CancellationToken | None = None,
        **kwargs,  # Allows for additional context/callbacks from caller (e.g., adapter)
    ) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent | None:
        """Primary entry point for agent execution, called by the orchestrator/adapter.

        Handles preparing input with agent state, tracing via Weave, calling the
        subclass's core `_process` logic, and returning the result.

        Args:
            message: The input message triggering the execution.
            cancellation_token: Token to signal cancellation.
            **kwargs: Additional keyword arguments (e.g., callbacks from adapter).

        Returns:
            Either an AgentOutput for standard agent results, 
            or a direct message type (StepRequest, ManagerRequest, ManagerMessage)
            for flow control agents, or an ErrorEvent if there were errors.

        """
        result = None
        logger.debug(f"Agent {self.id} received input via __call__.")

        # Prepare the final input by merging message data with internal agent state.
        try:
            final_input = await self._add_state_to_input(message)
        except Exception as e:
            logger.error(f"Agent {self.id}: Error preparing input state: {e}")
            error_output = ErrorEvent(source=self.id, content=f"Failed to prepare input state: {e}")
            return error_output

        # --- Weave Tracing ---
        # Get the current Weave call context if available.
        call = weave.get_current_call()
        if call:
            call.set_display_name(self.name)  # Set trace display name
            # TODO: Log inputs? Be careful about large data/PII.
            logger.debug(f"Agent {self.id} __call__ executing within Weave trace: {getattr(call.ref, 'id', 'N/A')}")
        else:
            logger.warning(f"Agent {self.id} __call__ executing outside Weave trace context.")

        # --- Langfuse tracing ---
        langfuse_context.update_current_observation(name=self.name,
            metadata=self.parameters,
        )

        # --- Execute Core Logic ---
        try:
            await public_callback(TaskProcessingStarted(agent_id=self.id, role=self.role, task_index=0))

            result = await self._process(message=final_input, cancellation_token=cancellation_token,
        public_callback=public_callback,  # Callback provided by adapter
        message_callback=message_callback,  # Callback provided by adapter
          **kwargs)

        except Exception as e:
            # Catch unexpected errors during _process.
            error_msg = f"Agent {self.id} error during _process: {e}"
            logger.error(error_msg)
            result = ErrorEvent(source=self.id, content=error_msg)

        finally:
            # Publish status update: Task Complete (Error)
            await public_callback(
                TaskProcessingComplete(agent_id=self.id, role=self.role, task_index=0, more_tasks_remain=False, is_error=False),
            )
        # Ensure we always return an AgentOutput, even if _process somehow returned None.
        if result:
            await public_callback(result)
            return result

        logger.debug(f"Agent {self.id} finished __call__.")
        return None

    @abstractmethod
    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,  # Callback provided by adapter
        message_callback: Callable | None = None,  # Callback provided by adapter,
        **kwargs) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        """Abstract method for core agent logic. Subclasses MUST implement this.

        This method receives the `AgentInput` (potentially augmented with internal state
        by `_add_state_to_input`) and should perform the agent's primary task.
        
        LLM agent implementations will typically return AgentOutput.
        Flow control agents (like host agents) will typically return StepRequest directly.
        Interface agents may return ManagerRequest or ManagerMessage.
        Tool agents typically return ToolOutput.

        Args:
            message: The fully prepared input message.
            cancellation_token: Token to monitor for cancellation requests.
            **kwargs: Additional arguments passed from `__call__`.

        Returns:
            An AgentOutput, StepRequest, ManagerRequest, ManagerMessage or ErrorEvent

        """
        raise NotImplementedError("Subclasses must implement the _process method.")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",  # ID/Name of the sending agent
        public_callback: Callable | None = None,  # Callback provided by adapter
        message_callback: Callable | None = None,  # Callback provided by adapter
        **kwargs,
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
        logger.debug(f"Agent {self.id} received message from {source} via _listen.")

        # Extract data from the message using the utility function
        extracted = extract_message_data(
            message=message,
            source=source,
            input_mappings=self.inputs,
        )

        # Update internal state (_data, _records)
        found = []
        for key, value in extracted.items():
            if value and value != [] and value != {}:
                if key == "records":
                    # Extract records from the data
                    records_data = {"records": value}
                    new_records = extract_records_from_data(records_data)
                    # Add to internal records list
                    self._records.extend(new_records)
                    found.append(key)
                else:
                    # Add other extracted data to the KeyValueCollector
                    self._data.add(key, value)
                    found.append(key)

        if found:
            logger.debug(f"Agent {self.id} extracted keys [{found}] from {source}.")

        # Add relevant message content to the conversation history (_model_context).
        # Exclude command messages and potentially filter based on message type.
        # TODO: Refine logic for which message types/content get added to history.
        if isinstance(message, AgentOutput):
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
            logger.debug(f"Agent {self.id} ignored message type {type(message)} for context history.")

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
            logger.debug(f"Agent {self.id} waiting for heartbeat (timeout: {timeout}s)...")
            await asyncio.wait_for(self._heartbeat.get(), timeout=timeout)
            logger.debug(f"Agent {self.id} received heartbeat.")
            self._heartbeat.task_done()  # Mark as processed
            return True
        except TimeoutError:
            logger.warning(f"Agent {self.id} timed out waiting for heartbeat.")
            return False
        except Exception as e:
            logger.error(f"Agent {self.id}: Error checking heartbeat: {e}")
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
            logger.error(f"Agent {self.id}: Error resolving input mappings: {e}")
            # Continue without resolved mappings? Or raise? Raising for clarity.
            raise ProcessingError(f"Error resolving input mappings for agent {self.id}") from e

        # Prepend conversation history from agent's context.
        # Ensure context list exists.
        if updated_inputs.context is None:
            updated_inputs.context = []
        try:
            history = await self._model_context.get_messages()
            # Prepend history so message.context comes after agent's context
            updated_inputs.context = history + updated_inputs.context
        except Exception as e:
            logger.error(f"Agent {self.id}: Error retrieving model context: {e}")
            # Decide how to handle context retrieval failure. Continue without history?

        # Append stored records.
        # Ensure records list exists.
        if updated_inputs.records is None:
            updated_inputs.records = []
        updated_inputs.records.extend(self._records)

        logger.debug(
            f"Agent {self.id}: Added state to input. Final input keys: {list(updated_inputs.inputs.keys())}, Context length: {len(updated_inputs.context)}, Records count: {len(updated_inputs.records)}",
        )
        return updated_inputs

    # The _extract_vars method is no longer needed as we're using the utility functions
