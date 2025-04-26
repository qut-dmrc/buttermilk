"""
Defines the core Agent base class, configuration, and handler decorator for Buttermilk.
"""

import asyncio
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps  # Import wraps for decorator
from typing import Any, Callable, Sequence

import jmespath  # For resolving input mappings
import weave  # For tracing

# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
from autogen_core import CancellationToken
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from pydantic import (
    BaseModel,
    PrivateAttr,
    computed_field,
)

from autogen_core.models import AssistantMessage, UserMessage
from buttermilk._core.config import AgentConfig

# Buttermilk core imports
from buttermilk._core.contract import (
    COMMAND_SYMBOL,  # Constant for command messages,
    AgentInput,  # Standard input message structure
    AgentOutput,
    ErrorEvent,  # Standard output message structure
    GroupchatMessageTypes,  # Union of types expected in group chat listening
    OOBMessages,  # Union of Out-Of-Band control messages
    ManagerMessage,  # Messages for display to the user
    ManagerRequest,  # Request sent to the manager
    StepRequest,  # Request to execute a specific step
    ToolOutput,  # The result of a tool execution
    UserInstructions,  # Message containing user instructions
)
from buttermilk._core.exceptions import FatalError, ProcessingError  # Custom exceptions
from buttermilk.utils.templating import KeyValueCollector  # Utility for managing state data
from buttermilk._core.log import logger  # Buttermilk logger instance
from buttermilk._core.types import Record  # Data record structure

# --- Buttermilk Handler Decorator ---


def buttermilk_handler(message_types: type):
    """
    Decorator to mark methods within a Buttermilk `Agent` subclass as handlers
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
        setattr(func, "_buttermilk_handler_message_type", message_types)
        setattr(func, "_is_buttermilk_handler", True)  # Marker attribute

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
    """
    Abstract Base Class for all Buttermilk agents.

    Inherits configuration from `AgentConfig` and defines the core execution
    interface and state management logic. Subclasses must implement the
    `_process` method for their primary logic.
    """

    # --- Internal State ---
    _records: list[Record] = PrivateAttr(default_factory=list)  # Stores data records relevant to the agent.
    _model_context: ChatCompletionContext = PrivateAttr(default_factory=UnboundedChatCompletionContext)  # Stores conversation history.
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)  # Stores key-value data extracted/passed via inputs mapping.
    # TODO: _message_types_handled seems unused. Confirm purpose or remove.
    # _message_types_handled: type[FlowMessage] = PrivateAttr(default=AgentInput)
    # Heartbeat queue, potentially used by orchestrator/adapter for liveness checks.
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))

    @computed_field()
    @property
    def _cfg(self) -> AgentConfig:
        """
        Extract AgentConfig parameters by creating a new AgentConfig from an Agent instance.

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
        """
        Initialize the agent state or resources. Called once by the orchestrator.
        Subclasses can override this to perform setup tasks (e.g., loading models, connecting to services).
        """
        logger.debug(f"Agent {self.id}: Base initialize.")
        pass  # Default implementation does nothing.

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """
        Reset the agent's internal state to its initial condition.
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
        pass  # Allow subclasses to add more reset logic.

    @weave.op()  # Mark the primary execution method for Weave tracing.
    async def __call__(
        self,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,  # Allows for additional context/callbacks from caller (e.g., adapter)
    ) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        """
        Primary entry point for agent execution, called by the orchestrator/adapter.

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

        # --- Execute Core Logic ---
        try:
            result = await self._process(message=final_input, cancellation_token=cancellation_token, **kwargs)

        except Exception as e:
            # Catch unexpected errors during _process.
            error_msg = f"Agent {self.id} error during _process: {e}"
            logger.error(error_msg)
            result = ErrorEvent(source=self.id, content=error_msg)

        # Ensure we always return an AgentOutput, even if _process somehow returned None.
        if result is None:
            msg = f"Agent {self.id} _process returned None. Creating default error output."
            logger.warning(msg)
            result = ErrorEvent(source=self.id, content=msg)

        logger.debug(f"Agent {self.id} finished __call__.")
        return result

    @abstractmethod
    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        """
        Abstract method for core agent logic. Subclasses MUST implement this.

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
        public_callback: Callable | None = None,  # Callback provided by adapter (unused by default)
        message_callback: Callable | None = None,  # Callback provided by adapter (unused by default)
        **kwargs,
    ) -> None:
        """
        Handles passively received messages from other agents in the group chat.

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
        # Map incoming message data to internal state (_data) using JMESPath defined in self.inputs.
        # TODO: Consider adding error handling for _extract_vars.
        datadict = {source.split("-", maxsplit=1)[0]: message.model_dump()}  # Simple source mapping
        extracted = await self._extract_vars(message=message, datadict=datadict)

        # Update internal state (_data, _records)
        for key, value in extracted.items():
            if key == "records":
                # Special handling for records: append to internal list.
                if not isinstance(value, Sequence) or isinstance(value, str):
                    value = [value]
                self._records.extend([r for r in value if isinstance(r, Record)])
                logger.debug(f"Agent {self.id} added {len(value)} records from {source}.")
            else:
                # Add other extracted data to the KeyValueCollector.
                self._data.add(key, value)
                logger.debug(f"Agent {self.id} updated state key '{key}' from {source}.")

        # Add relevant message content to the conversation history (_model_context).
        # Exclude command messages and potentially filter based on message type.
        # TODO: Refine logic for which message types/content get added to history.
        if isinstance(message, AgentOutput):
            # Use 'contents' if available (likely parsed output)
            content_to_add = getattr(message, "contents", None)
            if content_to_add:
                await self._model_context.add_message(
                    AssistantMessage(content=str(content_to_add), source=source)
                )  # Assume Assistant role for outputs
        elif isinstance(message, (UserInstructions, UserMessage)):
            content_to_add = getattr(message, "prompt", None)
            if content_to_add and not str(content_to_add).startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(content_to_add), source=source))
        elif isinstance(message, StepRequest) and message.content:
            # StepRequest has content field
            if not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content), source=source))
        elif isinstance(message, ManagerMessage) and message.content:
            # ManagerMessage and subclasses have content field
            if not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content), source=source))
        else:
            # don't log other types of messages to history by default
            logger.debug(f"Agent {self.id} ignored message type {type(message)} for context history.")
            pass

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> OOBMessages | None:
        """
        Handles Out-Of-Band (OOB) control messages.

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
        logger.debug(f"Agent {self.id} received OOB event: {type(message).__name__}. Default handler dropping.")
        # Example: Handle a specific reset request
        # if isinstance(message, ResetSignal):
        #    await self.on_reset(cancellation_token)
        #    return ResetAck(...) # Acknowledge reset
        return None  # Default: no response

    # --- Helper Methods ---

    async def _check_heartbeat(self, timeout=240) -> bool:
        """
        Waits for a heartbeat signal on the internal queue.

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
        except asyncio.TimeoutError:
            logger.warning(f"Agent {self.id} timed out waiting for heartbeat.")
            return False
        except Exception as e:
            logger.error(f"Agent {self.id}: Error checking heartbeat: {e}")
            return False

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """
        Augments the incoming `AgentInput` message with the agent's internal state.

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
            f"Agent {self.id}: Added state to input. Final input keys: {list(updated_inputs.inputs.keys())}, Context length: {len(updated_inputs.context)}, Records count: {len(updated_inputs.records)}"
        )
        return updated_inputs

    async def _extract_vars(self, message: GroupchatMessageTypes, datadict: dict) -> dict[str, Any]:
        """
        Extracts data from an incoming message based on the agent's `inputs` mapping configuration.

        Uses JMESPath expressions defined in `self.inputs` to query data from the
        `datadict` (which typically contains the message's model dump keyed by source).

        Args:
            message: The incoming message (used for context, potentially).
            datadict: A dictionary containing the message data, usually keyed by source agent role.

        Returns:
            A dictionary containing the extracted key-value pairs based on the mappings.
        """
        extracted = {}
        if not isinstance(self.inputs, dict):
            msg = f"Agent {self.id} 'inputs' configuration is not a dict: {type(self.inputs)}. Cannot extract vars."
            raise FatalError(msg)

        # Iterate through the input mappings defined in the agent's config.
        for key, mapping in self.inputs.items():
            if mapping and isinstance(mapping, str):
                try:
                    # Use JMESPath to search the datadict based on the mapping expression.
                    # Example: mapping = "judge.outputs.prediction" -> search datadict["judge"]["outputs"]["prediction"]
                    search_result = jmespath.search(mapping, datadict)
                    if search_result is not None and search_result != [] and search_result != {}:
                        # Store if JMESPath found something (could be False, 0, etc.)
                        extracted[key] = search_result
                        logger.debug(f"Agent {self.id}: Extracted '{key}' using mapping '{mapping}'. Found: {type(search_result)}")
                    else:
                        logger.debug(f"Agent {self.id}: Mapping '{mapping}' for key '{key}' yielded None.")

                except jmespath.exceptions.ParseError:
                    # If the mapping is just a plain string, not a JMESPath expression,
                    # potentially treat it as a default value or log a warning.
                    # Current behavior: Ignore non-JMESPath strings silently in this context.
                    # logger.debug(f"Mapping '{mapping}' for key '{key}' is not a valid JMESPath expression. Skipping.")
                    continue
                except Exception as e:
                    logger.warning(f"Agent {self.id}: Error applying JMESPath mapping '{mapping}' for key '{key}': {e}")
            else:
                # Handle non-string or empty mappings if necessary. Currently warns.
                logger.warning(f"Agent {self.id}: Invalid or complex input mapping for key '{key}': {mapping}. Skipping.")
        logger.debug(f"Agent {self.id}: Finished extracting vars. Keys extracted: {list(extracted.keys())}")
        return extracted
