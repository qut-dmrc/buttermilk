"""Defines the core Agent base class, its configuration, and the `buttermilk_handler`
decorator.

This module provides the foundational components for creating agents within the
Buttermilk framework. Agents are responsible for performing specific tasks as part
of a larger data processing flow. `AgentConfig` (from `config.py`) provides the
base configuration, and `Agent` provides the execution logic and state management.
The `buttermilk_handler` decorator is used to designate methods within agent
subclasses as handlers for specific message types, typically when integrating with
systems like Autogen.
"""

import asyncio
from abc import abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from functools import wraps  # Import wraps for decorator
from typing import TYPE_CHECKING, Any, Literal, Mapping, Union
import warnings

import weave  # For tracing - core dependency
from weave.trace.weave_client import Call, WeaveObject


if TYPE_CHECKING:
    from buttermilk._core.tool_definition import AgentToolDefinition
    from autogen_core import AgentRuntime

from autogen_core.tools import Tool, ToolSchema

# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
from autogen_core import (
    CancellationToken,
    message_handler,
    RoutedAgent,
    MessageContext,
    AgentId,
    AgentMetadata,
    DefaultTopicId,
    TopicId,
)
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
)

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core.config import AgentConfig

# Buttermilk core imports
from buttermilk._core.constants import COMMAND_SYMBOL  # Constant for command messages,
from buttermilk._core.context import agent_id_var
from buttermilk._core.types import Record  # Data record structure
from buttermilk.utils.templating import KeyValueCollector  # Utility for managing state data
from buttermilk._core.contract import (
    AgentAnnouncement,
    AgentInput,
    AgentOutput,  # Standard input message structure
    AgentTrace,
    ErrorEvent,
    GroupchatMessageTypes,  # Union of types expected in group chat listening
    ManagerMessage,  # Messages from the user
    OOBMessages,
    StepRequest,  # Union of Out-Of-Band control messages
    TaskProcessingComplete,
    TaskProcessingStarted,  # Request to execute a specific step
    FlowEvent,
    FlowMessage,
    AllMessages,
    HeartBeat,
)
from buttermilk._core.exceptions import ProcessingError  # Custom exceptions
from buttermilk._core.log import logger  # Buttermilk logger instance
from buttermilk._core.message_data import extract_message_data
from buttermilk._core.retry import RetryWrapper

# --- Base Agent Class ---


class Agent(RoutedAgent):
    """Base class for all Buttermilk agents, integrating with autogen_core's RoutedAgent.

    This class serves as the foundation for all specialized agents within the
    Buttermilk framework. It inherits its configuration structure from `AgentConfig`
    and defines a common interface for agent execution, state management, and
    lifecycle hooks.

    Subclasses are expected to implement the `_process` method, which contains
    the core logic for that agent's specific task (e.g., interacting with an
    LLM, calling an API, transforming data).

    The `Agent` class manages internal state such as data records, conversation
    history (model context), and extracted key-value data. It also provides
    methods for initialization, resetting state, and handling various types of
    messages and events.

    Attributes:
        session_id (str): A unique identifier for the current flow execution session.
            This helps in tracking and correlating agent activities within a specific run.
        _records (list[Record]): Internal list to store data `Record` objects relevant
            to the agent's current context or processing task.
        _model_context (ChatCompletionContext): Internal store for conversation history,
            particularly for agents interacting with chat-based models. Defaults to
            an `UnboundedChatCompletionContext`.
        _data (KeyValueCollector): Internal store for arbitrary key-value data that
            can be extracted from incoming messages (based on `inputs` mappings) or
            accumulated during processing.
        _heartbeat (asyncio.Queue): An internal queue used for heartbeat signals,
            allowing orchestrators or other components to check agent responsiveness.
        model_config (dict): Pydantic model configuration.
            - `extra`: "ignore" - Ignores extra fields during model parsing.
            - `arbitrary_types_allowed`: False - Disallows arbitrary types unless explicitly handled.
            - `populate_by_name`: True - Allows population by field name (alias support).
            - `validate_assignment`: True - Validates fields on assignment.

    """

    # --- Autogen Core Protocol Implementation ---
    _id: Union[AgentId, None] = None
    _runtime: Union["AgentRuntime", None] = None
    _config: AgentConfig

    # --- Configuration properties (delegated to _config) ---
    @property
    def agent_id(self) -> str:
        return self._config.agent_id

    @property
    def agent_name(self) -> str:
        return self._config.agent_name

    @property
    def role(self) -> str:
        return self._config.role

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._config.parameters or {}

    @property
    def inputs(self) -> Union[dict[str, str], None]:
        return self._config.inputs

    @property
    def session_id(self) -> str:
        """Get session_id from config if available."""
        return getattr(self._config, 'session_id', '')

    def __init__(self, **data: Any) -> None:
        """Initialize the Agent with configuration data and setup RoutedAgent."""
        # Create AgentConfig from the data
        self._config = AgentConfig(**data)

        # Initialize RoutedAgent with description
        RoutedAgent.__init__(self, description=self._config.description)

        # Initialize private attributes
        self._records = []
        self._model_context = UnboundedChatCompletionContext()
        self._data = KeyValueCollector()
        self._heartbeat = asyncio.Queue(maxsize=1)

    @property
    def metadata(self) -> AgentMetadata:
        """Metadata of the agent."""
        if self._id is None:
            raise RuntimeError("Agent not bound to runtime")
        return AgentMetadata(key=self._id.key, type=self._id.type, description=self.description)

    @property
    def id(self) -> AgentId:
        """ID of the agent."""
        if self._id is None:
            raise RuntimeError("Agent not bound to runtime")
        return self._id

    async def bind_id_and_runtime(self, id: AgentId, runtime: "AgentRuntime") -> None:
        """Function used to bind an Agent instance to an `AgentRuntime`.

        Args:
            id (AgentId): ID of the agent.
            runtime (AgentRuntime): AgentRuntime instance to bind the agent to.
        """
        self._id = id
        self._runtime = runtime

    async def save_state(self) -> Mapping[str, Any]:
        """Save the state of the agent. The result must be JSON serializable."""
        warnings.warn("save_state not implemented", stacklevel=2)
        return {}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load in the state of the agent obtained from `save_state`.

        Args:
            state (Mapping[str, Any]): State of the agent. Must be JSON serializable.
        """
        warnings.warn("load_state not implemented", stacklevel=2)
        pass

    async def close(self) -> None:
        """Called when the runtime is closed"""
        await self.cleanup()

    # --- Internal State ---
    _records: list[Record]
    _model_context: ChatCompletionContext
    _data: KeyValueCollector
    _heartbeat: Union[asyncio.Queue[Union[bool, None]], asyncio.Queue]
    _announcement_callback: Union[Callable[[Any], Awaitable[None]], None]

    @property
    def _cfg(self) -> AgentConfig:
        """Provides the agent's configuration.

        Returns:
            AgentConfig: The agent's configuration instance.
        """
        return self._config

    # --- Core Methods (Lifecycle & Interaction) ---

    async def cleanup(self) -> None:
        """Cleanup agent resources and state.

        Called when the agent is being shut down or when the session is being cleaned up.
        Subclasses should override this method to cleanup any resources they have allocated
        (e.g., file handles, network connections, background tasks).

        The base implementation clears internal state and should be called by subclasses
        using super().cleanup().
        """
        logger.debug(f"Agent {self.agent_name}: Base cleanup.")

        # Clear internal state
        self._records.clear()
        self._data.clear()

        # Clear model context
        if hasattr(self._model_context, "clear"):
            self._model_context.clear()
        elif hasattr(self._model_context, "reset"):
            await self._model_context.reset()

        # Clear heartbeat queue
        while not self._heartbeat.empty():
            try:
                self._heartbeat.get_nowait()
            except asyncio.QueueEmpty:
                break

    # --- Announcement Methods ---

    def get_available_tools(self) -> list[Tool]:
        """Get list of tools this agent can respond to.

        Returns:
            list[Tool]: List of tools.
        """
        return []

    def get_supported_message_types(self) -> list[str]:
        """Get list of OOBMessage types this agent handles.
        
        Returns:
            list[str]: List of supported message type names.
        """
        # Check if agent has defined supported messages
        if hasattr(self, "_supported_oob_messages"):
            return self._supported_oob_messages
        return []

    def create_announcement(
        self,
        announcement_type: Literal["initial", "response", "update"],
        status: Literal["joining", "active", "leaving"],
        responding_to: str | None = None
    ) -> AgentAnnouncement:
        """Create an agent announcement message.
        
        Args:
            announcement_type: Type of announcement (initial, response, update).
            status: Current agent status (joining, active, leaving).
            responding_to: Agent ID being responded to (for response type).
            
        Returns:
            AgentAnnouncement: The announcement message.
        """
        content_map = {
            "joining": f"Agent {self.agent_name} joining",
            "active": f"Agent {self.agent_name} active",
            "leaving": f"Agent {self.agent_name} leaving"
        }

        # Get ALL tool definitions (decorated methods + configured tools)
        tool_definitions = self.get_all_tool_definitions()

        # Use the first tool definition as the primary one
        # This now includes both decorated methods and configured tools
        primary_tool = tool_definitions[0] if tool_definitions else None

        return AgentAnnouncement(
            content=content_map.get(status, f"Agent {self.agent_name} status: {status}"),
            agent_config=self._cfg,
            available_tools=[tool.name for tool in self.get_available_tools()],
            supported_message_types=self.get_supported_message_types(),
            tool_definition=primary_tool,
            status=status,
            announcement_type=announcement_type,
            responding_to=responding_to,
            source=self.agent_id
        )

    async def send_announcement(
        self,
        announcement_type: Literal["initial", "response", "update"],
        status: Literal["joining", "active", "leaving"],
        responding_to: str | None = None,
        topic_id: TopicId | None = None
    ) -> None:
        """Send an agent announcement message.
        
        Args:
            announcement_type: Type of announcement.
            status: Current agent status.
            responding_to: Agent ID being responded to (for response type).
            topic_id: Topic to publish to (defaults to DefaultTopicId).
        """
        try:
            announcement = self.create_announcement(
                announcement_type=announcement_type,
                status=status,
                responding_to=responding_to
            )
            if hasattr(self, '_runtime') and self._runtime:
                await self.publish_message(
                    announcement, 
                    topic_id=topic_id or DefaultTopicId(type="default")
                )
                logger.debug(f"Agent {self.agent_name} sent {announcement_type} announcement with status {status}")
            else:
                logger.warning(f"Agent {self.agent_name} has no runtime to send announcement")
        except Exception as e:
            logger.warning(f"Agent {self.agent_name} failed to send announcement: {e}")

    async def initialize(
        self,
        **kwargs: Any
    ) -> None:
        """Initializes agent state or resources and sends joining announcement.

        Called once by the orchestrator. Subclasses can override this method to perform 
        asynchronous setup tasks such as loading models, establishing connections to external 
        services, or initializing complex state.
        
        Args:
            **kwargs: Arbitrary keyword arguments that might be passed by the
                orchestrator or calling context, providing additional setup parameters.
        """
        logger.debug(f"Agent {self.agent_name}: Base initialize.")
        # Set the agent ID in the context variable for tracing
        agent_id_var.set(self.agent_id)

        # Send joining announcement
        await self.send_announcement(
            announcement_type="initial",
            status="joining"
        )

    async def cleanup_with_announcement(self) -> None:
        """Cleanup agent and send leaving announcement."""
        # Send leaving announcement
        await self.send_announcement(
            announcement_type="update",
            status="leaving"
        )

        # Call base cleanup
        await self.cleanup()

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Resets the agent's internal state to its initial condition.

        This method is typically called by the orchestrator, for example,
        between different runs or tasks when the same agent instance is reused.
        It should clear any accumulated data, conversation history, or other
        stateful information to ensure a clean slate for the next operation.

        The base implementation resets `_records`, `_model_context`, `_data`,
        and clears the `_heartbeat` queue. Subclasses should call `super().on_reset()`
        if they override this method and then add their own specific reset logic.

        Args:
            cancellation_token: An optional `CancellationToken` that can be
                used to signal that the reset operation should be aborted.

        """
        logger.info(f"Agent {self.agent_name}: Resetting state.")
        self._records = []
        self._model_context = UnboundedChatCompletionContext()
        self._data = KeyValueCollector()
        # Clear heartbeat queue by consuming all items.
        while not self._heartbeat.empty():
            try:
                self._heartbeat.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Subclasses can add more reset logic here or by overriding.

    async def __call__(
        self,
        message: AgentInput,
        **kwargs: Any,
    ) -> AgentOutput:
        """Primary execution entry point for the agent, handling a single `AgentInput`.

        This method orchestrates the core processing logic of the agent. It is
        responsible for:
        1. Setting up tracing for the operation using Weave.
        2. Calling the abstract `_process` method, which must be implemented by
           subclasses to perform the agent's specific task.
        3. Ensuring the trace call is properly finished, regardless of success or failure.

        This method is typically invoked by an orchestrator or the runtime
        when a message needs to be processed by the agent.
        It is not intended to be called directly by user code in most cases;
        prefer using `invoke` for a more complete interaction pattern that includes
        callbacks and status updates.

        Args:
            message: The `AgentInput` message containing the data and parameters
                for the agent to process. This input may have already been augmented
                with agent state if called via `invoke` (which uses `_add_state_to_input`).
            **kwargs: Additional keyword arguments that might be passed by the caller.
                These are not directly used by the base `__call__` but are available
                for potential extensions or specific `_process` implementations.

        Returns:
            AgentOutput: An object containing the results of the agent's processing.
                The exact structure of `AgentOutput` can vary depending on the agent type.

        Raises:
            ProcessingError: If an error occurs during the `_process` method execution,
                it should be caught and potentially wrapped in a `ProcessingError`
                by the `_process` method itself or by the caller. This method primarily
                focuses on tracing orchestration.

        """
        result: AgentOutput | None = None  # Ensure result is defined for finally block
        logger.debug(f"Agent {self.agent_name} received input via __call__.")

        # --- Tracing ---
        trace_params = {"name": self.agent_name, "model": self._cfg.parameters.get("model"), **message.parameters, **message.metadata, **self.parameters}

        parent_call: Call | WeaveObject | None = None
        if message.parent_call_id:
            async def get_weave_call_with_retry(call_id: str) -> Call | WeaveObject:
                """Retry getting weave call to handle async upload timing."""
                return bm.weave.get_call(call_id)

            # Use RetryWrapper with shorter delays for weave call retrieval
            retry_wrapper = RetryWrapper(
                client=None,  # Not using client, just the retry logic
                max_retries=3,
                min_wait_seconds=0.1,
                max_wait_seconds=1.0,
                jitter_seconds=0.1,
                cooldown_seconds=0
            )

            try:
                parent_call = await retry_wrapper._execute_with_retry(
                    get_weave_call_with_retry,
                    message.parent_call_id
                )
            except Exception as e:  # Broad exception for Weave call retrieval
                logger.error(f"Agent {self.agent_name}: Could not retrieve parent call ID {message.parent_call_id} after retries. Error: {e}")
                parent_call = weave.get_current_call()  # Fallback to current call if specified parent not found
        else:
            parent_call = weave.get_current_call()

        child_call: Call | None = None
        op = weave.op(self._process, call_display_name=self.agent_name)

        # --- Execute Core Logic ---
        try:
            child_call = bm.weave.create_call(op, inputs=message.model_dump(mode="json"),
                                              parent=parent_call, display_name=self.agent_name, attributes=trace_params)

            if parent_call and child_call:
                parent_call._children.append(child_call)  # Nest this call for tracing
            result = await self._process(message=message)
            result.call_id = child_call.id
            result.tracing_link = child_call.ui_url

            return result
        finally:
            if child_call:
                # Mark the child call as complete, regardless of success or failure.
                # Output is passed to bm.weave.finish_call if result is not None
                bm.weave.finish_call(child_call, output=result or None, op=op)

    @message_handler
    async def handle_invocation(
        self,
        message: Union[AgentInput, StepRequest],
        ctx: MessageContext,
    ) -> AgentTrace:
        """Prepare input, calls the agent's core logic, and handles callbacks.

        This method provides a more complete interaction pattern than `__call__`.
        It performs the following steps:
        1. Augments the incoming `message` with the agent's internal state using
           `_add_state_to_input`.
        2. Notifies listeners (via `public_callback`) that task processing has started.
        3. Invokes the agent's core logic via `self.__call__`.
        4. Handles any errors during execution, creating an `ErrorEvent` if necessary.
        5. Notifies listeners that task processing has completed (or failed).
        6. Constructs an `AgentTrace` object containing details of the execution,
           including inputs, outputs (if any), and configuration.
        7. Publishes the `AgentTrace` to listeners.

        This is the recommended method for external callers (like orchestrators or
        adapters) to interact with the agent for a single processing step.

        Args:
            message: The initial `AgentInput` message for the agent.
            public_callback: An asynchronous callback function to publish messages
                (like status updates, traces) to a general or public topic.
            cancellation_token: An optional `CancellationToken` to signal if the
                operation should be aborted. (Currently not deeply integrated into the core loop).
            **kwargs: Additional keyword arguments that might be passed by the caller.
                These are not directly used by `invoke` but are available for potential extensions.

        Returns:
            AgentTrace: An object detailing the agent's execution for this invocation.

        Raises:
            ProcessingError: If `_add_state_to_input` fails. (Errors from `__call__`
                are caught and reported in the `AgentTrace` and `TaskProcessingComplete` event).

        """
        is_error = False

        # Define publishing helper
        async def publish(msg: Any) -> None:
            if hasattr(self, '_runtime') and self._runtime:
                await self.publish_message(msg, topic_id=ctx.topic_id or DefaultTopicId(type="default"))
            else:
                logger.warning(f"No runtime available for publishing: {msg}")

        try:
            final_input = await self._add_state_to_input(message)
        except Exception as e:
            # Log the error and re-raise as ProcessingError
            logger.error(f"Agent {self.agent_name}: Error preparing input state: {e}")
            raise ProcessingError(f"Agent {self.agent_name}: Error preparing input state: {e}") from e

        await publish(TaskProcessingStarted(agent_id=self.agent_id, role=self.role, task_index=0))

        try:
            result = await self.__call__(message=final_input)

            # Create the trace here with required values
            trace = AgentTrace(call_id=result.call_id, agent_id=self.agent_id,
                agent_info=self._cfg, tracing_link=result.tracing_link,
                inputs=final_input, parent_call_id=final_input.parent_call_id, outputs=result.outputs,
            )
        except ProcessingError as e:
            logger.error(f"Agent {self.agent_name} error during __call__: {e}", exc_info=True)
            result = ErrorEvent(source=self.agent_name, content=f"Processing error: {e}")
            is_error = True
            trace = AgentTrace(
                call_id=result.call_id,
                agent_id=self.agent_id,
                agent_info=self._cfg,
                inputs=final_input,
                parent_call_id=final_input.parent_call_id,
                outputs=result,
            )
        except Exception as e:
            logger.error(f"Agent {self.agent_name} error during __call__: {e}", exc_info=True)
            result = ErrorEvent(source=self.agent_name, content=f"Failed to call agent: {e}")
            is_error = True
            trace = AgentTrace(call_id=result.call_id, agent_id=self.agent_id,
                agent_info=self._cfg,
                inputs=final_input, parent_call_id=final_input.parent_call_id, outputs=result,
            )

        # Publish status update: Task Complete (including error if error)
        await publish(
            TaskProcessingComplete(agent_id=self.agent_id, role=self.role, task_index=0, more_tasks_remain=False, is_error=is_error),
        )

        logger.debug(f"Agent {self.agent_name} {self.agent_name} finished task {message}.")

        # Publish the trace
        await publish(trace)
        return trace

    async def invoke(
        self,
        message: Union[AgentInput, StepRequest],
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        **kwargs: Any,
    ) -> AgentTrace:
        """Public invoke method that delegates to the message handler.
        
        This method is kept for backwards compatibility and is called by external code.
        It creates a MessageContext and delegates to the handle_invocation message handler.
        """
        # If we have a runtime, use the proper message handler routing
        if hasattr(self, '_runtime') and self._runtime and hasattr(self, '_id') and self._id:
            # Send the message through the runtime which will route it back to our handler
            result = await self._runtime.send_message(
                message,
                sender=self._id,
                recipient=self._id,
                cancellation_token=cancellation_token,
            )
            return result
        else:
            # Fallback: Call the handler directly if no runtime available
            ctx = MessageContext(
                sender=AgentId(type=source or "unknown", key=source or "unknown"),
                topic_id=DefaultTopicId(type="default"),
                is_rpc=True,
                cancellation_token=cancellation_token or CancellationToken(),
            )

            result = await self.handle_invocation(message, ctx)
            return result

    @abstractmethod
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Abstract method for the agent's core processing logic.

        Subclasses **MUST** implement this method to define their specific behavior.
        This method receives an `AgentInput` (which may have been augmented with
        the agent's internal state by `_add_state_to_input` if called via `invoke`
        or `__call__`) and should perform the agent's primary task.

        The type of `AgentOutput` returned can vary based on the agent's purpose:
        - LLM agents might return structured data within `AgentOutput.outputs`.
        - Flow control agents (e.g., a host agent managing sub-tasks) might
          return messages like `StepRequest` (wrapped in `AgentOutput`).
        - Interface agents (e.g., for user interaction) might return `ManagerMessage`
          (wrapped in `AgentOutput`).
        - Tool-using agents might return `ToolOutput` (wrapped in `AgentOutput`).

        Args:
            message: The `AgentInput` message containing all necessary data and
                parameters for the agent to perform its task. The `inputs` attribute
                of the message will contain data resolved from `self._data` based on
                `self.inputs` mappings, and `context` will include historical messages.
            **kwargs: Additional keyword arguments that might be passed from the
                `__call__` method. Subclass implementations can choose to use these
                or ignore them.

        Returns:
            AgentOutput: An object containing the results of the processing.
                This object should include a `call_id` field if tracing is active,
                which is typically handled by the `__call__` method.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
            ProcessingError: Subclasses should raise this (or a more specific
                subclass of it) if a non-recoverable error occurs during processing.

        """
        raise NotImplementedError("Subclasses must implement the _process method.")

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None:
        """Handle passively received messages from other agents or sources.

        This method is called when the agent receives a message that is not a
        direct request for it to perform an action (i.e., not via `invoke` or
        `__call__`). Its primary purpose is to update the agent's internal state
        (`_data`, `_records`, `_model_context`) based on the content of the
        incoming message and the agent's `inputs` mapping configuration.

        It does not typically generate a direct response message but may publish
        internal status updates or log information.

        Args:
            message: The incoming message object. This can be various types as
                defined by `GroupchatMessageTypes` (e.g., `Record`, `AgentOutput`,
                `AgentTrace`, `ManagerMessage`).
            cancellation_token: A `CancellationToken` to signal if the listening
                operation should be aborted. (Currently not deeply integrated).
            source: A string identifier for the sender or source of the message.
            public_callback: An asynchronous callback function to publish messages
                to a general or public topic. (Currently not used in base implementation).
            **kwargs: Additional keyword arguments that might be passed by the caller.

        """
        # Extract source from sender
        source = str(ctx.sender).split("/", maxsplit=1)[0] if ctx.sender else "unknown"

        # Define publishing helper for handle_groupchat_message
        async def publish(msg: Any) -> None:
            if hasattr(self, '_runtime') and self._runtime:
                await self.publish_message(msg, topic_id=ctx.topic_id or DefaultTopicId(type="default"))
            else:
                logger.warning(f"No runtime available for publishing: {msg}")

        logger.debug(f"Agent {self.agent_name} received message from '{source}' via handle_groupchat_message.")

        # Handle Record type messages directly
        if isinstance(message, Record):
            self._records.append(message)
            logger.debug(f"Agent {self.agent_name} added Record {message.record_id} to internal state.")

        # Handle AgentOutput or AgentTrace containing a Record in its 'outputs'
        elif isinstance(message, (AgentOutput, AgentTrace)) and isinstance(getattr(message, "outputs", None), Record):
            self._records.append(message.outputs)  # type: ignore
            logger.debug(f"Agent {self.agent_name} added Record from {type(message).__name__}.outputs to internal state.")

        # For other message types, use extract_message_data utility
        elif self.inputs:  # Only extract if input mappings are defined
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=self.inputs,  # self.inputs is from AgentConfig
            )
            # Add extracted records to self._records
            extracted_records = extracted.pop("records", [])
            if extracted_records:
                self._records.extend(extracted_records)
                logger.debug(f"Agent {self.agent_name} extracted {len(extracted_records)} records via mappings.")

            # Add other extracted data to self._data
            found_keys = []
            for key, value in extracted.items():
                if value is not None and value not in ([], {}):  # Ensure value is meaningful
                    self._data.add(key, value)
                    found_keys.append(key)
            if found_keys:
                logger.debug(f"Agent {self.agent_name} extracted data for keys {found_keys} from {source} via mappings.")
        else:
            logger.debug(f"Agent {self.agent_name} has no input mappings defined; skipping data extraction for generic message type {type(message)}.")

        # Add relevant message content to the conversation history (_model_context).
        # This logic determines what parts of messages are considered for context.
        # Only add messages that are relevant to this agent's role and responsibilities.
        if isinstance(message, AgentTrace):
            # Only add traces to context if they are directly relevant to this agent
            # Avoid contaminating agent context with irrelevant traces from other agents
            should_add_to_context = (
                source == self.agent_name or  # Messages from this agent itself
                source == "manager" or        # Direct user/manager messages
                (hasattr(message, "agent_info") and
                 getattr(message.agent_info, "role", None) in ["USER", "MANAGER"])  # User-facing roles
            )

            if should_add_to_context:
                content_to_add = getattr(message, "contents", None)  # Prefer 'contents' if available
                if content_to_add is None and hasattr(message, "outputs"):  # Fallback to stringified outputs
                    content_to_add = str(message.outputs) if message.outputs else None
                if content_to_add:
                    await self._model_context.add_message(
                        AssistantMessage(content=str(content_to_add), source=source or self.agent_name),
                    )
        elif isinstance(message, ManagerMessage) and message.content:
            content_str = str(message.content)
            if not content_str.startswith(COMMAND_SYMBOL):  # Avoid adding command-like messages to history
                await self._model_context.add_message(UserMessage(content=content_str, source=source or "manager"))
        # elif isinstance(message, AgentOutput) and hasattr(message, 'outputs'):
        # output_str = str(message.outputs)
        # if output_str: # Avoid empty strings
        # await self._model_context.add_message(AssistantMessage(content=output_str, source=source or self.agent_name))
        else:
            logger.debug(f"Agent {self.agent_name} did not add message of type {type(message)} to context history from source '{source}'.")

        # Handle AgentAnnouncement messages
        if isinstance(message, AgentAnnouncement):
            # Check if this is from a host agent (initial announcement)
            if (message.announcement_type == "initial" and 
                message.agent_config.role in ["HOST", "ORCHESTRATOR"]):
                # Respond to host announcement
                await self.send_announcement(
                    announcement_type="response",
                    status="active",
                    responding_to=message.agent_config.agent_id,
                    topic_id=ctx.topic_id
                )
                logger.debug(f"Agent {self.agent_name} responded to host announcement from {message.agent_config.agent_id}")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        **kwargs: Any,
    ) -> None:
        """Backwards compatibility wrapper for _listen."""
        # Create a MessageContext
        ctx = MessageContext(
            sender=AgentId(type=source or "unknown", key=source or "unknown"),
            topic_id=DefaultTopicId(type="default"),
            is_rpc=False,
            cancellation_token=cancellation_token or CancellationToken(),
        )

        await self.handle_groupchat_message(message, ctx)

    # @message_handler
    # async def handle_control_message(
    #     self,
    #     message: OOBMessages,
    #     ctx: MessageContext,
    # ) -> Union[OOBMessages, None]:
    #     """Handles Out-Of-Band (OOB) control messages.

    #     This method is intended for processing messages that are not part of the
    #     standard request-response flow of agent processing, but rather control
    #     signals or special events. Examples include requests to reset the agent,
    #     status queries, or other administrative commands.

    #     Subclasses can override this method to react to specific types of OOB
    #     messages relevant to their functionality. The base implementation is a
    #     no-op for most messages but serves as a hook for future extensions or
    #     specific agent needs.

    #     Args:
    #         message: The Out-Of-Band (`OOBMessages`) message received.
    #         ctx: The message context containing sender info and cancellation token.

    #     Returns:
    #         Union[OOBMessages, None]: An optional `OOBMessage` as a direct response to
    #         the handled event, or `None` if no direct response is generated.

    #     """
    #     logger.debug(f"Agent {self.agent_name} received OOB event: {type(message).__name__}. Default handler is a no-op.")
    #     # Example of how a subclass might handle a specific OOB message:
    #     # if isinstance(message, YourSpecificOOBMessage):
    #     #     # Perform some action
    #     #     await self.on_reset(cancellation_token) # e.g., reset on a specific signal
    #     #     return OOBResponse(...) # Optionally return a response
    #     return None  # Default: no direct response

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        **kwargs: Any,
    ) -> Union[OOBMessages, None]:
        """Backwards compatibility wrapper for _handle_events."""
        # Create a MessageContext
        ctx = MessageContext(
            sender=AgentId(type=source or "unknown", key=source or "unknown"),
            topic_id=DefaultTopicId(type="default"),
            is_rpc=True,  # OOB messages are typically RPC-style
            cancellation_token=cancellation_token or CancellationToken(),
        )

        return await self.handle_control_message(message, ctx)

    @message_handler
    async def handle_step_request(
        self,
        message: StepRequest,
        ctx: MessageContext,
    ) -> Union[AgentTrace, None]:
        """Handle StepRequest messages specifically.
        
        StepRequest is a subclass of AgentInput, so we delegate to handle_invocation.
        This handler ensures StepRequests directed to this agent are properly handled.
        """
        # Only handle if the role matches this agent's role
        if message.role == self.role:
            return await self.handle_invocation(message, ctx)
        else:
            # Not for this agent, ignore
            logger.debug(f"Agent {self.agent_name} ignoring StepRequest for role {message.role}")
            return None

    @message_handler
    async def handle_heartbeat(
        self,
        message: HeartBeat,
        ctx: MessageContext,
    ) -> None:
        """Handle heartbeat messages.
        
        Puts the heartbeat signal into the agent's internal heartbeat queue.
        """
        try:
            self._heartbeat.put_nowait(message.go_next)
        except asyncio.QueueFull:
            # If the agent isn't processing heartbeats quickly enough.
            logger.debug(f"Heartbeat queue full for agent {self.agent_name}. Agent may be busy or stuck.")

    # --- Helper Methods ---

    async def _check_heartbeat(self, timeout: float = 240.0) -> bool:
        """Waits for a heartbeat signal on the internal `_heartbeat` queue.

        This method can be used by orchestrators or monitoring systems to check
        if an agent is responsive or ready for tasks. A heartbeat is typically
        sent by the agent itself or a managing component to indicate liveness.
        The agent must have a mechanism to put `True` or some value onto its
        `_heartbeat` queue for this method to succeed.

        Args:
            timeout: The maximum number of seconds to wait for a heartbeat signal.
                Defaults to 240 seconds.

        Returns:
            bool: `True` if a heartbeat signal was received within the timeout period,
            `False` otherwise (e.g., on timeout or if an error occurs).

        """
        try:
            logger.debug(f"Agent {self.agent_name} waiting for heartbeat (timeout: {timeout}s)...")
            await asyncio.wait_for(self._heartbeat.get(), timeout=timeout)
            logger.debug(f"Agent {self.agent_name} received heartbeat.")
            self._heartbeat.task_done()  # Mark the item from queue as processed
            return True
        except TimeoutError:  # More specific exception for timeout
            logger.warning(f"Agent {self.agent_name} timed out waiting for heartbeat after {timeout}s.")
            return False
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error checking heartbeat: {e!s}")
            return False

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """Augments an incoming `AgentInput` message with the agent's internal state.

        This crucial helper method prepares the final input that `_process` will receive.
        It merges several sources of information:
        1.  **Default Parameters**: Parameters defined in the agent's configuration
            (`self.parameters`) are used as a base.
        2.  **Message Parameters**: Parameters from the incoming `inputs.parameters`
            override any defaults.
        3.  **Resolved Input Mappings**: Data from `self._data` (which is populated
            by `_listen` based on `self.inputs` mappings) is resolved and added to
            `updated_inputs.inputs`. Incoming `inputs.inputs` can override these.
        4.  **Conversation History**: Messages from `self._model_context` are prepended
            to `updated_inputs.context`.
        5.  **Records**: If `updated_inputs.records` is empty, the most recent record(s)
            from `self._records` are used.

        Args:
            inputs: The original `AgentInput` message.

        Returns:
            AgentInput: A new `AgentInput` instance with the agent's current state
            (parameters, resolved data, context, records) merged into it.

        Raises:
            ProcessingError: If an error occurs during the resolution of input mappings
                from `self._data`.

        """
        updated_inputs = inputs.model_copy(deep=True)

        # 1. Merge agent's default parameters, letting message parameters override.
        if updated_inputs.parameters is None:
            updated_inputs.parameters = {}
        # Ensure self.parameters (from AgentConfig) is not None before merging
        merged_params = {**(self.parameters or {}), **updated_inputs.parameters}
        updated_inputs.parameters = merged_params

        # 2. Resolve input mappings using data stored in self._data.
        if updated_inputs.inputs is None:
            updated_inputs.inputs = {}
        if self.inputs:  # self.inputs is the mapping configuration from AgentConfig
            try:
                extracted_data = {}
                for key in self.inputs.keys():  # Iterate over configured input mapping keys
                    # Retrieve data from self._data; KeyValueCollector stores values in lists
                    data_values = self._data.get(key, [])
                    # Filter out empty/None values from the list
                    meaningful_values = [v for v in data_values if v is not None and v not in ([], {})]
                    if meaningful_values:
                        # If only one meaningful value, unwrap it from the list, else keep as list
                        extracted_data[key] = meaningful_values[0] if len(meaningful_values) == 1 else meaningful_values

                # Merge resolved mappings, letting original message inputs override
                merged_inputs_dict = {**extracted_data, **updated_inputs.inputs}
                updated_inputs.inputs = merged_inputs_dict
            except Exception as e:
                logger.error(f"Agent {self.agent_name}: Error resolving input mappings: {e!s}")
                raise ProcessingError(f"Error resolving input mappings for agent {self.agent_id}: {e!s}") from e

        # 3. Prepend conversation history from agent's context.
        if updated_inputs.context is None:
            updated_inputs.context = []
        try:
            history = await self._model_context.get_messages()
            updated_inputs.context = history + updated_inputs.context  # Prepend history
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error retrieving model context: {e!s}")
            # Decide handling: continue without history or raise? For now, log and continue.

        # 4. Ensure records list exists. Use the last saved one if input records are empty.
        if not updated_inputs.records and self._records:
            updated_inputs.records = [self._records[-1]]  # Use only the most recent record as default

        logger.debug(
            f"Agent {self.agent_id}: Added state to input. "
            f"Final input keys: {list(updated_inputs.inputs.keys()) if updated_inputs.inputs else []}, "
            f"Context length: {len(updated_inputs.context)}, "
            f"Records count: {len(updated_inputs.records)}.",
        )

        return updated_inputs

    def get_tool_definitions(self) -> list["AgentToolDefinition"]:
        """Generate structured tool definitions for this agent.
        
        This method extracts tool definitions from methods decorated with
        @tool or @MCPRoute decorators. It enables agents to automatically
        expose their capabilities as structured tool definitions that can
        be used by LLMs and MCP servers.
        
        Returns:
            List of AgentToolDefinition objects representing this agent's tools.
        """
        from buttermilk._core.mcp_decorators import extract_tool_definitions
        return extract_tool_definitions(self)
    
    def get_all_tool_definitions(self) -> list["AgentToolDefinition"]:
        """Get all tool definitions from both decorated methods and configured tools.
        
        This method combines:
        1. Tool definitions from @tool decorated methods
        2. Tool definitions from configured tools (if the agent has them)
        
        Returns:
            list[AgentToolDefinition]: All available tool definitions
        """
        from buttermilk._core.tool_definition import AgentToolDefinition
        
        # Start with decorated method tools
        tool_defs = self.get_tool_definitions()
        
        # Add configured tools if this agent has them
        if hasattr(self, '_tools_list') and self._tools_list:
            from autogen_core.tools import FunctionTool
            
            for tool in self._tools_list:
                if isinstance(tool, FunctionTool):
                    # Convert FunctionTool to AgentToolDefinition
                    # FunctionTool has name, description, and schema properties
                    schema = tool.schema
                    if isinstance(schema, dict) and 'function' in schema:
                        func_info = schema['function']
                        tool_def = AgentToolDefinition(
                            name=func_info.get('name', tool.name),
                            description=func_info.get('description', tool.description),
                            input_schema=func_info.get('parameters', {"type": "object", "properties": {}}),
                            output_schema={"type": "string"}  # Default output schema
                        )
                        tool_defs.append(tool_def)
                elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                    # Generic Tool with basic properties
                    tool_def = AgentToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        input_schema={"type": "object", "properties": {}},  # Default schema
                        output_schema={"type": "string"}
                    )
                    tool_defs.append(tool_def)
        
        return tool_defs

    async def handle_unified_request(self, request: "UnifiedRequest", **kwargs: Any) -> Any:
        """Handle a UnifiedRequest by routing to the appropriate tool or _process method.
        
        This method provides a unified interface for handling both tool-specific
        requests and general agent requests, supporting the new structured
        tool invocation system.
        
        Args:
            request: UnifiedRequest containing target, inputs, context, and metadata
            **kwargs: Additional keyword arguments passed to handlers
            
        Returns:
            The result of the tool or process execution
            
        Raises:
            ValueError: If the requested tool is not found
            ProcessingError: If execution fails
        """
        from buttermilk._core.tool_definition import UnifiedRequest

        if request.tool_name:
            # Route to specific tool
            tool_name = request.tool_name

            # First, try direct method name match
            if hasattr(self, tool_name) and callable(getattr(self, tool_name)):
                method = getattr(self, tool_name)
                if hasattr(method, "_tool_metadata") or hasattr(method, "_mcp_route"):
                    logger.debug(
                        f"Agent {self.agent_name} executing tool {tool_name} "
                        f"with inputs: {request.inputs}"
                    )

                    # Execute the tool method
                    if asyncio.iscoroutinefunction(method):
                        result = await method(**request.inputs)
                    else:
                        result = method(**request.inputs)

                    return result

            # If direct match fails, search for methods with matching tool metadata
            for attr_name in dir(self):
                if attr_name.startswith("_"):
                    continue

                attr = getattr(self, attr_name)
                if callable(attr):
                    # Check if it has tool metadata with matching name
                    if hasattr(attr, "_tool_metadata"):
                        if attr._tool_metadata.get("name") == tool_name:
                            logger.debug(
                                f"Agent {self.agent_name} executing tool {tool_name} "
                                f"(method: {attr_name}) with inputs: {request.inputs}"
                            )

                            # Execute the tool method
                            if asyncio.iscoroutinefunction(attr):
                                result = await attr(**request.inputs)
                            else:
                                result = attr(**request.inputs)

                            return result

            # Tool not found
            raise ValueError(
                f"Tool {tool_name} not found on agent {self.agent_name}"
            )
        else:
            # No specific tool - route to general _process method
            # Convert UnifiedRequest to AgentInput for backward compatibility
            agent_input = AgentInput(
                inputs=request.inputs,
                context=request.context.get("messages", []) if request.context else [],
                parameters=request.metadata,
                records=request.context.get("records", []) if request.context else []
            )

            # Call the standard process method
            result = await self._process(message=agent_input, **kwargs)

            # Extract outputs from AgentOutput if needed
            if hasattr(result, "outputs"):
                return result.outputs
            return result
