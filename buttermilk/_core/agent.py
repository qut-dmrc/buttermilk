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
import warnings
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

import weave  # For tracing - core dependency
from weave.trace.weave_client import Call, WeaveObject

if TYPE_CHECKING:
    from autogen_core import AgentRuntime

    from buttermilk._core.tool_definition import AgentToolDefinition

# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
from autogen_core import (
    AgentId,
    AgentMetadata,
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, UserMessage
from autogen_core.tools import Tool

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core.config import AgentConfig

# Buttermilk core imports
from buttermilk._core.constants import COMMAND_SYMBOL  # Constant for command messages
from buttermilk._core.contract import (
    AgentAnnouncement,
    AgentInput,
    AgentOutput,  # Standard input message structure
    AgentTrace,
    ConductorRequest,
    ErrorEvent,
    HeartBeat,
    ManagerMessage,  # Messages from the user
    StepRequest,  # Request to execute a specific step
    TaskProcessingComplete,
    TaskProcessingStarted,
    ToolOutput,
)
from buttermilk._core.exceptions import ProcessingError  # Custom exceptions
from buttermilk._core.log import logger  # Buttermilk logger instance
from buttermilk._core.message_data import extract_message_data
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.types import Record  # Data record structure
from buttermilk.utils.templating import KeyValueCollector  # Utility for managing state data

# --- Base Agent Class ---


class Agent(RoutedAgent):  # noqa: PLR0904
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
    def inputs(self) -> dict[str, str] | None:
        return self._config.inputs

    @property
    def session_id(self) -> str:
        """Get session_id from config if available."""
        return getattr(self._config, "session_id", "")

    def __init__(self, topic_id: TopicId | None = None, **data: Any) -> None:
        """Initialize the Agent with configuration data and setup RoutedAgent."""
        # Set groupchat topic ID, defaulting to a standard topic if not provided
        self._topic_id: TopicId = topic_id or DefaultTopicId(type="default")

        # Create AgentConfig from the data
        self._config = AgentConfig(**data)

        # Initialize RoutedAgent with description
        RoutedAgent.__init__(self, description=self._config.description)

        # Initialize private attributes
        self._records = []
        self._model_context = UnboundedChatCompletionContext()
        self._data = KeyValueCollector()
        self._heartbeat = asyncio.Queue(maxsize=1)
        self._announced = False

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

    async def close(self) -> None:
        """Called when the runtime is closed"""
        await self.cleanup()

    # --- Internal State ---
    _records: list[Record]
    _model_context: ChatCompletionContext
    _data: KeyValueCollector
    _heartbeat: asyncio.Queue[bool | None] | asyncio.Queue
    _announcement_callback: Callable[[Any], Awaitable[None]] | None
    _announced: bool = False  # Track if agent has announced itself

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

        We don't clean internal state because the whole Agent will be destroyed anyway.

        """
        logger.debug(f"Agent {self.agent_name}: No persistent resourcces to cleanup.")

    # --- Announcement Methods ---

    async def _publish(
        self,
        message: Any,
        topic_id: TopicId | None = None,
        *,
        highlight: bool = False,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Publish a message to the group chat or a specific topic.

        Args:
            message: The message to publish.
            highlight: Whether to highlight the message in logs.
            topic_id: Optional specific topic to publish to. Defaults to self._topic_id.
            cancellation_token: Optional cancellation token to cancel the operation.

        """
        # Use provided topic_id or fall back to the agent's default topic
        target_topic = topic_id or self._topic_id
        await super().publish_message(message, topic_id=target_topic, cancellation_token=cancellation_token)

        if not highlight and isinstance(message, (AgentTrace, AgentOutput)):
            highlight = True  # Highlight traces and outputs by default
        if highlight:
            logger.highlight(
                f"Agent {self.agent_name} ({self.agent_id}) sent {type(message).__name__} to {topic_id}.",
            )
        else:
            logger.debug(
                f"Agent {self.agent_name} ({self.agent_id}) sent {type(message).__name__} to {topic_id}.",
            )

    def get_available_tools(self) -> list[Tool]:
        """Get list of tools this agent can respond to.
        This is overridden in the LLMAgent class to load tools from the config.

        Returns:
            list[Tool]: List of tools.

        """
        return []

    # --- Core Execution Logic ---

    async def invoke(
        self,
        message: AgentInput | StepRequest,
    ) -> AgentTrace | None:
        """Prepare input, calls the agent's core logic, and handles callbacks.

        It performs the following steps:
        1. Augments the incoming `message` with the agent's internal state using
           `_add_state_to_input`.
        2. Notifies listeners (via `public_callback`) that task processing has started.
        3. Invokes the agent's core logic via `self._process`.
        4. Handles any errors during execution, creating an `ErrorEvent` if necessary.
        5. Notifies listeners that task processing has completed (or failed).
        6. Constructs an `AgentTrace` object containing details of the execution,
           including inputs, outputs (if any), and configuration.
        7. Publishes the `AgentTrace` to listeners.


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
            None: If the agent does not run.

        Raises:
            ProcessingError: If `_add_state_to_input` fails. (Errors from `__call__`
                are caught and reported in the `AgentTrace` and `TaskProcessingComplete` event).

        """
        await self._publish(TaskProcessingStarted(agent_id=self.agent_id, role=self.role, task_index=0), topic_id=self._topic_id)

        # --- Prepare the input state for processing ---
        try:
            final_input = await self._add_state_to_input(message)
        except Exception as e:
            logger.error(f"Error preparing data for Agent {self.agent_id}: {e}")
            # Create an ErrorEvent to capture the error
            err_result = ErrorEvent(source=self.agent_id, content=f"Invoke error: {e}")
            await self._publish(
                TaskProcessingComplete(agent_id=self.agent_id, role=self.role, is_error=True, error=[err_result]),
                topic_id=self._topic_id,
            )
            return None

        trace = await self.trace_and_execute(message=final_input)

        # If the agent didn't run, just exit.
        if not trace:
            return None

        # Publish the trace.
        # Importantly, StepRequests might be sent privately or to a subset of agents. But we
        # want to publish the trace to the general topic so it can be consumed by any interested parties.
        # So we publish to self._topic_id, not ctx.topic_id.
        await self._publish(trace, topic_id=self._topic_id)

        # Publish status update: Task Complete (including error if error)
        await self._publish(
            TaskProcessingComplete(agent_id=self.agent_id, role=self.role, task_index=0, more_tasks_remain=False, is_error=trace.is_error),
            topic_id=self._topic_id,
        )

        logger.debug(f"Agent {self.agent_name} finished task {message}.")

        return trace

    async def trace_and_execute(
        self,
        message: AgentInput,
    ) -> AgentTrace | None:
        """Primary execution entry point for the agent, handling a single `AgentInput`.

        This method orchestrates the core processing logic of the agent. It is
        responsible for:
        1. Setting up tracing for the operation using Weave.
        2. Calling the abstract `_process` method, which must be implemented by
           subclasses to perform the agent's specific task.
        3. Ensuring the trace call is properly finished, regardless of success or failure.

        Args:
            message: The `AgentInput` message containing the data and parameters
                for the agent to process. This input may have already been augmented
                with agent state by `_add_state_to_input`.

        Returns:
            - AgentTrace: An object containing the results of the agent's processing, with
            complete tracing information.

            - None: If the agent does not produce output.

        Raises:
            None.

        """
        result: AgentOutput | None = None  # Ensure result is defined for finally block
        tracing_link: str | None = None

        # --- Tracing ---
        trace_params = {
            "name": self.agent_name,
            "model": self._cfg.parameters.get("model"),
            **message.parameters,
            **message.metadata,
            **self.parameters,
        }

        op = weave.op(self.invoke, call_display_name=self.agent_name)
        parent_call = await get_parent_call(message)

        child_call = bm.weave.create_call(
            op,
            inputs=message.model_dump(mode="json"),
            parent=parent_call,
            display_name=self.agent_name,
            attributes=trace_params,
        )

        if parent_call:
            parent_call._children.append(child_call)  # Nest this call for tracing

        try:
            result = await self._process(message=message)
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error during invoke: {e}")
            # Create an ErrorEvent to capture the error
            err_result = ErrorEvent(source=self.agent_id, content=f"Invoke error: {e}")
            result = AgentOutput(agent_id=self.agent_id, outputs=None, call_id=child_call.id, error=[err_result])
        finally:
            # Mark the child call as complete, regardless of success or failure.
            # Output is passed to bm.weave.finish_call if result is not None
            bm.weave.finish_call(child_call, output=result or None, op=op)

            # TODO: try to force and wait for upload to weave here, so that we can get the trace link

        # --- Turn the result into AgentTrace for long-term storage ---
        if hasattr(child_call, "ui_url"):
            tracing_link = child_call.ui_url

            # Handle case where _process returns None (e.g., UI agents that don't produce output)
            if result is None:
                return None

        # Create AgentTrace from the result
        trace = AgentTrace(
            **result.model_dump() if result else {},
            call_id=child_call.id,
            agent_info=self._cfg,
            tracing_link=tracing_link,
            parent_call_id=parent_call.id if parent_call else None,
        )

        return trace

    @abstractmethod
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput | None:
        """Abstract method for the agent's core processing logic.

        Subclasses **MUST** implement this method to define their specific behavior.
        This method receives an `AgentInput` (which may have been augmented with
        the agent's internal state by `_add_state_to_input`) and should perform
        the agent's primary task.

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

    # --- Message Handlers ---
    @message_handler
    async def handle_conductor_request(
        self,
        message: "ConductorRequest",
        ctx: MessageContext,
    ) -> None:
        """Handle ConductorRequest messages by sending agent announcements.

        When a conductor requests information about this agent, respond with
        an announcement containing the agent's capabilities and configuration.

        Args:
            message: The ConductorRequest message.
            ctx: Message context containing sender and topic information.

        """
        # Get ALL tool definitions (decorated methods + configured tools)
        tool_definitions = self.get_tool_definitions()

        announcement = AgentAnnouncement(
            content=f"Agent {self.agent_name} active and available",
            agent_config=self._cfg,
            available_tools=[tool.name for tool in self.get_available_tools()],
            tool_definitions=tool_definitions,
            status="active",
            announcement_type="initial",
            responding_to=message.message_id if hasattr(message, "message_id") else None,
            source=self.agent_id,
        )

        await self._publish(
            announcement,
            topic_id=self._topic_id,
            highlight=True,
        )

        # Mark as announced
        self._announced = True

    @message_handler
    async def handle_request(
        self,
        message: AgentInput | StepRequest,
        ctx: MessageContext,
    ) -> AgentTrace | None:
        """Handle an invocation message, preparing input and calling the agent's core logic.

        This method is designed to be used by host agents to invoke
        the agent for a single processing step. It augments the incoming `message`
        with the agent's internal state, notifies listeners of task processing,
        and invokes the agent's core logic.

        Args:
            message: The initial `AgentInput` message for the agent.
            ctx: Message context containing sender and topic information.

        Returns:
            AgentTrace: An object detailing the agent's execution for this invocation.
            None: If the agent does not run.

        Raises:
            ProcessingError: If `_add_state_to_input` fails. (Errors from the actual call
                are caught and reported in the `AgentTrace` and `TaskProcessingComplete` event).

        """
        if isinstance(message, StepRequest) and message.role != self.role:
            # Only handle if the role matches this agent's role - create a "skipped" trace
            logger.warning(f"Agent {self.agent_name} skipped StepRequest due to role mismatch: requested {message.role}, agent is {self.role}")
            return None

        return await self.invoke(message=message)

    @message_handler
    async def handle_agent_output(
        self,
        message: AgentOutput | AgentTrace,
        ctx: MessageContext,
    ) -> None:
        """Handle AgentOutput or AgentTrace messages, extracting Records and data based on our configured input mappings.

        AgentTrace is a subclass of AgentOutput with additional run and tracing information.

        Args:
            message: The AgentTrace or AgentOutput message to process.
            ctx: Message context containing sender and topic information.

        """
        source = str(ctx.sender).split("/", maxsplit=1)[0] if ctx.sender else "unknown"

        # Extract data based on input mappings
        if self.inputs:  # Only extract if input mappings are defined
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=self.inputs,
            )
            # Add extracted records to self._records
            extracted_records = extracted.pop("records", [])
            for rec in extracted_records:
                try:
                    self._records.append(Record.model_validate(rec))
                    logger.debug(f"Agent {self.agent_name} extracted {len(extracted_records)} records via mappings.")
                except Exception as e:
                    logger.error(f"Agent {self.agent_name} failed to validate record {rec}: {e}")

            # Add other extracted data to self._data
            found_keys = []
            for key, value in extracted.items():
                if value is not None and value not in ([], {}):  # Ensure value is meaningful
                    self._data.add(key, value)
                    found_keys.append(key)
            if found_keys:
                logger.debug(f"Agent {self.agent_name} extracted data for keys {found_keys} from {source} via mappings.")
        else:
            logger.debug(f"Agent {self.agent_name} has no input mappings defined; skipping data extraction.")

        # Add relevant message content to the conversation history (_model_context).
        if content_to_add := getattr(message, "contents", None):
            await self._model_context.add_message(
                AssistantMessage(content=str(content_to_add), source=source or self.agent_name),
            )

    @message_handler
    async def handle_manager_message(
        self,
        message: ManagerMessage,
        ctx: MessageContext,
    ) -> None:
        """Handle ManagerMessage messages, adding non-command content to model context.

        Args:
            message: The ManagerMessage to process.
            ctx: Message context containing sender and topic information.

        """
        source = str(ctx.sender).split("/", maxsplit=1)[0] if ctx.sender else "manager"

        # Extract data based on input mappings if defined
        if self.inputs:
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=self.inputs,
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

        # Add to model context if not a command
        if message.content:
            content_str = str(message.content)
            if not content_str.startswith(COMMAND_SYMBOL):  # Avoid adding command-like messages to history
                await self._model_context.add_message(UserMessage(content=content_str, source=source))

    @message_handler
    async def handle_tool_output(
        self,
        message: ToolOutput,
        ctx: MessageContext,
    ) -> None:
        """Handle ToolOutput messages, extracting data based on input mappings.

        Args:
            message: The ToolOutput message to process.
            ctx: Message context containing sender and topic information.

        """
        source = str(ctx.sender).split("/", maxsplit=1)[0] if ctx.sender else "unknown"

        # Extract data based on input mappings
        if self.inputs:  # Only extract if input mappings are defined
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=self.inputs,
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
            logger.debug(f"Agent {self.agent_name} has no input mappings defined; skipping data extraction for ToolOutput.")

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
                    # Retrieve data from self._data; note that KeyValueCollector stores values in lists
                    data_values = self._data.get(key, [])
                    extracted_data[key] = data_values

                # Merge resolved mappings, letting original message inputs override
                merged_inputs_dict = {**extracted_data, **updated_inputs.inputs}
                updated_inputs.inputs = merged_inputs_dict
            except Exception as e:
                raise ProcessingError(f"Error resolving input mappings for agent {self.agent_id}: {e!s}") from e

        # 4. Prepend conversation history from agent's context.
        if updated_inputs.context is None:
            updated_inputs.context = []
        try:
            history = await self._model_context.get_messages()
            updated_inputs.context = history + updated_inputs.context  # Prepend history
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error retrieving model context: {e!s}")
            # Decide handling: continue without history or raise? For now, log and continue.

        # 5. Ensure records list exists. Use the last saved one if input records are empty.
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

        This method creates a tool definition for the agent's primary
        processing capability, allowing it to be invoked as a tool
        in the Autogen groupchat.

        Returns:
            List of AgentToolDefinition objects representing this agent's tools.

        """
        from buttermilk._core.tool_definition import AgentToolDefinition

        # Create a tool definition for the agent's main processing capability
        tool_def = AgentToolDefinition(
            name=f"{self.agent_name}_call",
            description=self.description or f"Process requests using {self.agent_name} agent",
            input_schema={
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "object",
                        "description": "Input data for the agent to process",
                    },
                    "context": {
                        "type": "object",
                        "description": "Shared context across agents",
                        "default": {},
                    },
                },
                "required": ["inputs"],
            },
            output_schema={
                "type": "object",
                "description": "Agent processing results",
            },
        )

        return [tool_def]


async def get_parent_call(
    message: AgentInput | None = None,
) -> Call | WeaveObject | None:

    parent_call: Call | WeaveObject | None = None

    if message and message.parent_call_id:

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
            cooldown_seconds=0,
        )

        try:
            parent_call = await retry_wrapper._execute_with_retry(
                get_weave_call_with_retry,
                message.parent_call_id,
            )
        except Exception as e:  # Broad exception for Weave call retrieval
            logger.error(f"Could not retrieve parent call ID {message.parent_call_id} after retries. Error: {e}")
            parent_call = weave.get_current_call()  # Fallback to current call if specified parent not found
    else:
        parent_call = weave.get_current_call()
    return parent_call
