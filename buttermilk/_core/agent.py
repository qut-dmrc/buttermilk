"""Core Agent base class for the Buttermilk framework.

Agents are responsible for performing specific tasks as part of a larger
data processing flow. `AgentConfig` provides the base configuration,
and `Agent` provides the execution logic and state management.
"""

import asyncio
import warnings
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from opentelemetry import trace

from buttermilk import bm, logger
from buttermilk._core.config import AgentConfig

# Buttermilk core imports
from buttermilk._core.constants import COMMAND_SYMBOL  # Constant for command messages
from buttermilk._core.contract import (
    AgentAnnouncement,
    AgentInput,
    AgentOutput,  # Standard input message structure
    ConductorRequest,
    ErrorEvent,
    ExecutionTrace,
    StepRequest,  # Request to execute a specific step
    TaskProcessingComplete,
    TaskProcessingStarted,
    UserResponseMessage,  # Messages from the user
)
from buttermilk._core.exceptions import FatalError, ProcessingError  # Custom exceptions
from buttermilk._core.message_data import extract_message_data
from buttermilk._core.messages import AssistantMessage, UserMessage
from buttermilk._core.runtime_types import (
    AgentIdentity,
    ChatHistory,
    DefaultTopicId,
    MessageContext,
    TopicId,
    _build_handler_registry,
    dispatch_message,
    message_handler,
)
from buttermilk._core.tool_types import CancellationToken, Tool
from buttermilk._core.types import BaseRecord  # Data record structure
from buttermilk.utils.templating import (
    KeyValueCollector,
)  # Utility for managing state data


# Utility functions for agent tracing
def get_agent_type_for_trace(agent: Any) -> str:
    """Get simplified agent type for tracing.

    Returns lowercase agent class name (e.g., 'judge', 'fetchagent').
    This follows OTEL semantic conventions for component types.

    Args:
        agent: Agent instance

    Returns:
        Lowercase agent class name

    Examples:
        >>> from buttermilk.agents.judge import Judge
        >>> judge = Judge(...)
        >>> get_agent_type_for_trace(judge)
        'judge'
    """
    return agent.__class__.__name__.lower()


def create_agent_trace_info(
    agent: Any,
    template_hash: str | None = None,
    hash_collector: Any | None = None,
) -> dict[str, Any]:
    """Create comprehensive agent info for ExecutionTrace.

    Captures agent identity and static config for reproducibility:
    - Agent identity (type, name, role)
    - Template name and hash
    - Model configuration
    - Optional hash collection for systematic tracing

    Note: Template variables (criteria, instructions, etc.) go in trace.inputs,
    not in agent_info. This function captures only static agent configuration.

    Args:
        agent: Agent instance
        template_hash: Optional pre-computed template hash.
                      If not provided, will use agent.parameters.get("template_hash")
        hash_collector: Optional HashCollector with all hashes

    Returns:
        Dictionary of agent trace information including hashes

    Example:
        >>> trace_info = create_agent_trace_info(judge_agent, template_hash="abc123")
        >>> trace_info["agent_type"]  # "judge"
        >>> trace_info["agent_class"]  # "buttermilk.agents.judge.Judge"
        >>> trace_info["template"]  # "judge.jinja2"
        >>> trace_info["model"]  # "gemini-2.0-flash"
    """
    agent_info = {
        # Identity - simple and full
        "agent_type": get_agent_type_for_trace(agent),  # Simple: "judge"
        "agent_class": f"{agent.__class__.__module__}.{agent.__class__.__name__}",  # Full
        "agent_name": agent.agent_name,
        "agent_role": agent.role,
        # Static config for reproducibility (template vars go in trace.inputs)
        "template": agent.parameters.get("template"),
        "template_hash": template_hash or agent.parameters.get("template_hash"),
        "model": agent.parameters.get("model"),
        # Additional metadata
        "description": agent.description,
    }

    # Merge hash collector attributes if provided
    if hash_collector:
        agent_info.update(hash_collector.to_span_attributes())

    # Remove None values to keep traces clean
    return {k: v for k, v in agent_info.items() if v is not None}


# --- Base Agent Class ---


class Agent:
    """Base class for all Buttermilk agents.

    Subclasses implement `_process` for their core logic. The orchestrator
    injects a `_publish_fn` callback for message dispatch, and calls
    `dispatch()` to route incoming messages to the correct handler.
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
    def record_mapping(self) -> str | None:
        """Get the record JMESPath mapping from config."""
        return self._config.record

    @property
    def context_mapping(self) -> str | None:
        """Get the context JMESPath mapping from config."""
        return self._config.context

    @property
    def session_id(self) -> str:
        """Get session_id from config if available."""
        return getattr(self._config, "session_id", "")

    @property
    def required_inputs(self) -> list[str] | None:
        """Get the list of required input keys from config.

        Returns None if not set (no filtering), empty list if explicitly
        set to filter all inputs, or a list of keys to whitelist.
        """
        return self._config.required

    def get_effective_bm(self) -> Any:
        """Get the effective BM instance (session-scoped if available, otherwise global singleton).

        This method provides agents with transparent access to BM functionality while
        supporting session-level observability isolation. When agents are created through
        the orchestration framework (FlowRunner -> Orchestrator -> Agent), they automatically
        receive session-scoped BM instances for proper observability separation.

        Returns:
            BM instance to use for operations. Returns session-scoped BM if one was
            injected during agent creation, otherwise falls back to the global singleton.

        Example:
            >>> # In an agent's _process method:
            >>> bm = self.get_effective_bm()
            >>> storage = bm.get_storage(self.data["input_source"])
            >>> # Storage access is now session-isolated for multi-session environments

        Note:
            Using `self.get_effective_bm()` provides session isolation benefits in
            API and orchestrated environments.
        """
        # Check if BM was injected via config
        if hasattr(self._config, "bm") and self._config.bm is not None:
            return self._config.bm
        # Fall back to global singleton
        from buttermilk._core.dmrc import get_bm

        return get_bm()

    def __init__(
        self,
        topic_id: TopicId | str | None = None,
        publish_fn: Callable[..., Awaitable[None]] | None = None,
        **data: Any,
    ) -> None:
        self._topic_id: TopicId = TopicId(topic_id) if topic_id else DefaultTopicId(type="default")
        self._config = AgentConfig(**data)
        self._publish_fn = publish_fn

        self._model_context = ChatHistory()
        self._data = KeyValueCollector()
        self._heartbeat = asyncio.Queue(maxsize=1)
        self._announced = False
        self._tools = self._get_available_tools()
        self._handler_registry: dict[type, str] | None = None

    @property
    def identity(self) -> AgentIdentity:
        """Native agent identity."""
        return AgentIdentity(
            key=self.agent_id,
            type=self.agent_name,
            description=self.description,
        )

    # --- Message dispatch ---

    def _get_handler_registry(self) -> dict[type, str]:
        if self._handler_registry is None:
            self._handler_registry = _build_handler_registry(type(self))
        return self._handler_registry

    async def dispatch(self, message: Any, ctx: MessageContext) -> Any:
        """Dispatch an incoming message to the appropriate @message_handler."""
        return await dispatch_message(self, message, ctx)

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

    def _get_available_tools(self) -> list[Tool]:
        """Get list of tools this agent can respond to.

        Returns:
            list[Tool]: List of tools.

        """

        import hydra
        from omegaconf import OmegaConf

        from buttermilk.utils._tools import create_tool_functions

        tools = {}
        for tool_name, tool in self._config.tools.items():
            if OmegaConf.is_config(tool):
                # If the tool configuration is an OmegaConf object, instantiate it
                tool_cfg = hydra.utils.instantiate(tool)
                tools[tool_name] = tool_cfg
            else:
                tools[tool_name] = tool

        return create_tool_functions(tools)

    # --- Core Methods (Lifecycle & Interaction) ---

    async def cleanup(self) -> None:
        """Cleanup agent resources and state.

        Called when the agent is being shut down or when the session is being cleaned up.
        Subclasses should override this method to cleanup any resources they have allocated
        (e.g., file handles, network connections, background tasks).

        We don't clean internal state because the whole Agent will be destroyed anyway.

        """
        logger.debug(f"Agent {self.agent_name}: No persistent resourcces to cleanup.")

    # --- Publishing ---

    async def _publish(
        self,
        message: Any,
        topic_id: TopicId | str | None = None,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Publish a message to the group chat or a specific topic."""
        target_topic = topic_id or self._topic_id
        if not self._publish_fn:
            logger.debug(f"Agent {self.agent_name} ({self.agent_id}) sent {type(message).__name__} (no runtime).")
            return
        logger.debug(
            f"Agent {self.agent_name} ({self.agent_id}) sent {type(message).__name__} to {target_topic}.",
        )
        await self._publish_fn(message, target_topic)

    # --- Core Execution Logic ---

    async def invoke(
        self,
        message: AgentInput | StepRequest | str,
        *,
        context: Any | None = None,
        **kwargs: Any,
    ) -> ExecutionTrace | None:
        """Prepare input, calls the agent's core logic, and handles callbacks.

        It performs the following steps:
        1. Augments the incoming `message` with the agent's internal state using
           `_add_state_to_input`.
        2. Notifies listeners (via `public_callback`) that task processing has started.
        3. Invokes the agent's core logic via `self._process`.
        4. Handles any errors during execution, creating an `ErrorEvent` if necessary.
        5. Notifies listeners that task processing has completed (or failed).
        6. Constructs an `ExecutionTrace` object containing details of the execution,
           including inputs, outputs (if any), and configuration.
        7. Publishes the `ExecutionTrace` to listeners.


        Args:
            message: The initial `AgentInput` message for the agent.
            public_callback: An asynchronous callback function to publish messages
                (like status updates, traces) to a general or public topic.
            cancellation_token: An optional `CancellationToken` to signal if the
                operation should be aborted. (Currently not deeply integrated into the core loop).
            **kwargs: Additional keyword arguments that might be passed by the caller.
                These are not directly used by `invoke` but are available for potential extensions.

        Returns:
            ExecutionTrace: An object detailing the agent's execution for this invocation.
            None: If the agent does not run.

        Raises:
            ProcessingError: If `_add_state_to_input` fails. (Errors from the actual call
                are caught and reported in the `ExecutionTrace` and `TaskProcessingComplete` event).

        """
        await self._publish(
            TaskProcessingStarted(agent_id=self.agent_id, role=self.role),
            topic_id=self._topic_id,
        )

        # --- Prepare the input state for processing ---
        try:
            final_input = await self._add_state_to_input(message)
        except Exception as e:
            logger.error(f"Error preparing data for Agent {self.agent_id}: {e}")
            await self._publish(
                TaskProcessingComplete(
                    agent_id=self.agent_id,
                    role=self.role,
                    is_error=True,
                    error=str(e),  # TaskProcessingComplete.error expects string
                ),
                topic_id=self._topic_id,
            )
            return None

        try:
            trace_object = await self.trace_and_execute(message=final_input)

            # If the agent didn't run, just exit.
            if not trace_object:
                return None
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error during invoke: {e}")
            await self._publish(
                TaskProcessingComplete(
                    agent_id=self.agent_id,
                    role=self.role,
                    is_error=True,
                    error=str(e),  # TaskProcessingComplete.error expects string
                ),
                topic_id=self._topic_id,
            )
            return None

        # Store trace to BigQuery if configured
        try:
            from buttermilk.utils.trace_writer import get_trace_writer

            trace_writer = get_trace_writer()
            await trace_writer.add(trace_object)
        except Exception as e:
            logger.warning(f"Failed to store trace: {e}")

        # Publish the ExecutionTrace result.
        # Importantly, StepRequests might be sent privately or to a subset of agents. But we
        # want to publish the trace to the general topic so it can be consumed by any interested parties.
        # So we publish to self._topic_id, not ctx.topic_id.
        await self._publish(trace_object, topic_id=self._topic_id)

        # Publish status update: Task Complete (including error if error)
        await self._publish(
            TaskProcessingComplete(
                agent_id=self.agent_id,
                role=self.role,
                is_error=trace_object.is_error,
            ),
            topic_id=self._topic_id,
        )

        logger.debug(f"Agent {self.agent_name} finished task {message}.")

        return trace_object

    async def trace_and_execute(
        self,
        message: AgentInput,
    ) -> ExecutionTrace | None:
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
            - ExecutionTrace: An object containing the results of the agent's processing, with
            complete tracing information.

            - None: If the agent does not produce output.

        Raises:
            None.

        """
        result: AgentOutput | None = None  # Ensure result is defined for finally block
        tracing_link: str | None = None
        # Initialize parent_call for tracing with just an ID; enrich later if we are using Weave

        # --- Tracing ---
        {
            "name": self.agent_name,
            "model": (self._config.parameters or {}).get("model"),
            **(message.parameters or {}),
            **(message.metadata or {}),
            **(self.parameters or {}),
        }
        await bm.get_weave_client()

        # Get OTEL tracer for agent spans
        tracer = trace.get_tracer("buttermilk.agent")

        # Create comprehensive agent trace info using our utility function
        agent_trace_info = create_agent_trace_info(
            agent=self,
            template_hash=None,  # Will be populated from parameters if available
        )

        # Build OTEL span attributes from agent trace info
        span_attributes = {
            # Core agent identity
            "agent.name": self.agent_name,
            "agent.id": self.agent_id,
            "agent.type": agent_trace_info.get("agent_type"),  # Simple: "judge", "fetchagent"
            "agent.class": agent_trace_info.get("agent_class"),  # Full: "buttermilk.agents.judge.Judge"
            "agent.role": agent_trace_info.get("agent_role"),
            # Critical parameters for reproducibility
            "agent.model": agent_trace_info.get("model"),
            "agent.template": agent_trace_info.get("template"),
            "agent.template_hash": agent_trace_info.get("template_hash"),
        }

        # Add session/parent context
        if session_id := getattr(message, "session_id", None):
            span_attributes["session_id"] = session_id
        if parent_call_id := getattr(message, "parent_call_id", None):
            span_attributes["parent_call_id"] = parent_call_id

        # Filter out None values to avoid OpenTelemetry warnings
        span_attributes = {k: v for k, v in span_attributes.items() if v is not None}

        with tracer.start_as_current_span(
            f"agent.{self.agent_name}",
            attributes=span_attributes,
        ) as otel_span:
            try:
                logger.debug(f"Invoking Agent {self.agent_id} with args: {message}")
                # Weave has been removed

                # Run without weave tracing
                result = await self._process(message=message)

                otel_span.set_status(trace.Status(trace.StatusCode.OK))
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error during invoke: {e}")
                # Create an ErrorEvent to capture the error
                err_result = ErrorEvent(source=self.agent_id, content=f"Invoke error: {e}")
                result = AgentOutput(agent_id=self.agent_id, outputs=None, error=[err_result])
                otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
            finally:
                # Weave tracing has been removed - nothing to finalize
                pass

        # --- Turn the result into ExecutionTrace for long-term storage ---
        # Handle case where _process returns None (e.g., UI agents that don't produce output)
        # This doesn't include errors or null results since they are captured in the AgentOutput
        if result is None:
            return None

        # Create ExecutionTrace from the result, overwriting call_id and parent_call_id with
        # values directly from Weave.
        #
        # For inputs: Use resolved_inputs from metadata if subclass provided it (e.g., LLM agents
        # set this with template variables). Otherwise, extract just the input data from the message,
        # not the entire message wrapper with metadata.
        trace_inputs = result.metadata.get("resolved_inputs") if hasattr(result, "metadata") else None
        if trace_inputs is None:
            # Default: use message.inputs if available, otherwise fall back to whole message
            trace_inputs = message.inputs if hasattr(message, "inputs") else message

        trace_object = ExecutionTrace.from_output(
            result,
            parent_call_id=message.parent_call_id if hasattr(message, "parent_call_id") else None,
            call_id=result.call_id if hasattr(result, "call_id") else None,
            inputs=trace_inputs,
            agent_info={
                "component_name": self.agent_name,
                "agent_class": self.__class__.__name__,
                "agent_id": self.agent_id,
                "role": self.role,
            },
            tracing={"tracing_link": tracing_link} if tracing_link else None,
            record=message.record if hasattr(message, "record") else None,
        )

        return trace_object

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
        - Interface agents (e.g., for user interaction) might return `UserResponseMessage`
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
    @message_handler  # Announce on ConductorRequest
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

        # Convert AgentToolDefinition objects to ToolSchema for serialization
        tool_schemas = [tool_def.schema for tool_def in tool_definitions]

        announcement = AgentAnnouncement(
            content=f"Agent {self.agent_name} active and available",
            agent_config=self._config,
            available_tools=[],
            tool_definitions=tool_schemas,
            status="active",
            announcement_type="initial",
            source=self.agent_id,
        )

        await self._publish(announcement, topic_id=self._topic_id)

        # Mark as announced
        self._announced = True

    @message_handler  # Invoke on StepRequest
    async def handle_request(
        self,
        message: StepRequest,
        ctx: MessageContext,
    ) -> ExecutionTrace | None:
        """Handle an invocation message, preparing input and calling the agent's core logic.

        This method is designed to be used by host agents to invoke
        the agent for a single processing step. It augments the incoming `message`
        with the agent's internal state, notifies listeners of task processing,
        and invokes the agent's core logic.

        Args:
            message: The initial `AgentInput` message for the agent.
            ctx: Message context containing sender and topic information.

        Returns:
            ExecutionTrace: An object detailing the agent's execution for this invocation.
            None: If the agent does not run.

        Raises:
            ProcessingError: If `_add_state_to_input` fails. (Errors from the actual call
                are caught and reported in the `ExecutionTrace` and `TaskProcessingComplete` event).

        """
        if message.role != self.role:
            # Only handle if the role matches this agent's role
            return None

        return await self.invoke(message=message)

    @message_handler  # Add agent output messages to model context
    async def handle_agent_output(
        self,
        message: AgentOutput | ExecutionTrace,
        ctx: MessageContext,
    ) -> None:
        """Handle AgentOutput or ExecutionTrace messages, extracting Records and data based on our configured input mappings.

        ExecutionTrace is a subclass of AgentOutput with additional run and tracing information.

        Args:
            message: The ExecutionTrace or AgentOutput message to process.
            ctx: Message context containing sender and topic information.

        """
        source = ctx.sender.key if ctx.sender else "unknown"

        # Build complete input mappings including record and context
        # Start with regular inputs, then add record/context if configured
        all_mappings: dict[str, str] = dict(self.inputs) if self.inputs else {}
        if self.record_mapping:
            all_mappings["record"] = self.record_mapping
        if self.context_mapping:
            all_mappings["context"] = self.context_mapping

        # Extract data based on input mappings
        if all_mappings:  # Only extract if any mappings are defined
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=all_mappings,
            )
            # Add extracted data to self._data
            found_keys = []
            for key, value in extracted.items():
                if value is not None and value not in (
                    [],
                    {},
                ):  # Ensure value is meaningful
                    self._data.add(key, value)
                    found_keys.append(key)
            if found_keys:
                logger.debug(f"Agent {self.agent_name} extracted data for keys {found_keys} from {source} via mappings.")
        else:
            logger.debug(f"Agent {self.agent_name} has no input mappings defined; skipping data extraction.")

        # Add relevant message content to the conversation history (_model_context).
        if content_to_add := getattr(message, "content", None):
            await self._model_context.add_message(
                AssistantMessage(content=str(content_to_add), source=source or self.agent_name),
            )

    @message_handler  # Add UserResponseMessage content to model context
    async def handle_user_response_message(
        self,
        message: UserResponseMessage,
        ctx: MessageContext,
    ) -> None:
        """Handle UserResponseMessage messages, adding non-command content to model context.

        Args:
            message: The UserResponseMessage to process.
            ctx: Message context containing sender and topic information.

        """
        source = ctx.sender.key if ctx.sender else "manager"

        # Extract data based on input mappings if defined
        if self.inputs:
            extracted = extract_message_data(
                message=message,
                source=source,
                input_mappings=self.inputs,
            )
            # Add extracted data to self._data
            found_keys = []
            for key, value in extracted.items():
                if value is not None and value not in (
                    [],
                    {},
                ):  # Ensure value is meaningful
                    self._data.add(key, value)
                    found_keys.append(key)
            if found_keys:
                logger.debug(f"Agent {self.agent_name} extracted data for keys {found_keys} from {source} via mappings.")

        # Add to model context if not a command
        if message.content:
            content_str = str(message.content)
            if not content_str.startswith(COMMAND_SYMBOL):  # Avoid adding command-like messages to history
                await self._model_context.add_message(UserMessage(content=content_str, source=source))

    # --- Helper Methods ---

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """Augments an incoming `AgentInput` message with the agent's internal state.

        This crucial helper method prepares the final input that `_process` will receive.
        It merges several sources of information:
        1.  **Default Parameters**: Parameters defined in the agent's configuration
            (`self.parameters`) are used as a base.
        2.  **Message Parameters**: Parameters from the incoming `inputs.parameters`
            override any defaults.
        3.  **Record Mapping**: If `self.record_mapping` is configured (via config.record),
            extract record from `self._data` and set `updated_inputs.record`.
        4.  **Context Mapping**: If `self.context_mapping` is configured (via config.context),
            extract context from `self._data` (currently handled via _model_context).
        5.  **Resolved Input Mappings**: Data from `self._data` (which is populated
            by `_listen` based on `self.inputs` mappings) is resolved and added to
            `updated_inputs.inputs`. Incoming `inputs.inputs` can override these.
            Note: 'record' and 'context' keys in inputs are skipped (handled explicitly above).
        6.  **Conversation History**: Messages from `self._model_context` are prepended
            to `updated_inputs.context`.

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

        # 2. Handle record mapping explicitly using config.record field
        # This happens BEFORE processing regular inputs to avoid special-case logic in the loop
        if self.record_mapping and not updated_inputs.record:
            try:
                # Extract record from self._data using the configured mapping key
                # The mapping value (JMESPath expression) was already used by _listen to populate _data
                # Here we just need to check if the key exists in _data and extract the value
                # For backward compatibility: if config.record is set, we look for 'record' in _data
                record_values = self._data.get("record", [])
                if record_values:
                    record_data = record_values[-1]  # Get most recent
                    if isinstance(record_data, dict):
                        updated_inputs.record = BaseRecord.from_dict(record_data)
                    else:
                        updated_inputs.record = record_data
            except Exception as e:
                raise ProcessingError(f"Error resolving record mapping for agent {self.agent_id}: {e!s}") from e

        # 3. Handle context mapping explicitly using config.context field
        # Note: Currently context comes from _model_context.get_messages(), but if explicit
        # context mapping is configured, we should handle it here
        # For now, keeping existing behavior (context handled in step 5 below)

        # 4. Resolve regular input mappings using data stored in self._data.
        if updated_inputs.inputs is None:
            updated_inputs.inputs = {}
        if self.inputs:  # self.inputs is the mapping configuration from AgentConfig
            try:
                extracted_data = {}
                for key in self.inputs.keys():  # Iterate over configured input mapping keys
                    # Skip 'record' - handled explicitly above via self.record_mapping
                    if key == "record":
                        continue

                    # Skip 'context' - handled separately below
                    if key == "context":
                        continue

                    # Retrieve data from self._data; note that KeyValueCollector stores values in lists
                    data_values = self._data.get(key, [])
                    extracted_data[key] = data_values

                # Merge resolved mappings, letting original message inputs override
                merged_inputs_dict = {**extracted_data, **updated_inputs.inputs}
                updated_inputs.inputs = merged_inputs_dict
            except Exception as e:
                raise ProcessingError(f"Error resolving input mappings for agent {self.agent_id}: {e!s}") from e

        # 5. Prepend conversation history from agent's context.
        if updated_inputs.context is None:
            updated_inputs.context = []
        try:
            history = await self._model_context.get_messages()
            updated_inputs.context = history + updated_inputs.context  # Prepend history
        except Exception as e:
            logger.error(f"Agent {self.agent_name}: Error retrieving model context: {e!s}")
            # Decide handling: continue without history or raise? For now, log and continue.

        # 6. Cleanup and validation
        # TODO: @nicsuzor decide if we need to remove inputs that are not in the Agent's input schema.

        # Remove empty lists from inputs (JMESPath returns [] when no match)
        if updated_inputs.inputs:
            updated_inputs.inputs = {k: v for k, v in updated_inputs.inputs.items() if not (isinstance(v, list) and len(v) == 0)}

        # Filter inputs to only include keys in required list
        if self.required_inputs is not None and updated_inputs.inputs:
            filtered_inputs = {k: v for k, v in updated_inputs.inputs.items() if k in self.required_inputs}
            updated_inputs.inputs = filtered_inputs

        # Validate all required inputs are present (fail-fast)
        # Only validate if required is set and non-empty
        if self.required_inputs:
            available_keys = set(updated_inputs.inputs.keys()) if updated_inputs.inputs else set()
            missing_keys = set(self.required_inputs) - available_keys
            if missing_keys:
                raise FatalError(
                    f"Agent {self.agent_id} is missing required inputs: {sorted(missing_keys)}. "
                    f"Available inputs: {sorted(available_keys)}. "
                    f"Ensure the pipeline provides all required inputs."
                )

        logger.debug(
            f"Agent {self.agent_id}: Added state to input. "
            f"Final input keys: {list(updated_inputs.inputs.keys()) if updated_inputs.inputs else []}, "
            f"Context length: {len(updated_inputs.context)}, "
            f"Has record: {updated_inputs.record is not None}.",
        )
        # DEBUG: Log actual values to diagnose template unfilled issue
        if updated_inputs.inputs:
            for k, v in updated_inputs.inputs.items():
                val_preview = str(v)[:100] if v else "<EMPTY>"
                logger.debug(f"Agent {self.agent_id}: input[{k}] = {val_preview}")

        return updated_inputs

    def get_tool_definitions(self) -> list[Tool]:
        """Generate structured tool definitions for this agent.

        Returns:
            List of Tool objects representing this agent's capabilities.
        """
        from buttermilk._core.tool_definition import AgentToolDefinition

        # Create a tool definition for the agent's main processing capability
        tool_def = AgentToolDefinition(
            name=f"{self.role}_call",
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
