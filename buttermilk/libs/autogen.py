"""
Provides the adapter layer to integrate Buttermilk agents with the autogen-core runtime.

This module defines `AutogenAgentAdapter`, which wraps a standard Buttermilk `Agent`
and exposes it to the Autogen ecosystem as an `autogen_core.RoutedAgent`. It handles
message translation, routing via topics, and lifecycle management within the Autogen
runtime.
"""
import asyncio
from collections.abc import Awaitable, Callable
from typing import Sequence, Union # Added Union for type hints

from autogen_core import (
    DefaultTopicId,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId, # Identifier for message topics.
    message_handler, # Decorator to register methods as message handlers.
)

from buttermilk._core.agent import Agent, AgentConfig, ToolOutput # Buttermilk base agent and config.
from buttermilk._core.contract import (
    CONDUCTOR, # Constant representing the Conductor role.
    AgentInput, # Standard input message for Buttermilk agents.
    AgentOutput,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    GroupchatMessageTypes,
    HeartBeat,
    ManagerMessage,
    FlowMessage,
    ManagerRequest,
    ManagerResponse, # Response from the Manager/UI.
    OOBMessages, # Union type for Out-Of-Band control messages.
    TaskProcessingComplete, # Status message indicating task completion.
    TaskProcessingStarted, # Status message indicating task start.
    UserInstructions, # Instructions from the user (potentially via Manager).
    AllMessages, # Union of all possible message types (likely for broader type hints).
)
# TODO: These specific agent imports might create coupling. Consider if interfaces or protocols could be used.
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.agents.flowcontrol.sequencer import Sequencer
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.bm import logger # Buttermilk logger instance.


# Define the expected return type more accurately using Union.
# Can return None, a single output, or a sequence of outputs.
MaybeSequenceOutput = Union[
    None,
    AgentOutput,
    ToolOutput,
    TaskProcessingComplete,
    TaskProcessingStarted,
    ConductorResponse,
    Sequence[Union[AgentOutput, ToolOutput, TaskProcessingComplete, ConductorResponse]],
]

class AutogenAgentAdapter(RoutedAgent):
    """
    Wraps a Buttermilk `Agent` to function as an `autogen_core.RoutedAgent`.

    This class acts as a bridge, translating messages and method calls between
    the Autogen runtime and a standard Buttermilk agent. It uses Autogen's
    `@message_handler` decorator to route different message types (AgentInput,
    GroupchatMessageTypes, ConductorRequest, OOBMessages, HeartBeat) to the
    appropriate methods of the wrapped Buttermilk agent (`__call__`, `_listen`,
    `_handle_events`). It also manages publishing responses and status updates
    back to the Autogen topic.

    Attributes:
        agent (Agent): The instance of the Buttermilk agent being wrapped.
        topic_id (TopicId): The default Autogen topic this agent primarily interacts with.
        is_manager (bool): Flag indicating if the wrapped agent is considered a manager/conductor type.
                           (Used for specific initialization logic).
    """

    agent: Agent # The wrapped Buttermilk agent instance.
    topic_id: TopicId # The primary topic ID for this agent adapter.
    is_manager: bool # Flag for manager-like agents requiring special init.

    def __init__(
        self,
        topic_type: str,  # The string type used to create the default TopicId.
        agent: Agent | None = None,  # Optional pre-instantiated Buttermilk agent.
        agent_cls: type[Agent] | None = None,  # Optional Buttermilk agent class.
        agent_cfg: AgentConfig | None = None,  # Optional config if instantiating from class.
    ) -> None:
        """
        Initializes the AutogenAgentAdapter.

        Requires either a pre-instantiated `agent` or both `agent_cls` and `agent_cfg`
        to instantiate a new agent. Sets up the agent, topic ID, and Autogen base class.
        Also performs specific initialization for manager-type agents.

        Args:
            topic_type: String identifier for the default Autogen topic.
            agent: An already initialized Buttermilk Agent instance.
            agent_cls: The class of the Buttermilk Agent to instantiate.
            agent_cfg: The configuration object for instantiating `agent_cls`.

        Raises:
            ValueError: If insufficient arguments are provided to determine the agent instance.
        """
        if agent:
            # Use the provided agent instance.
            self.agent = agent
            logger.debug(f"Adapter initialized with pre-instantiated agent: {self.agent.id} ({type(self.agent).__name__})")
        elif agent_cls and agent_cfg:
            # Instantiate the agent using the provided class and config.
            try:
                self.agent = agent_cls(**agent_cfg.model_dump())
                logger.debug(f"Adapter instantiated agent: {self.agent.id} ({agent_cls.__name__})")
            except Exception as e:
                logger.error(f"Failed to instantiate agent {agent_cls.__name__} with config {agent_cfg.id}: {e}", exc_info=True)
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__}") from e
        else:
            # Insufficient information provided.
            raise ValueError("AutogenAgentAdapter requires either a pre-instantiated 'agent' "
                             "or both 'agent_cls' and 'agent_cfg'.")

        # Set the default topic ID based on the provided type string.
        self.topic_id = DefaultTopicId(type=topic_type)

        # Initialize the base Autogen RoutedAgent class, using the Buttermilk agent's description.
        super().__init__(description=self.agent.description)

        # TODO: This check for manager types is brittle. Relying on specific class types
        # (UIAgent, LLMHostAgent, Sequencer) makes it hard to extend.
        # Consider using a role property, capability flag, or specific interface instead.
        self.is_manager = isinstance(self.agent, (UIAgent, LLMHostAgent, Sequencer))
        logger.debug(f"Agent {self.agent.id} identified as manager: {self.is_manager}")

        # Perform agent-specific asynchronous initialization in the current event loop.
        # Manager agents get a callback to publish messages back through the adapter.
        if self.is_manager:
            # This allows UI agents, for example, to send user input back into the Autogen flow.
            init_task = self.agent.initialize(input_callback=self._make_publish_callback())
            asyncio.create_task(init_task)
            logger.debug(f"Scheduled initialization for manager agent {self.agent.id} with callback.")
        else:
            # Non-manager agents might still have async initialization tasks.
            init_task = self.agent.initialize()
            asyncio.create_task(init_task)
            logger.debug(f"Scheduled standard initialization for agent {self.agent.id}.")

    @message_handler
    async def _heartbeat(self, message: HeartBeat, ctx: MessageContext) -> None:
        """Handles internal HeartBeat messages (if used by the runtime/orchestrator)."""
        # This allows external control/checking if the agent is alive or ready for work.
        # Puts a signal into the agent's internal heartbeat queue.
        try:
            self.agent._heartbeat.put_nowait(message.go_next)
        except asyncio.QueueFull:
            # If the agent isn't processing heartbeats quickly enough.
            logger.debug(f"Heartbeat queue full for agent {self.agent.id}. Agent may be busy or stuck.")

    @message_handler
    async def handle_invocation(
        self,
        message: AgentInput, # Handles the standard Buttermilk agent input message.
        ctx: MessageContext, # Provides context like sender, topic, cancellation token.
    ) -> MaybeSequenceOutput:
        """
        Handles direct invocation requests (`AgentInput`) for the agent to perform its primary task.

        This typically corresponds to the Buttermilk agent's `__call__` method.
        It publishes TaskProcessingStarted/Complete messages and the agent's output
        to the default topic.

        Args:
            message: The input data and prompt for the agent.
            ctx: The Autogen message context.

        Returns:
            The direct output from the agent's execution, which might be None, a single
            response object (AgentOutput, ToolOutput, etc.), or a sequence of them.
            Autogen uses this return value as the response to the `send_message` call.
        """
        logger.debug(f"Agent {self.agent.id} received AgentInput invocation.")
        output: MaybeSequenceOutput = None

        # Publish status update: Task Started
        # Use self.type (which is the agent's registered type/role in Autogen)
        await self.publish_message(TaskProcessingStarted(agent_id=self.agent.id, role=self.type, task_index=0), topic_id=self.topic_id)

        try:
            # Delegate the actual work to the wrapped Buttermilk agent's __call__ method.
            # Pass the cancellation token from Autogen context.
            output = await self.agent(
                message=message,
                cancellation_token=ctx.cancellation_token,
                # Provide callbacks for the agent to publish messages back if needed during execution.
                public_callback=self._make_publish_callback(topic_id=self.topic_id),
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id), # Callback for topic of incoming message
                source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown", # Extract sender ID
            )
            logger.debug(f"Agent {self.agent.id} completed invocation. Output type: {type(output).__name__}")

            # If the agent returned something directly, publish it to the main topic.
            if output and not isinstance(output, TaskProcessingComplete): # Don't republish completion status
                # TODO: Handle sequences of outputs correctly if agent returns multiple messages.
                # Currently might only publish the sequence object itself, not individual items.
                await self.publish_message(output, topic_id=self.topic_id)

            # Publish status update: Task Complete (Success)
            await self.publish_message(
                TaskProcessingComplete(agent_id=self.agent.id, role=self.type, task_index=0, more_tasks_remain=False, is_error=False),
                topic_id=self.topic_id,
            )
            return output # Return the direct output to the caller in Autogen.

        except Exception as e:
            logger.error(f"Error during agent {self.agent.id} invocation: {e}", exc_info=True)
            # Publish status update: Task Complete (Error)
            await self.publish_message(
                TaskProcessingComplete(agent_id=self.agent.id, role=self.type, task_index=0, more_tasks_remain=False, is_error=True),
                topic_id=self.topic_id,
            )
            # Publish an ErrorEvent message for listeners.
            await self.publish_message(ErrorEvent(source=self.agent.id, content=str(e)), topic_id=self.topic_id)
            # Do not return the exception itself, let Autogen handle failed `send_message`.
            return None

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes, # Handles messages intended for general group chat consumption.
        ctx: MessageContext,
    ) -> None:
        """
        Handles broadcast/group chat messages by delegating to the agent's `_listen` method.

        This allows agents to react to general messages published on the topic, even if not
        directly addressed to them. Typically used for information sharing or awareness.

        Args:
            message: The group chat message (can be various types).
            ctx: The Autogen message context.
        """
        logger.debug(f"Agent {self.agent.id} received group chat message: {type(message).__name__}")
        try:
            # Delegate to the agent's _listen method for passive processing.
            await self.agent._listen(
                message=message,
                cancellation_token=ctx.cancellation_token,
                public_callback=self._make_publish_callback(topic_id=self.topic_id), # Callback for default topic
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id), # Callback for specific incoming topic
                source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown", # Extract sender ID
            )
        except Exception as e:
            logger.error(f"Error during agent {self.agent.id} listening to group chat message: {e}", exc_info=True)
            # TODO: Consider publishing an ErrorEvent here as well?

    @message_handler
    async def handle_conductor_request(
        self,
        message: ConductorRequest, # Handles specific requests targeted at Conductor agents.
        ctx: MessageContext,
    ) -> MaybeSequenceOutput:
        """
        Handles `ConductorRequest` messages, typically intended for agents acting as conductors/hosts.

        Delegates processing to the wrapped agent's `__call__` method, similar to `handle_invocation`,
        but specifically typed for ConductorRequest. The response (often a `StepRequest` or
        `ConductorResponse`) is returned directly to the caller in Autogen.

        Args:
            message: The conductor request message.
            ctx: The Autogen message context.

        Returns:
            The direct output from the agent's execution.
        """
        logger.debug(f"Agent {self.agent.id} received ConductorRequest.")
        output: MaybeSequenceOutput = None
        # Note: No TaskProcessingStarted/Complete messages published here, assuming conductor interactions are synchronous requests.
        try:
            # Delegate to the agent's __call__ method.
            output = await self.agent(
                message=message,
                cancellation_token=ctx.cancellation_token,
                public_callback=self._make_publish_callback(topic_id=self.topic_id),
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id),
                source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown", # Extract sender ID
            )
            logger.debug(f"Agent {self.agent.id} completed ConductorRequest. Output type: {type(output).__name__}")
            # Note: Conductor responses are typically returned directly, not published separately by the adapter.
            return output
        except Exception as e:
            logger.error(f"Error during agent {self.agent.id} handling ConductorRequest: {e}", exc_info=True)
            # TODO: Consider publishing ErrorEvent or returning a specific error response?
            return None # Return None on error

    @message_handler
    async def handle_control_message(
        self,
        message: OOBMessages, # Handles out-of-band control messages.
        ctx: MessageContext,
    ) -> Union[OOBMessages, Sequence[OOBMessages], None]:
        """
        Handles Out-Of-Band (OOB) control messages.

        Delegates to the agent's `_handle_events` method. These messages are for
        control signals outside the main data flow (e.g., reset, status requests).
        Any response returned by the agent is sent back directly to the caller.

        Args:
            message: The OOB control message.
            ctx: The Autogen message context.

        Returns:
            An OOB message response, a sequence of them, or None.
        """
        logger.debug(f"Agent {self.agent.id} received control message: {type(message).__name__}")
        response: Union[OOBMessages, Sequence[OOBMessages], None] = None
        try:
            # Delegate to the agent's _handle_events method.
            response = await self.agent._handle_events(
                message=message,
                cancellation_token=ctx.cancellation_token,
                source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",  # Extract sender ID
            )
            logger.debug(f"Agent {self.agent.id} completed handling control message. Response type: {type(response).__name__}")
            return response # Return response directly for OOB messages.
        except Exception as e:
            logger.error(f"Error during agent {self.agent.id} handling control message: {e}", exc_info=True)
            # TODO: Return an error OOB message?
            return None

    def _make_publish_callback(self, topic_id: TopicId | None = None) -> Callable[[FlowMessage], Awaitable[None]]:
        """
        Creates an asynchronous callback function that the wrapped Buttermilk agent can use
        to publish messages back into the Autogen system via this adapter.

        This is crucial for agents (like UI agents or managers) that need to send messages
        (e.g., user input, commands) during their operation or initialization.

        Args:
            topic_id: The specific Autogen topic to publish the message to.
                      Defaults to the adapter's main `self.topic_id`.

        Returns:
            An async callback function that takes a `FlowMessage` and publishes it.
        """
        # Determine the target topic ID, defaulting to the adapter's main topic.
        target_topic_id = topic_id or self.topic_id

        async def publish_callback(message: FlowMessage) -> None:
            """The actual callback that publishes the message using the adapter."""
            logger.debug(f"Publish callback invoked by agent {self.agent.id}. Publishing {type(message).__name__} to topic {target_topic_id}")
            # Use the adapter's inherited publish_message method.
            await self.publish_message(
                message,
                topic_id=topic_id,
            )

        return publish_callback
