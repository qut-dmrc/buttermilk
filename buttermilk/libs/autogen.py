"""Provides the adapter layer to integrate Buttermilk agents with the autogen-core runtime.

This module defines `AutogenAgentAdapter`, which wraps a standard Buttermilk `Agent`
and exposes it to the Autogen ecosystem as an `autogen_core.RoutedAgent`. It handles
message translation, routing via topics, and lifecycle management within the Autogen
runtime.
"""

import asyncio
from collections.abc import (
    Awaitable,
    Callable,  # Added Union for type hints
)

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    TopicId,  # Identifier for message topics.
    message_handler,  # Decorator to register methods as message handlers.
)

from buttermilk._core.agent import Agent  # Buttermilk base agent and config.
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import (
    AgentInput,  # Standard input message for Buttermilk agents.
    AllMessages,
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    GroupchatMessageTypes,
    HeartBeat,
    OOBMessages,
    StepRequest,  # Union type for Out-Of-Band control messages.
)
from buttermilk._core.log import logger


class AutogenAgentAdapter(RoutedAgent):
    """Wraps a Buttermilk `Agent` to function as an `autogen_core.RoutedAgent`.

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
        _background_tasks (set): Set of background tasks created by this adapter.

    """

    agent: Agent  # The wrapped Buttermilk agent instance.
    topic_id: TopicId  # The primary topic ID for this agent adapter.
    _background_tasks: set[asyncio.Task]  # Track all background tasks

    def __init__(
        self,
        topic_type: str,  # The string type used to create the default TopicId.
        agent: Agent | None = None,  # Optional pre-instantiated Buttermilk agent.
        agent_cls: type[Agent] | None = None,  # Optional Buttermilk agent class.
        agent_cfg: AgentConfig | None = None,  # Optional config if instantiating from class.
        registration_callback: Callable[[str, Agent], None] | None = None,
    ) -> None:
        """Initializes the AutogenAgentAdapter.

        Requires either a pre-instantiated `agent` or both `agent_cls` and `agent_cfg`
        to instantiate a new agent. Sets up the agent, topic ID, and Autogen base class.

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
            logger.debug(f"Adapter initialized with pre-instantiated agent: {self.agent.agent_name} ({type(self.agent).__name__})")
            if registration_callback:
                registration_callback(self.agent.agent_id, self.agent)
        elif agent_cls and agent_cfg:
            # Instantiate the agent using the provided class and config.
            try:
                # Convert AgentConfig to dict if needed
                if hasattr(agent_cfg, 'model_dump'):
                    config_dict = agent_cfg.model_dump()
                else:
                    config_dict = agent_cfg
                self.agent = agent_cls(**config_dict)
                logger.debug(f"Adapter instantiated agent: {self.agent.agent_name} ({agent_cls.__name__})")
                if registration_callback:
                    registration_callback(self.agent.agent_id, self.agent)
            except Exception as e:
                logger.error(f"Failed to instantiate agent {agent_cls.__name__} with config {agent_cfg}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__}") from e
        else:
            # Insufficient information provided.
            raise ValueError("AutogenAgentAdapter requires either a pre-instantiated 'agent' or both 'agent_cls' and 'agent_cfg'.")

        # Set the default topic ID based on the provided type string.
        self.topic_id = DefaultTopicId(type=topic_type)
        
        # Initialize task tracking
        self._background_tasks = set()

        # Initialize the base Autogen RoutedAgent class, using the Buttermilk agent's description.
        super().__init__(description=self.agent.description)

        # This allows UI agents, for example, to send user input back into the Autogen flow.
        # Check if agent supports announcements
        if hasattr(self.agent, 'initialize_with_announcement'):
            # Initialize with announcement capability
            init_task = self.agent.initialize_with_announcement(
                callback_to_groupchat=self._make_publish_callback(),
                public_callback=self._make_publish_callback()
            )
        else:
            # Standard initialization
            init_task = self.agent.initialize(callback_to_groupchat=self._make_publish_callback())
        
        task = asyncio.create_task(init_task)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        logger.debug(f"Scheduled initialization for agent {self.agent.agent_name} with callback.")

    def _make_publish_callback(self, topic_id: TopicId | None = None) -> Callable[[FlowMessage], Awaitable[None]]:
        """Create an asynchronous callback function that the wrapped Buttermilk agent can use
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

        async def publish_callback(message: FlowMessage | FlowEvent) -> None:
            """The actual callback that publishes the message using the adapter."""
            logger.debug(f"Publish callback invoked by agent {self.agent.agent_name}. Publishing {type(message).__name__} to topic {target_topic_id}")
            # Use the adapter's inherited publish_message method.
            await self.publish_message(
                message,
                topic_id=target_topic_id,
            )

        return publish_callback

    async def cleanup(self) -> None:
        """Clean up the adapter and its wrapped agent.
        
        Cancels all background tasks and calls cleanup on the wrapped agent.
        """
        logger.debug(f"Cleaning up AutogenAgentAdapter for agent {self.agent.agent_name}")
        
        # Cancel all background tasks
        if self._background_tasks:
            logger.debug(f"Cancelling {len(self._background_tasks)} background tasks for agent {self.agent.agent_name}")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for background tasks cancellation in adapter for {self.agent.agent_name}")
            
            self._background_tasks.clear()
        
        # Cleanup the wrapped agent
        try:
            # Store the announcement callback if agent supports it
            if hasattr(self.agent, '_announcement_callback'):
                self.agent._announcement_callback = self._make_publish_callback()
            
            # Check if agent supports cleanup with announcement
            if hasattr(self.agent, 'cleanup_with_announcement'):
                await self.agent.cleanup_with_announcement()
            else:
                await self.agent.cleanup()
            
            logger.debug(f"Agent {self.agent.agent_name} cleanup completed")
        except Exception as e:
            logger.warning(f"Error during agent cleanup for {self.agent.agent_name}: {e}")

    async def on_unregister(self) -> None:
        """Called when the agent is being unregistered from the runtime.
        
        This is the proper place to send leaving announcements.
        """
        logger.debug(f"Agent {self.agent.agent_name} being unregistered")
        await self.cleanup()

    @message_handler
    async def _heartbeat(self, message: HeartBeat, ctx: MessageContext) -> None:
        """Handles internal HeartBeat messages (if used by the runtime/orchestrator)."""
        # This allows external control/checking if the agent is alive or ready for work.
        # Puts a signal into the agent's internal heartbeat queue.
        try:
            self.agent._heartbeat.put_nowait(message.go_next)
        except asyncio.QueueFull:
            # If the agent isn't processing heartbeats quickly enough.
            logger.debug(f"Heartbeat queue full for agent {self.agent.agent_name}. Agent may be busy or stuck.")

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,  # Handles messages intended for general group chat consumption.
        ctx: MessageContext,
    ) -> None:
        """Handle broadcast/group chat messages by delegating to the agent's `_listen` method.

        This allows agents to react to general messages published on the topic, even if not
        directly addressed to them. Typically used for information sharing or awareness.

        Args:
            message: The group chat message (can be various types).
            ctx: The Autogen message context.

        """
        logger.debug(f"Agent {self.agent.agent_name} received group chat message: {type(message).__name__}")

        public_callback = self._make_publish_callback(topic_id=self.topic_id)

        try:
            # Delegate to the agent's _listen method for processing.
            await self.agent._listen(
                message=message,
                cancellation_token=ctx.cancellation_token,
                public_callback=public_callback,  # Callback for default topic
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id),  # Callback for specific incoming topic
                source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",  # Extract sender ID
            )
        except Exception as e:
            msg = f"Error listening to chat message: {e}"
            logger.error(msg, exc_info=False)
            await public_callback(ErrorEvent(source=self.agent.agent_id, content=msg))

    @message_handler
    async def handle_control_message(
        self,
        message: OOBMessages,  # Handles out-of-band control messages.
        ctx: MessageContext,
    ) -> AllMessages | None:
        """Handles Out-Of-Band (OOB) control messages.

        Delegates to the agent's `_handle_events` method. These messages are for
        control signals outside the main data flow (e.g., reset, status requests).
        Any response returned by the agent is sent back directly to the caller.

        Args:
            message: The OOB control message.
            ctx: The Autogen message context.

        Returns:

        """
        response: AllMessages | None = None

        try:
            if isinstance(message, StepRequest) and message.role == self.agent.role:
                # Call agent's main execution method
                # StepRequest is a subclass of AgentInput, so we can use it directly.
                response = await self.agent.invoke(message=message,
                    cancellation_token=ctx.cancellation_token,
                    source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",  # Extract sender ID
                public_callback=self._make_publish_callback(topic_id=self.topic_id),  # Callback for default topic
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id),  # Callback for specific incoming topic
                )
            else:
                # Delegate to the agent's _handle_events method.
                response = await self.agent._handle_events(
                    message=message,
                    cancellation_token=ctx.cancellation_token,
                    source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",  # Extract sender ID
                public_callback=self._make_publish_callback(topic_id=self.topic_id),  # Callback for default topic
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id),  # Callback for specific incoming topic
                )
            logger.debug(f"Agent {self.agent.agent_name} completed handling control message: {type(message).__name__}. Response type: {type(response).__name__ if response else 'N/A'}")

            return response  # Return response directly for OOB messages.
        except Exception as e:
            msg = f"Error during agent {self.agent.agent_id} handling control message {type(message).__name__}: {e}"
            logger.error(msg, exc_info=True)
            await self.publish_message(ErrorEvent(source=self.agent.agent_id, content=msg), topic_id=self.topic_id)
            return None

    @message_handler
    async def handle_invocation(
        self,
        message: AgentInput | StepRequest,  # Handles the standard Buttermilk agent input message.
        ctx: MessageContext,  # Provides context like sender, topic, cancellation token.
    ) -> AllMessages:
        """Handles direct invocation requests (`AgentInput`) for the agent to perform its primary task.

        This typically corresponds to the Buttermilk agent's `__call__` method.
        It publishes TaskProcessingStarted/Complete messages and the agent's output
        to the default topic.

        Args:
            message: The input data and prompt for the agent.
            ctx: The Autogen message context.

        Returns:
            The direct output from the agent's execution, which might be None, a single
            response object (AgentTrace, ToolOutput, etc.), or a sequence of them.
            Autogen uses this return value as the response to the `send_message` call.

        """
        logger.debug(f"Agent {self.agent.agent_name} received AgentInput invocation.")
        output: AllMessages | None = None

        try:
            # Delegate the actual work to the wrapped Buttermilk agent's __call__ method.
            # Pass the cancellation token from Autogen context.
            output = await self.agent.invoke(
                message=message,
                cancellation_token=ctx.cancellation_token,
                # Provide callbacks for the agent to publish messages back if needed during execution.
                public_callback=self._make_publish_callback(topic_id=self.topic_id),
                message_callback=self._make_publish_callback(topic_id=ctx.topic_id),  # Callback for topic of incoming message
                source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",  # Extract sender ID
            )
            logger.debug(f"Agent {self.agent.agent_name} completed invocation. Output type: {type(output).__name__}")

            return output  # Return the direct output to the caller in Autogen.

        except Exception as e:
            logger.error(f"Error during agent {self.agent.agent_name} invocation: {e}")
            # Publish an ErrorEvent message for listeners.
            await self.publish_message(ErrorEvent(source=self.agent.agent_id, content=str(e)), topic_id=self.topic_id)
            # Do not return the exception itself, let Autogen handle failed `send_message`.
            return None
