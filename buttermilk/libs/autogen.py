import asyncio
from collections.abc import Awaitable, Callable
from typing import AsyncGenerator

from autogen_core import (
    DefaultTopicId,CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)

from buttermilk._core.agent import Agent, AgentConfig
from buttermilk._core.contract import (
    CONDUCTOR,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
    FlowMessage,
    ManagerRequest,
    OOBMessages,
    UserInstructions,AllMessages,
)
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.ui.generic import UIAgent 
from buttermilk.bm import logger


class AutogenAgentAdapter(RoutedAgent):
    """Adapter for integrating Buttermilk agents with Autogen runtime.

    This adapter wraps a Buttermilk agent to make it compatible with the Autogen
    routing and messaging system. It handles message passing between the systems
    and manages agent initialization.

    Attributes:
        agent (Agent): The wrapped Buttermilk agent
        topic_id (TopicId): The topic identifier for message routing

    """

    def __init__(
        self,
        topic_type: str,
        agent: Agent = None,
        agent_cls: type = None,
        agent_cfg: AgentConfig = None,
    ):
        """Initialize the adapter with either an agent instance or configuration.

        Args:
            topic_type: The topic type for message routing
            agent: Optional pre-instantiated agent
            agent_cls: Optional agent class to instantiate
            agent_cfg: Optional agent configuration for instantiation

        Raises:
            ValueError: If neither agent nor agent_cfg is provided

        """
        if agent:
            self.agent = agent
        else:
            if not agent_cfg:
                raise ValueError("Either agent or agent_cfg must be provided")
            self.agent: Agent = agent_cls(
                **agent_cfg.model_dump(),
            )
        self.topic_id: TopicId = DefaultTopicId(type=topic_type)
        super().__init__(description=self.agent.description)

        self.is_manager = isinstance(self.agent, UIAgent) or isinstance(self.agent, HostAgent)

        # Take care of any initialization the agent needs to do in this event loop
        if self.is_manager:
            asyncio.create_task(self.agent.initialize(input_callback=self.handle_input()))
        else:
            asyncio.create_task(self.agent.initialize())


    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> AllMessages | None: 
        """Handle incoming messages by delegating to the wrapped agent."""
        # Pass messages along to the agent in two stages.
        # - First, pass to the listen function, which just extracts data
        #   that the agent is configured to listen for.
        # - Second, call _process, and the agent may choose to respond.
        #   Agent input variables can be supplied either from its own
        #   internal state or by the orchestrator in the _process() call.

        # First, enforce division into control and in-band messages
        if not isinstance(message, GroupchatMessageTypes):
            logger.warning(f"{self.id} received unexpected message type: {type(message)}")
            return None
        # Don't allow an agent to react to its own results.
        if isinstance(message, FlowMessage) and message.source == self.agent.id:
            return None
        # Autogen adds sender info, check that too
        if ctx.sender and ctx.sender.type == self.type:
             logger.debug(f"Agent {self.id} ignoring message from self ({ctx.sender.type})")
             return None
        
        # Listen first. For normal messages, extract and record data from the message.
        try:
            await self.agent.listen(message=message, ctx=ctx)
        except Exception as e:
            logger.error(f"Error during listen in agent {self.agent.id}: {e}", exc_info=True)
            return
               
        # Process 
        try:
            # Use agent's main processing method
            async for output in self.agent(message, cancellation_token=ctx.cancellation_token):
                 if output:
                     # Publish yielded messages
                     await self.publish_message(output, topic_id=self.topic_id)

            return None
        except Exception as e:
            logger.error(f"Error processing message in agent {self.agent.id}: {e}", exc_info=False)
            error_output = ErrorEvent(
                source=self.agent.id, role=self.agent.role, content=f"Error: {e} {e.args=}",                
            )
            await self.publish_message(error_output, topic_id=self.topic_id)
            return None
            


    @message_handler(match=lambda self,x,y: self.is_manager)
    async def handle_control_message(
        self,
        message: OOBMessages,
        ctx: MessageContext,
    ) -> OOBMessages | None: 
        """Handle separately isolated out-of-band control messages"""
        if self.is_manager:
            return await self.agent.handle_control_message(message=message, ctx=ctx)
        return None


    def handle_input(self) -> Callable[[UserInstructions], Awaitable[None]] | None:
        """Create a callback for handling user input if needed.

        Returns:
            Optional callback function for user input handling

        """
        """Messages come in from the UI and get sent back out through Autogen."""

        async def input_callback(user_message: UserInstructions) -> None:
            """Callback function to handle user input"""
            await self.publish_message(
                user_message,
                topic_id=self.topic_id,
            )
        
        return input_callback