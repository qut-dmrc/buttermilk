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
    GroupchatMessages,
    ManagerMessage,
    FlowMessage,
    ManagerRequest,
    OOBMessages,
    UserInstructions,AllMessages,
)
from buttermilk.agents.flowcontrol.types import HostAgent
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
    async def receive_message(
        self,
        message: AllMessages,
        ctx: MessageContext,
    ) -> AllMessages | None: 
        """Handle incoming messages by delegating to the wrapped agent."""
        # Pass messages along to the agent, which then responds if required.
        # The agent's internal logic or orchestrator-provided context handles history.
        
        # First, divide into control and in-band messages
        if isinstance(message, GroupchatMessages):
            # For normal messages, extract and record data from the message.
            await self.agent.listen(message=message, ctx=ctx)
        elif isinstance(message, OOBMessages):
            # Control messages should be isolated out-of-band
            if self.is_manager:
                return await self.handle_oob(message=message, ctx=ctx)
            # Otherwise, ignore the message.
            return None
        else:
            raise ValueError(f"Unexpected message type: {type(message)}")
        
        # Process using the wrapped agent's _process method, which may yield outputs
        try:
            agent_output = None
            async for agent_output in self.agent._process(message, cancellation_token=ctx.cancellation_token):
                if self.id.type.startswith(CONDUCTOR):
                    # If we are the host, our replies are coordination, not content.
                    # In that case, don't publish them, only return them directly.
                    pass
                elif agent_output:
                    # Otherwise, send it out to all subscribed agents.
                    await self.publish_message(
                        agent_output,
                        topic_id=self.topic_id,
                    )
                
            return agent_output  # returns the last message generated

        except Exception as e:
            logger.error(f"Error processing request in agent {self.agent.id}: {e}", exc_info=True)
            error_output = AgentOutput(
                source=self.agent.id,
                role=self.agent.role,
                content=f"Error processing request: {e}",
                error=[str(e)],
                # Attempt to pass records if available in the input message
                records=getattr(message, 'records', []),
            )
            # Publish the error message 
            await self.publish_message(error_output, topic_id=self.topic_id)
            return error_output



    @message_handler
    async def handle_oob(
        self,
        message: OOBMessages,
        ctx: MessageContext,
    ) -> None:
        """Handle out-of-band control messages."""
        # Delegate to the agent's control message handler
        # Note: handle_control_message is async but doesn't return anything significant for routing
        await self.agent.handle_control_message(message)

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

        # Check against the generic UIAgent base class
        if isinstance(self.agent, UIAgent):
            return input_callback
        return None
