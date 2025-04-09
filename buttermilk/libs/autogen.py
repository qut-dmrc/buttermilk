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
    ConductorResponse,
    ErrorEvent,
    GroupchatMessageTypes,
    HeartBeat,
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
    async def _heartbeat(self, message: HeartBeat, ctx:MessageContext) -> None:
        """Handle heartbeat messages by adding to the wrapped agent's queue."""
        try:
            self.agent._heartbeat.put_nowait(message.go_next)
        except asyncio.QueueFull:
            logger.warning(f"Heartbeat failed, agent {self.id} is running behind.")

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None: 
        """Handle incoming group messages by delegating to the wrapped agent."""
        response = await self.agent._listen(message=message, ctx=ctx)
        if response:
            await self.publish_message(response, topic_id=self.topic_id)

    @message_handler
    async def handle_invocation(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> AgentOutput | None: 
        """Handle public request for agent to act. It's possible to return a value
        to the caller, but usually any result would be published back to the group."""
        return await self.agent.invoke(message=message, ctx=ctx)

    @message_handler
    async def handle_conductor_request(
        self,
        message: ConductorRequest,
        ctx: MessageContext,
    ) -> ConductorResponse | None:
        """Handle conductor requests privately."""
        return await self.agent.invoke_privately(message=message, ctx=ctx)

    @message_handler
    async def handle_control_message(
        self,
        message: OOBMessages,
        ctx: MessageContext,
    ) -> OOBMessages | None: 
        """Handle control messages sent OOB. Any response must also be OOB."""
        return await self.agent.handle_control_message(message=message, ctx=ctx)

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
