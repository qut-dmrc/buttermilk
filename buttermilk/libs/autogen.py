import asyncio
from collections.abc import Awaitable, Callable
from typing import Sequence

from autogen_core import (
    DefaultTopicId,CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)

from buttermilk._core.agent import Agent, AgentConfig, ToolOutput
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
    TaskProcessingComplete,
    TaskProcessingStarted,
    UserInstructions,
    AllMessages,
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
            asyncio.create_task(self.agent.initialize(input_callback=self._make_publish_callback()))
        else:
            asyncio.create_task(self.agent.initialize())

    @message_handler
    async def _heartbeat(self, message: HeartBeat, ctx:MessageContext) -> None:
        """Handle heartbeat messages by adding to the wrapped agent's queue."""
        try:
            self.agent._heartbeat.put_nowait(message.go_next)
        except asyncio.QueueFull:
            logger.debug(f"Heartbeat failed, agent {self.id} is idle or running behind.")

    @message_handler
    async def handle_invocation(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> (
        Sequence[AgentOutput | ToolOutput | TaskProcessingComplete] | AgentOutput | ToolOutput | TaskProcessingComplete | TaskProcessingStarted | None
    ):
        """Handle public request for agent to act. It's possible to return a value
        to the caller, but usually any result would be published back to the group."""
        response = None
        public_callback = self._make_publish_callback(topic_id=self.topic_id)

        # Signal that we have started work
        await public_callback(TaskProcessingStarted(agent_id=self.id, role=self.type, task_index=0))
        response = await self.agent(
            message=message,
            cancellation_token=ctx.cancellation_token,
        )
        if response:
            await public_callback(response)
        await public_callback(TaskProcessingComplete(agent_id=self.id, role=self.type, task_index=0, more_tasks_remain=False, is_error=False))
        return response

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None:
        """Handle incoming group messages by delegating to the wrapped agent."""
        await self.agent._listen(
            message=message,
            cancellation_token=ctx.cancellation_token,
            public_callback=self._make_publish_callback(topic_id=self.topic_id),
            message_callback=self._make_publish_callback(topic_id=ctx.topic_id),
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )

    @message_handler
    async def handle_conductor_request(
        self,
        message: ConductorRequest,
        ctx: MessageContext,
    ) -> (
        ConductorResponse
        | Sequence[ConductorResponse | AgentOutput | ToolOutput | TaskProcessingComplete]
        | AgentOutput
        | ToolOutput
        | TaskProcessingComplete
        | None
    ):
        """Handle conductor requests privately."""

        output = await self.agent(
            message=message,
            cancellation_token=ctx.cancellation_token,
            public_callback=self._make_publish_callback(topic_id=self.topic_id),
            message_callback=self._make_publish_callback(topic_id=ctx.topic_id),
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )
        return output  # only the last matching message

    @message_handler
    async def handle_control_message(
        self,
        message: OOBMessages,
        ctx: MessageContext,
    ) -> OOBMessages | Sequence[OOBMessages] | None:
        """Handle control messages sent OOB. Any response must also be OOB."""
        response = None
        response = await self.agent._handle_control_message(
            message=message,
            cancellation_token=ctx.cancellation_token,
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )
        return response

    def _make_publish_callback(self, topic_id=None) -> Callable[[UserInstructions], Awaitable[None]]:
        """Create a callback for handling publishing from client if required.

        Returns:
            Optional callback function.

        """
        """Messages come in from the UI and get sent back out through Autogen."""
        if not topic_id:
            topic_id = self.topic_id

        async def input_callback(message: FlowMessage) -> None:
            """Callback function to handle user input"""
            await self.publish_message(
                message,
                topic_id=topic_id,
            )

        return input_callback
