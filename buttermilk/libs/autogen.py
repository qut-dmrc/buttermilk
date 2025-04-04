import asyncio
from collections.abc import Awaitable, Callable

from autogen_core import (
    DefaultTopicId,
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
    ManagerMessage,
    ManagerRequest,
    UserInstructions,
)
from buttermilk.agents.flowcontrol.types import HostAgent
from buttermilk.agents.ui.console import UIAgent
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

        # Take care of any initialization the agent needs to do in this event loop
        asyncio.create_task(self.agent.initialize(input_callback=self.handle_input()))

    async def _process_request(
        self,
        message: AgentInput | ConductorRequest,
    ) -> AgentOutput | None:
        """Process an incoming request by delegating to the wrapped agent.

        Args:
            message: The input message to process

        Returns:
            Optional agent output from processing the request

        """
        agent_output = None
        try:
            # Process using the wrapped agent
            agent_output = await self.agent(message)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            # raise ProcessingError(f"Error processing request: {e}") from e

        if agent_output:
            if self.id.type.startswith(CONDUCTOR):
                # If we are the host, our replies are coordination, not content.
                # In that case, don't publish them, only return them directly.
                return agent_output
            # Otherwise, send it out to all subscribed agents.
            await self.publish_message(
                agent_output,
                topic_id=self.topic_id,
            )
            return agent_output

        return None

    @message_handler
    async def handle_request(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> AgentOutput | None:
        """Handle incoming agent input messages.

        This handler is triggered when an agent receives an input message requesting
        its services.

        Args:
            message: The input message to process
            ctx: Message context with sender information

        Returns:
            Optional agent output from processing the request

        """
        source = str(ctx.sender) if ctx and ctx.sender else message.type
        return await self._process_request(message)

    @message_handler
    async def handle_output(
        self,
        message: AgentOutput | UserInstructions,
        ctx: MessageContext,
    ) -> None:
        """Handle output messages from other agents.

        This handler processes outputs from other agents that might be relevant to
        this agent.

        Args:
            message: The output message to process
            ctx: Message context with sender information

        """
        try:
            source = str(ctx.sender.type)
            # if ctx and ctx.sender else message.type
            # ignore messages sent by us
            if source != self.id:
                response = await self.agent.receive_output(message)
                if response:
                    await self.publish_message(
                        response,
                        topic_id=self.topic_id,
                    )
            return
        except ValueError:
            logger.warning(
                f"Agent {self.agent.id} received unsupported message type: {type(message)}",
            )

    @message_handler
    async def handle_oob(
        self,
        message: ManagerMessage | ManagerRequest | ConductorRequest,
        ctx: MessageContext,
    ) -> ManagerMessage | ManagerRequest | AgentOutput | None:
        """Handle out-of-band control messages.

        Args:
            message: The control message to process
            ctx: Message context with sender information

        Returns:
            Optional response to the control message

        """
        """Control messages do not get broadcast around."""
        return await self.agent.handle_control_message(message)

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

        if isinstance(self.agent, (UIAgent, HostAgent)):
            return input_callback
        return None
