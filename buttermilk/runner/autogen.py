import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Self

import pydantic
import shortuuid
from autogen_core import (
    AgentType,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from pydantic import Field, PrivateAttr

from buttermilk._core.agent import Agent, AgentConfig
from buttermilk._core.contract import (
    CLOSURE,
    CONDUCTOR,
    CONFIRM,
    MANAGER,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    FlowMessage,
    GroupchatMessages,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
    UserInstructions,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.agents.flowcontrol.types import HostAgent
from buttermilk.agents.ui.console import UIAgent
from buttermilk.bm import bm, logger


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
                response = await self.agent.on_messages([message])
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


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _step_generator = PrivateAttr(default=None)
    _user_confirmation: asyncio.Queue
    _agent_types: dict = PrivateAttr(default={})  # mapping of agent types
    # Additional configuration
    max_wait_time: int = Field(
        default=300,
        description="Maximum time to wait for agent responses in seconds",
    )

    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(
            type=f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}",
        ),
    )
    _last_message: AgentOutput | None = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def open_queue(self) -> Self:
        self._user_confirmation = asyncio.Queue(maxsize=1)
        return self

    async def run(self, request: Any = None) -> None:
        """Main execution method that sets up agents and manages the flow"""
        try:
            # Setup autogen runtime environment
            await self._setup_runtime()
            self._step_generator = self._get_next_step()

            # start the agents
            await self._runtime.publish_message(
                FlowMessage(),
                topic_id=self._topic,
            )
            await asyncio.sleep(1)

            # First, introduce ourselves, and prompt the user for input
            await self._send_ui_message(
                ManagerRequest(
                    content=f"Started {self.flow_name}: {self.description}. Please enter your question or prompt and let me know when you're ready to go.",
                ),
            )
            if not await self._user_confirmation.get():
                await self._send_ui_message(
                    ManagerMessage(content="OK, shutting down thread."),
                )
                return

            while True:
                try:
                    # Get next step from our CONDUCTOR agent
                    step = await anext(self._step_generator)

                    # For now, ALWAYS get confirmation from the user (MANAGER) role
                    confirm_step = ManagerRequest(
                        content="Here's my proposed next step. Do you want to proceed?",
                        inputs=step.arguments,
                    )
                    confirm_step.inputs["prompt"] = step.prompt
                    confirm_step.inputs["description"] = step.description

                    await self._send_ui_message(confirm_step)
                    if not await self._user_confirmation.get():
                        # User did not confirm plan; go back and get new instructions
                        continue
                    # Run next step
                    await self._execute_step(step)

                except StopAsyncIteration:
                    logger.info("AutogenOrchestrator.run: Flow completed.")
                    break
                except ProcessingError as e:
                    logger.error(f"Error in AutogenOrchestrator.run: {e}")
                except FatalError:
                    raise
                except Exception as e:  # This is only here for debugging for now.
                    logger.exception(f"Error in AutogenOrchestrator.run: {e}")

                await asyncio.sleep(0.1)

        except FatalError as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()

    async def _get_next_step(self) -> AsyncGenerator[StepRequest, None]:
        """Determine the next step based on the current flow data.

        This generator yields a series of steps to be executed in sequence,
        with each step containing the role and prompt information.

        Yields:
            StepRequest: An object containing:
                - 'role' (str): The agent role/step name to execute
                - 'prompt' (str): The prompt text to send to the agent
                - Additional key-value pairs that might be needed for agent execution

        Example:
            >>> async for step in self._get_next_step():
            >>>     await self._execute_step(**step)

        """
        for step_name in self.agents.keys():
            yield StepRequest(role=step_name, source=self.flow_name)

    async def _setup_runtime(self):
        """Initialize the autogen runtime and register agents"""
        # loop = asyncio.get_running_loop()
        self._runtime = SingleThreadedAgentRuntime()

        await self._register_collectors()
        await self._register_human_in_the_loop()
        # Register agents for each step
        await self._register_agents()

        # Start the runtime
        self._runtime.start()

    async def _register_human_in_the_loop(self) -> None:
        """Register a human in the loop agent"""

        # Register a human in the loop agent
        async def user_confirm(
            _agent: ClosureContext,
            message: ManagerResponse,
            ctx: MessageContext,
        ) -> None:
            # Add confirmation signal to queue
            if isinstance(message, ManagerResponse):
                try:
                    self._user_confirmation.put_nowait(message.confirm)
                except asyncio.QueueFull:
                    logger.debug(
                        f"User confirmation queue is full. Discarding confirmation: {message.confirm}",
                    )
            # Ignore other messages right now.

        await ClosureAgent.register_closure(
            self._runtime,
            CONFIRM,
            user_confirm,
            subscriptions=lambda: [
                TypeSubscription(
                    topic_type=topic_type,
                    agent_type=CONFIRM,
                )
                # Subscribe to the general topic and all step topics.
                for topic_type in [self._topic.type] + list(self.agents.keys())
            ],
            unknown_type_policy="ignore",  # only react to appropriate messages
        )

    async def _register_collectors(self) -> None:
        # Collect data from groupchat messages
        async def collect_result(
            _agent: ClosureContext,
            message: GroupchatMessages,
            ctx: MessageContext,
        ) -> None:
            # Process and collect responses
            if not message.error:
                if isinstance(message, AgentOutput):
                    source = None
                    if ctx and ctx.sender:
                        try:
                            # get the step name from the list of agents if we can
                            source = [
                                k
                                for k, v in self._agent_types.items()
                                if any([a[0].type == ctx.sender.type for a in v])
                            ][0]
                        except Exception as e:  # noqa
                            logger.warning(
                                f"{self.flow_name} collector is relying on agent naming conventions to find source keys. Please look into this and try to fix.",
                            )
                    if not source:
                        source = (
                            str(ctx.sender.type)
                            if ctx and ctx.sender
                            else message.agent_id
                        )

                        source = source.split(
                            "-",
                            1,
                        )[0]

                    if message.outputs:
                        self._flow_data.add(key=source, value=message)

                # Add to the shared history
                if message.content:
                    self.history.append(f"{message._type}: {message.content}")
                # Harvest any records
                if isinstance(message, AgentOutput) and message.records:
                    self._records.extend(message.records)

        await ClosureAgent.register_closure(
            self._runtime,
            CLOSURE,
            collect_result,
            subscriptions=lambda: [
                TypeSubscription(
                    topic_type=topic_type,
                    agent_type=CLOSURE,
                )
                # Subscribe to the general topic and all step topics.
                for topic_type in [self._topic.type] + list(self.agents.keys())
            ],
            unknown_type_policy="ignore",  # only react to appropriate messages
        )

    async def _register_agents(self) -> None:
        """Register all agent variants for each step"""
        for step_name, step in self.agents.items():
            step_agent_type = []
            for agent_cls, variant in step.get_configs():
                agent_cfg = variant.model_dump()
                agent_cfg["id"] = f"{step_name}-{shortuuid.uuid()[:6]}"
                # Register the agent with the runtime
                agent_type: AgentType = await AutogenAgentAdapter.register(
                    self._runtime,
                    agent_cfg["id"],
                    lambda v=variant, cls=agent_cls: AutogenAgentAdapter(
                        agent_cfg=v,
                        agent_cls=cls,
                        topic_type=self._topic.type,
                    ),
                )
                # Add subscription for this agent
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=self._topic.type,
                        agent_type=agent_type,
                    ),
                )

                # Also subscribe to a step-specific topic
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=step_name,
                        agent_type=agent_type,
                    ),
                )
                logger.debug(
                    f"Registered agent {agent_type} with id {agent_cfg['id']}, subscribed to {self._topic.type} and {step_name}.",
                )

                step_agent_type.append((agent_type, variant))
            # Store the registered agents for this step
            self._agent_types[step_name] = step_agent_type

    async def _ask_agents(
        self,
        step_name: str,
        message: AgentInput,
    ) -> list[AgentOutput]:
        """Ask agent directly for input"""
        tasks = []
        input_message = message.model_copy()

        for agent_type, _ in self._agent_types[step_name]:
            agent_id = await self._runtime.get(agent_type)
            task = self._runtime.send_message(
                message=input_message,
                recipient=agent_id,
            )

            tasks.append(task)

        # Wait for all agents to respond
        responses = await asyncio.gather(*tasks)
        return responses

    async def _send_ui_message(self, message: ManagerMessage | ManagerRequest) -> None:
        """Send a message to the UI agent"""
        topic_id = DefaultTopicId(type=MANAGER)
        await self._runtime.publish_message(message, topic_id=topic_id)

    async def _execute_step(
        self,
        step: StepRequest,
    ) -> None:
        message = await self._prepare_step(step=step)
        topic_id = DefaultTopicId(type=step.role)
        await self._runtime.publish_message(message, topic_id=topic_id)

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            await self._runtime.stop_when_idle()
            await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")
