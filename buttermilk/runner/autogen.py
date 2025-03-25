import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from distutils.util import strtobool
from typing import Any

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
    AgentInput,
    AgentMessages,
    AgentOutput,
    FlowMessage,
    ManagerMessage,
    UserConfirm,
    UserInput,
)
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.agents.flowcontrol.types import HostAgent
from buttermilk.agents.ui.console import UIAgent
from buttermilk.bm import bm, logger
from buttermilk.exceptions import ProcessingError

CONDUCTOR = "HOST"
MANAGER = "MANAGER"
CLOSURE = "COLLECTOR"
CONFIRM = "CONFIRM"


class AutogenAgentAdapter(RoutedAgent):
    """Adapter for Autogen runtime"""

    def __init__(
        self,
        topic_type: str,
        agent: Agent = None,
        agent_cls: type = None,
        agent_cfg: AgentConfig = None,
    ):
        if agent:
            self.agent = agent
        else:
            if not agent_cfg:
                raise ValueError("Either agent or agent_cfg must be provided")
            self.agent: Agent = agent_cls(**agent_cfg.model_dump())
        self.topic_id: TopicId = DefaultTopicId(type=topic_type)
        super().__init__(description=self.agent.description)

        # Take care of any initialization the agent needs to do in this event loop
        input_callback = self.handle_input()
        asyncio.create_task(self.agent.initialize(input_callback=input_callback))

    @message_handler
    async def handle_request(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> AgentOutput | None:
        # Process using the wrapped agent
        source = str(ctx.sender) if ctx and ctx.sender else message.type
        agent_output = await self.agent(message, source=source)
        if agent_output:
            await self._runtime.publish_message(
                agent_output,
                topic_id=self.topic_id,
                sender=self.id,
            )
            # give this a second to make sure messages are collected before proceeding.
            await asyncio.sleep(1)
            return agent_output
        return None

    @message_handler
    async def handle_output(
        self,
        message: AgentOutput | UserInput,
        ctx: MessageContext,
    ) -> None:
        try:
            source = str(ctx.sender) if ctx and ctx.sender else message.type
            # ignore messages sent by us
            if source != self.id:
                response = await self.agent.receive_output(message, source=source)
                if response:
                    await self._runtime.publish_message(
                        response,
                        topic_id=self.topic_id,
                        sender=self.id,
                    )
            return
        except ValueError:
            logger.warning(
                f"Agent {self.agent.id} received unsupported message type: {type(message)}",
            )

    @message_handler
    async def handle_oob(
        self,
        message: ManagerMessage | UserConfirm,
        ctx: MessageContext,
    ) -> ManagerMessage | UserConfirm | None:
        """Control messages between only User Interfaces and the Conductor."""
        if isinstance(self.agent, (UIAgent, HostAgent)):
            return await self.agent.handle_control_message(message)
        return None

    def handle_input(self) -> Callable[[str], Awaitable[None]]:
        async def input_callback(user_input: str) -> None:
            """Callback function to handle user input"""
            try:
                confirm = bool(strtobool(user_input))
                await self._runtime.publish_message(
                    UserConfirm(confirm=confirm),
                    topic_id=self.topic_id,
                    sender=self.id,
                )
                return
            except:
                # Not a simple yes / no, send as text.
                pass

            if user_input.strip() == "":
                # newline on console
                await self._runtime.publish_message(
                    UserConfirm(confirm=True),
                    topic_id=self.topic_id,
                    sender=self.id,
                )
            else:
                await self._runtime.publish_message(
                    UserInput(
                        content=user_input,
                    ),
                    topic_id=self.topic_id,
                    sender=self.id,
                )

        return input_callback


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _agents: dict[str, list[tuple[AgentType, AgentConfig]]] = PrivateAttr(
        default_factory=dict,
    )  # mapping of step to registered agents and their individual configs
    _step_generator = PrivateAttr(default=None)
    _user_confirmation: asyncio.Queue = PrivateAttr(default_factory=asyncio.Queue)
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
            while await self._user_confirmation.get():
                step = await anext(self._step_generator)
                await self._execute_step(step_name=step.pop("role"), **step)

                # send another confirmation message
                topic_id = DefaultTopicId(type=MANAGER)
                await self._runtime.publish_message(
                    AgentInput(
                        agent_id=CONDUCTOR,
                    ),
                    topic_id=topic_id,
                )
                await asyncio.sleep(0.1)

        except ProcessingError as e:
            logger.error(f"Error in AutogenOrchestrator.run: {e}")
        except Exception as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()

    async def _get_next_step(self) -> AsyncGenerator[dict[str, str]]:
        for step in self.steps:
            yield {
                "role": step.id,
                "prompt": "",
            }

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
            message: UserConfirm,
            ctx: MessageContext,
        ) -> None:
            # Add confirmation signal to queue
            if isinstance(message, UserConfirm):
                await self._user_confirmation.put(message.confirm)
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
                for topic_type in [self._topic.type] + [step.id for step in self.steps]
            ],
            unknown_type_policy="ignore",  # only react to appropriate messages
        )

    async def _register_collectors(self) -> None:
        # Register a closure agent
        async def collect_result(
            _agent: ClosureContext,
            message: AgentMessages | UserInput,
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
                                for k, v in self._agents.items()
                                if any([a[0].type == ctx.sender.type for a in v])
                            ][0]
                        except:
                            logger.warning(
                                f"{self.name} collector is relying on agent naming conventions to find source keys. Please look into this and try to fix.",
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
                    self._history.append(f"{message._type}: {message.content}")
                # Harvest any records
                if message.records:
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
                for topic_type in [self._topic.type] + [step.id for step in self.steps]
            ],
            unknown_type_policy="ignore",  # only react to appropriate messages
        )

    async def _register_agents(self) -> None:
        """Register all agent variants for each step"""
        for step in self.steps:
            step_agent_type = []
            for agent_cls, variant in step.get_configs():
                agent_cfg = variant.model_dump()
                agent_cfg["id"] = f"{step.id}-{shortuuid.uuid()[:6]}"
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
                        topic_type=step.id,
                        agent_type=agent_type,
                    ),
                )
                logger.debug(
                    f"Registered agent {agent_type} with id {agent_cfg['id']}, subscribed to {self._topic.type} and {step.id}.",
                )

                step_agent_type.append((agent_type, variant))
            # Store the registered agents for this step
            self._agents[step.id] = step_agent_type

    async def _ask_agents(
        self,
        step_name,
        message: FlowMessage,
    ) -> list[FlowMessage]:
        """Ask agent directly for input"""
        tasks = []
        # Send message with appropriate inputs for this agent
        input_message = message.model_copy()
        input_message.inputs = await self._prepare_inputs(step_name=step_name)
        input_message.inputs.update({"prompt": message.content})

        for agent_type, _ in self._agents[step_name]:
            agent_id = await self._runtime.get(agent_type)
            task = self._runtime.send_message(
                message=input_message,
                recipient=agent_id,
            )

            tasks.append(task)

        # Wait for all agents to respond
        responses = await asyncio.gather(*tasks)
        return responses

    async def _execute_step(
        self,
        step_name: str,
        prompt: str = "",
        **inputs,
    ) -> None:
        message = await self._prepare_step_message(step_name, prompt, **inputs)
        topic_id = DefaultTopicId(type=step_name)
        await self._runtime.publish_message(message, topic_id=topic_id)
        # give this a second to make sure messages are collected before proceeding.
        await asyncio.sleep(1)

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            await self._runtime.stop_when_idle()
            await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")
