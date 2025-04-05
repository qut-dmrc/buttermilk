import asyncio
from typing import Any, Self

import pydantic
import shortuuid
from autogen_core import (
    AgentType,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from pydantic import Field, PrivateAttr

from buttermilk._core.contract import (
    CLOSURE,
    CONFIRM,
    MANAGER,
    AgentInput,
    AgentOutput,
    FlowMessage,
    GroupchatMessages,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
)
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.bm import bm, logger
from buttermilk.libs.autogen import AutogenAgentAdapter


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
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
        """Main execution method that sets up agents and manages the flow.

        By default, this just sends an initial message to spawn the agents
        and then loops until cancelled.
        """
        try:
            # Setup autogen runtime environment
            await self._setup_runtime()

            # start the agents
            await self._runtime.publish_message(
                FlowMessage(agent_id=self.flow_name, agent_role="orchestrator"),
                topic_id=self._topic,
            )
            if request:
                await self._runtime.publish_message(request, topic_id=self._topic)
            while True:
                await asyncio.sleep(1)
        except StopAsyncIteration:
            logger.info("AutogenOrchestrator.run: Flow completed.")
        finally:
            # Clean up resources
            await self._cleanup()

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
