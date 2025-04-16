import asyncio
from typing import Any, AsyncGenerator, Self

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
    message_handler,
)
from pydantic import Field, PrivateAttr
import weave

from buttermilk._core import TaskProcessingComplete
from buttermilk._core.contract import (
    CLOSURE,
    CONFIRM,
    MANAGER,
    AgentInput,
    AgentOutput,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
    UserInstructions,
)
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.bm import bm, logger
from buttermilk.libs.autogen import AutogenAgentAdapter


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _agent_types: dict = PrivateAttr(default={})  # mapping of agent types
    # Additional configuration
    max_wait_time: int = Field(
        default=300,
        description="Maximum time to wait for agent responses in seconds",
    )
    completion_threshold_ratio: float = Field(
        default=0.8,
        description="Ratio of agents that must complete a step before proceeding (0.0 to 1.0)",
    )
    _current_step_name: str | None = PrivateAttr(default=None)
    _completed_agents_current_step: set[str] = PrivateAttr(default_factory=set)
    _expected_agents_current_step: set[str] = PrivateAttr(default_factory=set)

    _step_completion_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(
            type=f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}",
        ),
    )

    async def _setup(self):
        """Initialize the autogen runtime and register agents"""
        # loop = asyncio.get_running_loop()
        self._runtime = SingleThreadedAgentRuntime()

        # Register agents for each step
        await self._register_agents()
        await self._register_collectors()
        # Start the runtime
        self._runtime.start()

    async def _register_agents(self) -> None:
        """Register all agent variants for each step"""
        for step_name, step in self.agents.items():
            step_agent_type = []
            for agent_cls, variant in step.get_configs():
                # Register the agent with the runtime
                agent_type: AgentType = await AutogenAgentAdapter.register(
                    self._runtime,
                    variant.id,
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
                    f"Registered agent {agent_type} with id {variant.role}, subscribed to {self._topic.type} and {step_name}.",
                )

                step_agent_type.append((agent_type, variant))
            # Store the registered agents for this step
            self._agent_types[step_name.lower()] = step_agent_type

    async def _ask_agents(
        self,
        step_name: str,
        message: AgentInput|StepRequest,
    ) -> list[AgentOutput]:
        """Ask agent directly for input"""
        tasks = []
        input_message = message.model_copy()

        for agent_type, _ in self._agent_types[step_name.lower()]:
            agent_id = await self._runtime.get(agent_type)
            task = self._runtime.send_message(
                message=input_message,
                recipient=agent_id,
            )

            tasks.append(task)

        # Wait for all agents to respond
        responses = await asyncio.gather(*tasks)
        return [r for r in responses if r and isinstance(r, AgentOutput)]

    async def _send_ui_message(self, message: ManagerMessage | ManagerRequest) -> None:
        """Send a message to the UI agent"""
        topic_id = DefaultTopicId(type=MANAGER)
        await self._runtime.publish_message(message, topic_id=topic_id)

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            await self._runtime.stop_when_idle()
            await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")

    @message_handler
    async def _handle_completion_message(self, message: TaskProcessingComplete, context: MessageContext) -> None:
        """Handles messages indicating an agent has finished its task."""
        # Ensure message is relevant to the step we are waiting for
        if context.sender.type in self._expected_agents_current_step:
            self._completed_agents_current_step.add(context.sender.type)
            logger.debug(
                f"Agent {context.sender.type} completed step {self._current_step_name}. "
                f"Completed: {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)}"
            )

            required_completions = max(1, int(len(self._expected_agents_current_step) * self.completion_threshold_ratio))
            if len(self._completed_agents_current_step) >= required_completions:
                logger.info(f"Completion threshold reached for step '{self._current_step_name}'.")
                self._step_completion_event.set()  # Signal that enough agents are done

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
            # Reset completion tracking
            self._current_step_name = step_name
            self._completed_agents_current_step.clear()
            self._step_completion_event.clear()
            # Determine agents expected for *this* step based on its role/name
            self._expected_agents_current_step = {variant.id for agent_type, variant in self._agent_types.get(step_name, [])}

            yield StepRequest(role=step_name, description=f"Call {step_name}.")

            try:
                # Wait for enough completions, with a timeout
                await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout waiting for step completion. "
                    f"Proceeding with {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)} completed agents."
                )
            finally:
                self._step_completion_event.clear()  # Ensure event is clear for next wait
                self._current_step_name = None  # Clear current step after flow finishes

    async def _execute_step(
        self,
        step: AgentInput,
    ) -> AgentOutput | None:
        topic_id = DefaultTopicId(type=step.role)
        await self._runtime.publish_message(step, topic_id=topic_id)
        return None

    async def _register_collectors(self) -> None:
        async def _handle_completion_message(self, message: TaskProcessingComplete, context: MessageContext) -> None:
            """Handles messages indicating an agent has finished its task."""
            # Ensure message is relevant to the step we are waiting for
            if context.sender.type in self._expected_agents_current_step:
                self._completed_agents_current_step.add(context.sender.type)
                logger.debug(
                    f"Agent {context.sender.type} completed step {self._current_step_name}. "
                    f"Completed: {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)}"
                )

                required_completions = max(1, int(len(self._expected_agents_current_step) * self.completion_threshold_ratio))
                if len(self._completed_agents_current_step) >= required_completions:
                    logger.info(f"Completion threshold reached for step '{self._current_step_name}'.")
                    self._step_completion_event.set()  # Signal that enough agents are done

        await ClosureAgent.register_closure(
            self._runtime,
            CLOSURE,
            _handle_completion_message,
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
