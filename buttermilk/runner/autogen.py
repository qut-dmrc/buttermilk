import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import shortuuid
from autogen_core import (
    AgentType,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.model_context import UnboundedChatCompletionContext
from pydantic import Field, PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    AgentMessages,
    AgentOutput,
    FlowMessage,
)
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.ui import IOInterface
from buttermilk.bm import logger
from buttermilk.exceptions import ProcessingError


class AutogenAgentAdapter(RoutedAgent):
    """Adapter for Autogen runtime"""

    def __init__(self, topic_type: str, agent: Agent = None, agent_cls: type = None, agent_cfg: dict = {}):
        if agent:
            self.agent = agent
        else:
            self.agent: Agent = agent_cls(**agent_cfg)
        self.topic_type = topic_type
        super().__init__(description=self.agent.description)

    @message_handler
    async def handle_request(self, message: AgentInput, ctx: MessageContext) -> AgentOutput:
        # Process using the wrapped agent
        agent_output = await self.agent(message)

        return agent_output

    async def handle_message(self, message: Any, **kwargs) -> AgentOutput:
        """Alternative entry point for non-decorated handling"""
        if isinstance(message, AgentInput):
            return await self.handle_request(message, kwargs.get("ctx"))
        return AgentOutput(agent=self.agent.agent_id, error="Unsupported message type")


class AutogenIOAdapter(RoutedAgent):
    def __init__(self, topic_type: str, interface: IOInterface, description="UserProxy"):
        self.interface: IOInterface = interface
        self.topic_type = topic_type
        super().__init__(description=description)

    @message_handler
    async def handle_request(self, message: AgentInput, ctx: MessageContext) -> AgentOutput:
        # Request input from the UI
        user_input = await self.interface.get_input(message.content, source=ctx.sender.type if ctx.sender else self.id.type)

        return AgentOutput(content=user_input, agent=self.id.type)

    @message_handler
    async def handle_message(self, message: AgentMessages, ctx: MessageContext) -> None:
        # Send to the UI
        await self.interface.send_output(message,
                source=ctx.sender.type if ctx.sender else self.id.type)


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr(default_factory=SingleThreadedAgentRuntime)
    _context: UnboundedChatCompletionContext = PrivateAttr(default_factory=UnboundedChatCompletionContext)
    _agents: dict[str, list[tuple[AgentType, dict]]] = PrivateAttr(
        default_factory=dict,
    )  # mapping of step to registered agents and their individual configs
    _step_generator = PrivateAttr(default=None)

    # Additional configuration
    max_wait_time: int = Field(
        default=300,
        description="Maximum time to wait for agent responses in seconds",
    )

    _topic_type: str = PrivateAttr(
        default_factory=lambda: f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}",
    )
    _last_message: AgentOutput | None = PrivateAttr(default=None)

    async def run(self, request: Any = None) -> None:
        """Main execution method that sets up agents and manages the flow"""
        try:
            # Setup autogen runtime environment
            await self._setup_runtime()

            # Initialize the generator once
            self._step_generator = self._get_next_step()

            next_step = await self._get_next_step()
            next_step = await anext(generator_object)

            while await self.interface.confirm(
                f"Proceed with next step: {next_step}? Otherwise, please provide alternate instructions.",
            ):
                await self._execute_step(next_step)

                next_step = await self._get_next_step()
            # Clean up resources
            await self._cleanup()

        except Exception as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")

    async def _get_next_step(self) -> AsyncGenerator[dict[str, str]]:
        for step in self.steps:
            yield {
                "role": step.name,
                "question": "",
            }

    async def _setup_runtime(self):
        """Initialize the autogen runtime and register agents"""
        # Start the runtime
        self._runtime.start()

        # Register the interface as the user agent
        user_agent_type = await AutogenIOAdapter.register(
            self._runtime,
            "User",
            lambda: AutogenIOAdapter(
                topic_type=self._topic_type,
                interface=self.interface,
            ),
        )
        self._user_agent_id = await self._runtime.get(user_agent_type)

        # Add subscription for the user agent
        await self._runtime.add_subscription(
            TypeSubscription(
                topic_type=self._topic_type,
                agent_type=user_agent_type,
            ),
        )

        # Register agents for each step
        await self._register_agents()

    async def _register_agents(self) -> None:
        """Register all agent variants for a specific step"""
        step_agents = []

        for step in self.steps:
            for agent_cls, variant in step.get_configs():

                # Register the agent with the runtime
                agent_type: AgentType = await AutogenAgentAdapter.register(
                    self._runtime,
                    variant["agent_id"],
                    lambda v=variant, cls=agent_cls: AutogenAgentAdapter(
                        agent_cfg=v,
                        agent_cls=cls,
                        topic_type=self._topic_type,
                    ),
                )

                # Add subscription for this agent
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=self._topic_type,
                        agent_type=agent_type,
                    ),
                )

                # Also subscribe to a step-specific topic
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=step.agent_id,
                        agent_type=agent_type,
                    ),
                )

                step_agents.append((agent_type, variant))

            # Store the registered agents for this step
            self._agents[step.name] = step_agents

    async def _execute_step(
        self,
        step_name: str,
        prompt: str = "",
    ) -> list[FlowMessage]:
        """Execute a step by sending requests to relevant agents and collecting responses"""
        if step_name not in self._agents or len(self._agents[step_name]) == 0:
            raise ProcessingError(f"No agents registered for step {step_name}")

        tasks = []
        # Get the chat context and records
        context = await self._context.get_messages()
        records = self._flow_data.get("records")

        # Send message to each agent for this step
        for agent_type, config in self._agents[step_name]:
            # prepare the step inputs
            mapped_inputs = self._flow_data._resolve_mappings(config["inputs"])
            agent_id = await self._runtime.get(agent_type)

            message = AgentInput(
                prompt=prompt,
                inputs=mapped_inputs,
                records=records,
                context=context,
            )
            # Create task for sending message
            task = self._runtime.send_message(
                message=message,
                recipient=agent_id,
            )
            tasks.append(task)

        # Wait for all agents to respond
        responses = await asyncio.gather(*tasks)

        # Process and collect responses
        for result in responses:
            if result and not result.error:
                await self.store_results(step=step_name, result=result)
                await self.interface.send_output(result)

        return responses

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            await self._runtime.stop_when_idle()
            await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")
