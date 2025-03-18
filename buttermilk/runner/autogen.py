import asyncio
from typing import Any, Dict, List

import shortuuid
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.model_context import UnboundedChatCompletionContext
from pydantic import Field, PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentMessages, AgentOutput
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.ui import IOInterface
from buttermilk.bm import logger


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
        return AgentOutput(agent = self.agent.agent_id, error="Unsupported message type")

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
    _agents: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    _topic_type: str = PrivateAttr()
    _user_agent_id: AgentId = PrivateAttr()
    _conductor_id: str = PrivateAttr()

    # Additional configuration
    max_wait_time: int = Field(default=300, description="Maximum time to wait for agent responses in seconds")
    _topic_type = PrivateAttr(default_factory=lambda: f"flow-{shortuuid.uuid()[:8]}")

    async def run(self, request=None) -> None:
        """Main execution method that sets up agents and manages the flow"""

        try:
            # Setup autogen runtime environment
            await self._setup_runtime()

            # Get initial input if not provided
            prompt = request.get("prompt") if request else ""
            if not prompt:
                prompt = await self.interface.get_input("Enter your prompt or request:")

            # Process each step in the flow
            for step_index, step in enumerate(self.steps):
                # Create confirmation message for this step
                confirm_message = f"Ready to proceed with step #{step_index} '{step.name}'? (y/n)"
                if not await self.interface.confirm(confirm_message):
                    logger.info(f"User cancelled at step '{step.name}'")
                    break

                # Collect inputs for this step
                mapped_inputs = self._flow_data._resolve_mappings(step.inputs)

                # Execute the step through registered agents
                step_inputs = AgentInput(prompt=prompt, inputs=mapped_inputs, records=self._records)
                await self._execute_step(step.name, step_inputs)
                prompt = None
            # Clean up resources
            await self._cleanup()

        except Exception as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")

        return

    async def _setup_runtime(self):
        """Initialize the autogen runtime and register agents"""
        # Start the runtime
        self._runtime.start()

        # Register the interface as the user agent
        user_agent_type = await AutogenIOAdapter.register(
            self._runtime,
            "User",
            lambda: AutogenIOAdapter(
                topic_type=self._topic_type, interface=self.interface
            )
        )
        self._user_agent_id = await self._runtime.get(user_agent_type)

        # Add subscription for the user agent
        await self._runtime.add_subscription(
            TypeSubscription(
                topic_type=self._topic_type,
                agent_type=user_agent_type,
            )
        )

        # Register agents for each step
        await self._register_agents()

    async def _register_agents(self):
        """Register all agent variants for a specific step"""
        step_agents = []

        for step in self.steps:
            for agent_cls, variant in step.get_configs():

                # Register the agent with the runtime
                agent_type = await AutogenAgentAdapter.register(
                    self._runtime,
                    variant["agent_id"],
                    lambda v=variant, cls=agent_cls: AutogenAgentAdapter(
                        agent_cfg=v, agent_cls=cls,
                        topic_type=self._topic_type,
                    )
                )

                # Add subscription for this agent
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=self._topic_type,
                        agent_type=agent_type,
                    )
                )

                # Also subscribe to a step-specific topic
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=step.agent_id,
                        agent_type=agent_type,
                    )
                )

                step_agents.append(agent_type)

            # Store the registered agents for this step
            self._agents[step.name] = step_agents
        return step_agents

    async def _execute_step(self, step_name: str, inputs: AgentInput) -> None:
        """Execute a step by sending requests to relevant agents and collecting responses"""
        if step_name not in self._agents:
            logger.warning(f"No agents registered for step {step_name}")
            return None

        results = []
        tasks = []

        # Get the chat context
        context = await self._context.get_messages()

        # Send message to each agent for this step
        for agent_type in self._agents[step_name]:
            agent_id = await self._runtime.get(agent_type)

            # Create task for sending message
            task = self._runtime.send_message(
                message=inputs,
                recipient=agent_id,
            )
            tasks.append(task)

        # Wait for all agents to respond
        if tasks:
            responses = await asyncio.gather(*tasks)

            # Process and collect responses
            for result in responses:
                if result and not result.error:
                    await self.store_results(step=step_name, result=result)
                    await self.interface.send_output(result)

        return

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            await self._runtime.stop_when_idle()
            await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")
