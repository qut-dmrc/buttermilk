import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import shortuuid
from autogen_core import (
    AgentType,
    ClosureAgent,
    ClosureContext,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
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
)
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.agents.ui.console import UIAgent
from buttermilk.bm import bm, logger
from buttermilk.exceptions import ProcessingError

CONDUCTOR = "HOST"
MANAGER = "MANAGER"
CLOSURE = "COLLECTOR"


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
        self.topic_type = topic_type
        super().__init__(description=self.agent.description)

    @message_handler
    async def handle_request(self, message: AgentInput, ctx: MessageContext) -> AgentOutput:
        # Process using the wrapped agent
        source = str(ctx.sender) if ctx and ctx.sender else message.type
        agent_output = await self.agent(message, source=source)

        return agent_output

    @message_handler
    async def handle_output(self, message: AgentOutput, ctx: MessageContext) -> None:
        try:
            source = str(ctx.sender) if ctx and ctx.sender else message.type
            agent_output = await self.agent.receive_output(message, source=source)
            return
        except ValueError:
            logger.warning(
                f"Agent {self.agent.agent_id} received unsupported message type: {type(message)}",
            )

    @message_handler
    async def handle_oob(
        self,
        message: ManagerMessage,
        ctx: MessageContext,
    ) -> ManagerMessage | None:
        """Control messages between only User Interfaces and the Conductor."""
        if isinstance(self.agent, UIAgent):
            return await self.agent.confirm(message)
        return None


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _agents: dict[str, list[tuple[AgentType, AgentConfig]]] = PrivateAttr(
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

            next_step = await anext(self._step_generator)
            while True:
                await self._execute_step(
                    step_name=next_step["role"],
                    prompt=next_step.get("question", ""),
                )

                next_step = await anext(self._step_generator)

                user_input = await self._ask_user(
                    question=f"Proceed with next step: {next_step}? Otherwise, please provide alternate instructions.",
                )
                if not all(ui.confirm for ui in user_input):
                    # User does not want to proceed
                    raise ProcessingError("User does not want to proceed.")
        except ProcessingError:
            logger.error("User does not want to proceed.")
            raise
        except Exception as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()

    async def _get_next_step(self) -> AsyncGenerator[dict[str, str]]:
        for step in self.steps:
            yield {
                "role": step.name,
                "question": "",
            }

    async def _setup_runtime(self):
        """Initialize the autogen runtime and register agents"""
        # loop = asyncio.get_running_loop()
        self._runtime = SingleThreadedAgentRuntime()

        await self._register_collectors()

        # Register agents for each step
        await self._register_agents()

        # Start the runtime
        self._runtime.start()

    async def _register_collectors(self) -> None:
        # Register a closure agent
        async def collect_result(
            _agent: ClosureContext,
            message: AgentMessages,
            ctx: MessageContext,
        ) -> None:
            # Process and collect responses
            if not message.error:
                if message.payload:
                    self._flow_data.add(key=str(_agent.id.type), value=message.payload)

                # Harvest any records
                if message.records:
                    self._records.extend(message.records)

                # Add to the shared history
                self._history.append(f"{_agent.id} {ctx.sender}: {message.content}")

        await ClosureAgent.register_closure(
            self._runtime,
            CLOSURE,
            collect_result,
            subscriptions=lambda: [
                TypeSubscription(
                    topic_type=self._topic_type,
                    agent_type=CLOSURE,
                ),
            ],
        )

    async def _register_agents(self) -> None:
        """Register all agent variants for each step"""
        for step in self.steps:
            step_agents = []
            for agent_cls, variant in step.get_configs():
                # Register the agent with the runtime
                agent_type: AgentType = await AutogenAgentAdapter.register(
                    self._runtime,
                    variant.agent_id,
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
                        topic_type=step.name,
                        agent_type=agent_type,
                    ),
                )

                step_agents.append((agent_type, variant))
            # Store the registered agents for this step
            self._agents[step.name] = step_agents

    async def _ask_user(self, question: str = "") -> list[ManagerMessage]:
        """Ask user for input"""
        tasks = []
        for agent_type, _ in self._agents[MANAGER]:
            message = ManagerMessage(content=question)
            agent_id = await self._runtime.get(agent_type)
            task = self._runtime.send_message(
                message=message,
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
    ) -> list[FlowMessage]:
        """Execute a step by sending requests to relevant agents and collecting responses"""
        if step_name not in self._agents or len(self._agents[step_name]) == 0:
            raise ProcessingError(f"No agents registered for step {step_name}")

        tasks = []
        config = None
        for config in self.steps:
            if step.name == step_name:
                break
        if not config:
            raise ProcessingError(f"Cannot find config for step {step_name}.")

        # Send message with appropriate inputs for this step
        mapped_inputs = await self._prepare_inputs(config=config)
        message = AgentInput(
                content=prompt,
                payload=mapped_inputs,
        )
        topic_id = await self._runtime.get(self._topic_type)
        await self._runtime.publish_message(message, topic_id=topic_id)

        # Wait for all agents to respond
        responses = await asyncio.gather(*tasks)

        return responses

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            await self._runtime.stop_when_idle()
            await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")
