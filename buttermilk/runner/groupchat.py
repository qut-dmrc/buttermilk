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
from buttermilk._core.agent import ConductorRequest, FatalError, ProcessingError
from buttermilk._core.contract import (
    CLOSURE,
    CONDUCTOR,
    CONFIRM,
    END,
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

    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(
            type=f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}",
        ),
    )

    async def _setup(self):
        """Initialize the autogen runtime and register agents"""
        self._runtime = SingleThreadedAgentRuntime()

        # Register agents for each step
        await self._register_agents()
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

    async def _execute_step(
        self,
        step: AgentInput,
    ) -> AgentOutput | None:
        topic_id = DefaultTopicId(type=step.role)
        await self._runtime.publish_message(step, topic_id=topic_id)
        return None

    async def _get_next_step(self) -> StepRequest | None:
        """Determine the next step based on the current flow data."""

        # Each step, we proceed by asking the CONDUCTOR agent what to do.
        request = ConductorRequest(
            role=self.flow_name,
            inputs={"participants": dict(self._agent_types.items()), "task": self.params.get("task")},
        )
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        # Determine the next step based on the response
        if len(responses) != 1 or not (instructions := responses[0].outputs) or not (isinstance(instructions, StepRequest)):
            raise ProcessingError("Conductor could not get next step.")

        next_step = instructions.role
        if next_step == END:
            raise StopAsyncIteration("Host signaled that flow has been completed.")

        if next_step.lower() not in self._agent_types:
            raise ProcessingError(
                f"Step {next_step} not found in registered agents.",
            )

        # We're going to wait at least 10 seconds between steps.
        await asyncio.sleep(10)
        return instructions

    async def _run(self, request: StepRequest | None = None) -> None:
        """Main execution method that sets up agents and manages the flow.

        By default, this runs through a sequence of pre-defined steps.
        """
        try:
            await self._setup()
            if request:
                step = await self._prepare_step(request)
                await self._execute_step(step)
                # we haven't started yet, so we're going to send a completion through manually
                # this code shouldn't be here, it's autogen specific -- should be in groupchat.py
                await asyncio.sleep(5)
                await self._runtime.publish_message(
                    (TaskProcessingComplete(agent_id=step.role, role=step.role, task_index=-1, more_tasks_remain=False)), topic_id=self._topic
                )
            while True:
                try:
                    # Loop until we receive an error
                    await asyncio.sleep(1)

                    # # Get next step in the flow
                    request = await self._get_next_step()

                    if not await self._in_the_loop(request):
                        # User did not confirm plan; go back and get new instructions
                        continue

                    if request:
                        step = await self._prepare_step(request)
                        await self._execute_step(step)

                except ProcessingError as e:
                    # non-fatal error
                    logger.error(f"Error in Orchestrator run: {e}")
                    continue
                except (StopAsyncIteration, KeyboardInterrupt):
                    raise
                except FatalError:
                    raise
                except Exception as e:  # This is only here for debugging for now.
                    logger.exception(f"Error in Orchestrator.run: {e}")
                    raise FatalError from e

        except (StopAsyncIteration, KeyboardInterrupt):
            logger.info("Orchestrator.run: Flow completed.")
        except FatalError as e:
            logger.exception(f"Error in Orchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()
