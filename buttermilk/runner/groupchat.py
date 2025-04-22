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
from pydantic import Field, PrivateAttr, model_validator
import weave

from buttermilk._core import TaskProcessingComplete
from buttermilk._core.agent import Agent, ConductorRequest, FatalError, ProcessingError  # Added Agent
from buttermilk._core.contract import (
    CLOSURE,
    CONDUCTOR,
    # Added QualScore
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
    # Removed QualScore from here
)
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.types import Record, RunRequest  # Added Record
from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore  # Added QualScore import here
from buttermilk.agents.fetch import FetchRecord
from buttermilk.bm import bm, logger
from buttermilk.libs.autogen import AutogenAgentAdapter


class AutogenOrchestrator(Orchestrator):
    """Orchestrator that uses Autogen's routing and messaging system"""

    # Private attributes
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _agent_types: dict = PrivateAttr(default={})  # mapping of agent types
    _user_confirmation: asyncio.Queue[ManagerResponse] = PrivateAttr()

    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(
            type=f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}",
        ),
    )

    @model_validator(mode="after")
    def open_queue(self) -> Self:
        """Initialize user confirmation queue."""
        self._user_confirmation = asyncio.Queue(maxsize=1)
        return self

    async def _setup(self):
        """Initialize the autogen runtime and register agents"""
        self._runtime = SingleThreadedAgentRuntime()

        # Register agents for each step
        await self._register_agents()

        # Create an agent to interact with the user, if any
        await self._register_human_in_the_loop()

        # Start the runtime
        self._runtime.start()

    async def _register_agents(self) -> None:
        """Register all agent variants for each step"""
        for step_name, step in self.agents.items():
            step_agent_type = []
            for agent_cls, variant in step.get_configs():
                # We register the *Adapter* class.
                # Create a proper factory function that returns an adapter instance
                # with the correct parameters following autogen_core.BaseAgent.register signature
                def create_adapter_instance(cfg=variant, cls=agent_cls):
                    return AutogenAgentAdapter(
                        agent_cfg=cfg,
                        agent_cls=cls,
                        topic_type=self._topic.type,
                    )

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
                        topic_type=step_name.upper(),
                        agent_type=agent_type,
                    ),
                )
                logger.debug(
                    f"Registered agent {agent_type} with id {variant.role}, subscribed to {self._topic.type} and {step_name}.",
                )

                step_agent_type.append((agent_type, variant))
            # Store the registered agents for this step
            self._agent_types[step_name.upper()] = step_agent_type

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
                    self._user_confirmation.put_nowait(message)
                except asyncio.QueueFull:
                    logger.debug(
                        f"User confirmation queue is full. Discarding confirmation: {message}",
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

    async def _ask_agents(
        self,
        step_name: str,
        message: AgentInput | ConductorRequest | StepRequest,
    ) -> list[AgentOutput]:
        """Ask agent directly for input"""
        tasks = []
        input_message = message.model_copy()

        for agent_type, _ in self._agent_types[step_name.upper()]:
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
        logger.debug(f"Publishing UI message of type {type(message).__name__} to MANAGER topic")
        await self._runtime.publish_message(message, topic_id=topic_id)

        # Also publish to the main topic to ensure visibility
        logger.debug(f"Publishing UI message of type {type(message).__name__} to main topic {self._topic.type}")
        await self._runtime.publish_message(message, topic_id=self._topic)

    async def _cleanup(self):
        """Clean up resources when flow is complete"""
        try:
            # Stop the runtime
            if self._runtime._run_context:
                # runtime is started
                await self._runtime.stop_when_idle()
                await asyncio.sleep(2)  # Give it some time to properly shut down
        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")

    async def _execute_step(
        self,
        step: StepRequest,
    ):

        try:
            message = AgentInput(prompt=step.prompt, records=self._records)
            _ = await self._ask_agents(
                step.role,
                message=message,
            )
            await asyncio.sleep(0.1)

        except Exception as e:
            msg = f"Error during step execution: {e}", exc_info=False
            raise ProcessingError(msg) from e

    async def _get_host_suggestion(self) -> StepRequest | None:  # Allow None return
        """Determine the next step based on the current flow data."""

        # Each step, we proceed by asking the CONDUCTOR agent what to do.
        conductor_inputs = {"participants": dict(self._agent_types.items())}
        request = ConductorRequest(inputs=conductor_inputs, prompt=self.params.get("task", ""), records=self._records)
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        # Determine the next step based on the response
        valid_responses = [r for r in responses if isinstance(r, AgentOutput) and not r.is_error and isinstance(r.outputs, StepRequest)]

        if not valid_responses:
            logger.warning("Conductor did not return a valid StepRequest.")
            return None

        # Use the first valid response
        # Add assertion for type checker clarity
        assert isinstance(valid_responses[0].outputs, StepRequest), f"Expected StepRequest, got {type(valid_responses[0].outputs)}"
        instructions = valid_responses[0].outputs

        # We're going to wait a bit between steps.
        await asyncio.sleep(5)
        return instructions

    async def _run(self, request: RunRequest | None = None) -> None:
        """Main execution method that sets up agents and manages the flow.

        By default, this runs through a sequence of pre-defined steps.
        """
        try:
            await self._setup()
            if request:
                # Fixed: Pass list(self.data)
                fetch = FetchRecord(data=list(self.data))
                fetch_output = await fetch._run(record_id=request.record_id, uri=request.uri, prompt=request.prompt)
                # Fixed: Extract results list
                if fetch_output and fetch_output.results:
                    self._records = fetch_output.results

            while True:
                try:
                    # Loop until we receive an error
                    await asyncio.sleep(1)

                    # # Get next step in the flow
                    if not (step := await self._get_host_suggestion()):
                        # No next step at the moment; wait and try a bit
                        await asyncio.sleep(10)
                        continue

                    if step.role == END:
                        raise StopAsyncIteration("Host signaled that flow has been completed.")

                    if not await self._in_the_loop(step):
                        # User did not confirm plan; go back and get new instructions
                        continue

                    await self._execute_step(step=step)

                    # sleep a bit
                    await asyncio.sleep(5)

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
            logger.error(f"Error in Orchestrator.run: {e}")
        except Exception as e:
            logger.exception(f"Error in Orchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()
