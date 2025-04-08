import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from autogen_core import ClosureAgent, ClosureContext, MessageContext, TypeSubscription

from buttermilk._core.contract import (
    CLOSURE,
    CONDUCTOR,
    CONFIRM,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk.bm import logger
from buttermilk.runner.groupchat import AutogenOrchestrator


class Selector(AutogenOrchestrator):

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
        self._next_step = None

        # store the last message received, so that any changes in instructions
        # are incorporated before executing the next step
        _last_message = self._last_message

        # Each step, we proceed by asking the CONDUCTOR agent what to do.
        participants = "\n".join([f"- {id}: {step.description}" for id, step in self.agents.items()])
        request = ConductorRequest(
            source="Selector",
            role=self.flow_name,
            inputs={"participants": participants, "task": self.params.get("task")},
        )
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        if len(responses) > 1:
            raise ProcessingError("Conductor returned multiple responses.")

        instructions = responses[0]

        # TODO(NS): Add finish condition
        # return

        # Determine the next step based on the response
        if not instructions or not (next_step := instructions.outputs.get("role")):
            raise ProcessingError("Next step not found from conductor.")

        if next_step not in self._agent_types:
            raise ProcessingError(
                f"Step {next_step} not found in registered agents.",
            )

        if self._last_message == _last_message:
            # No change to inputs
            yield StepRequest(
                role=next_step,
                source=self.flow_name,
                prompt=instructions.outputs.pop("prompt", ""),
                description=instructions.outputs.pop("plan", ""),
                tool=instructions.outputs.get("tool", None),
                arguments=instructions.outputs,
            )
        # wait a bit and go around again
        await asyncio.sleep(10)

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
            message: GroupchatMessageTypes,
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
                        source = str(ctx.sender.type) if ctx and ctx.sender else message.source

                        source = source.split(
                            "-",
                            1,
                        )[0]

                    if message.outputs:
                        self._flow_data.add(key=source, value=message)

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

    async def run(self, request: Any = None) -> None:
        """Main execution method that sets up agents and manages the flow"""
        try:
            # Setup autogen runtime environment
            await self._setup_runtime()
            await self._register_human_in_the_loop()

            # start the agents
            await self._runtime.publish_message(
                FlowMessage(source=self.flow_name, role="orchestrator"),
                topic_id=self._topic,
            )
            await asyncio.sleep(1)

            # First, introduce ourselves, and prompt the user for input
            await self._send_ui_message(
                ManagerRequest(
                    source=self.flow_name,
                    role="orchestrator",
                    content=f"Started {self.flow_name}: {self.description}. Please enter your question or prompt and let me know when you're ready to go.",
                ),
            )
            # Just wait for the first message to come through before doing anything else.
            await self._user_confirmation.get()

            while True:
                try:

                    await self._send_ui_message(
                        ManagerRequest(
                            source=self.flow_name,
                            role="orchestrator",
                            content=f"Shall I go ahead and determine the next step?",
                        ),
                    )
                    await asyncio.sleep(5)
                    # Just wait for the first message to come through before doing anything else.
                    while not await self._user_confirmation.get():
                        await asyncio.sleep(1)

                    # Get next step in the flow
                    step = await anext(self._get_next_step())

                    # For now, ALWAYS get confirmation from the user (MANAGER) role
                    confirm_step = ManagerRequest(
                        source=self.flow_name,
                        role="orchestrator",
                        content="Here's my proposed next step. Do you want to proceed?\n`" + str(step.description) + "`",
                        arguments=step.arguments,
                        prompt=step.prompt,
                        description=step.description,
                    )

                    await self._send_ui_message(confirm_step)
                    if not await self._user_confirmation.get():
                        # User did not confirm plan; go back and get new instructions
                        await asyncio.sleep(5)
                        continue
                    # Run next step
                    await self._execute_step(step)

                except StopAsyncIteration:
                    logger.info("SelectorOrchestrator.run: Flow completed.")
                    break
                except ProcessingError as e:
                    logger.error(f"Error in SelectorOrchestrator.run: {e}")
                    await self._send_ui_message(
                        ManagerRequest(
                            source=self.flow_name,
                            role="orchestrator",
                            content=f"Unable to get next step. Confirm to try again or enter another prompt.",
                        ),
                    )
                    await asyncio.sleep(5)
                    # Just wait for the first message to come through before doing anything else.
                    while not await self._user_confirmation.get():
                        await asyncio.sleep(1)
                except FatalError:
                    raise
                except Exception as e:  # This is only here for debugging for now.
                    logger.exception(f"Error in SelectorOrchestrator.run: {e}")

                await asyncio.sleep(0.1)

        except FatalError as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()
