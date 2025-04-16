import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Self

from autogen_core import ClosureAgent, ClosureContext, MessageContext, TypeSubscription
from pydantic import model_validator

from buttermilk._core.contract import (
    CLOSURE,
    CONDUCTOR,
    CONFIRM,
    END,
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
import time


class Selector(AutogenOrchestrator):
    _user_confirmation: asyncio.Queue

    @model_validator(mode="after")
    def open_queue(self) -> Self:
        self._user_confirmation = asyncio.Queue(maxsize=1)
        return self

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

        # Each step, we proceed by asking the CONDUCTOR agent what to do.
        participants = "\n".join([f"- {id}: {step.description}" for id, step in self.agents.items()])
        participants += f"\n - {END}: Conclude the conversation."

        request = ConductorRequest(
            role=self.flow_name,
            inputs={"participants": participants, "task": self.params.get("task")},
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

        yield instructions

    async def _setup(self) -> None:
        await super()._setup()
        await self._register_collectors()
        await self._register_human_in_the_loop()  # First, introduce ourselves, and prompt the user for input
        msg = ManagerMessage(
            role="orchestrator",
            content=f"Started {self.flow_name}: {self.description}. Please enter your question or prompt and let me know when you're ready to go.",
        )
        await self._runtime.publish_message(msg, topic_id=self._topic)

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

    async def _wait_for_human(self, timeout=240) -> bool:
        """Wait for human confirmation"""
        t0 = time.time()
        while True:
            try:
                msg = self._user_confirmation.get_nowait()
                if msg.halt:
                    raise StopAsyncIteration("User requested halt.")
                return msg.confirm
            except asyncio.QueueEmpty:
                if time.time() - t0 > timeout:
                    return False
                await asyncio.sleep(1)

    async def _in_the_loop(self, step: StepRequest) -> bool:
        """Send a message to the UI agent to confirm permission to run."""

        # For now, ALWAYS get confirmation from the user (MANAGER) role
        confirm_step = ManagerRequest(
            role="orchestrator",
            content=f"Here's my proposed next step:\n\n{str(step.description)}\n\n```{step.role}: {step.prompt}```\n\nDo you want to proceed?",
            prompt=step.prompt,
            description=step.description,
        )

        await self._send_ui_message(confirm_step)
        response = await self._wait_for_human()
        if not response:
            return False
        return True
