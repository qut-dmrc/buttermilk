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

    async def _setup(self) -> None:
        await super()._setup()
        await self._register_human_in_the_loop()  # First, introduce ourselves, and prompt the user for input

        # await self._register_collectors()
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
