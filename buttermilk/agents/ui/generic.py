import asyncio
from collections.abc import Awaitable
from typing import Any, AsyncGenerator, Callable, Coroutine

from autogen_core import CancellationToken
from pydantic import PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    FlowMessage,
    ManagerMessage,
    ManagerRequest,OOBMessages
)


class UIAgent(Agent):
    _input_task: asyncio.Task
    _input_callback: Any = PrivateAttr(...)
    _trace_this = False

    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentOutput | None:
        """Send or receive input from the UI."""
        raise NotImplementedError

    async def _handle_control_message(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> OOBMessages | None:
        """Process control messages for agent coordination.

        Args:
            message: The control message to process

        Returns:
            Optional response to the control message

        Agents generally do not listen in to control messages,
        but user interfaces do.

        """
        #     # Ask the user for confirmation
        #     await self.listen(message, **kwargs)
        #     async for _ in self._process(message):
        #         pass
        #     return
        return None

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def cleanup(self) -> None:
        """Clean up resources"""
