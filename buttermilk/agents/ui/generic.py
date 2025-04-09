import asyncio
from typing import Any, AsyncGenerator

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

    async def _process(
        self,
        message: FlowMessage,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput|None, None]:
        """Send or receive input from the UI."""
        raise NotImplementedError
        yield # Required for async generator typing

    # async def handle_control_message(
    #     self,
    #     message: OOBMessages, **kwargs):
    #     """Process control messages for agent coordination.

    #     Args:
    #         message: The control message to process

    #     Returns:
    #         Optional response to the control message

    #     Agents generally do not listen in to control messages,
    #     but user interfaces do.

    #     """
    #     # Ask the user for confirmation
    #     await self.listen(message, **kwargs)
    #     async for _ in self._process(message):
    #         pass
    #     return

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def cleanup(self) -> None:
        """Clean up resources"""
