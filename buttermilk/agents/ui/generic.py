import asyncio
from typing import Any

from pydantic import PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    UserRequest,
)


class UIAgent(Agent):
    _input_task: asyncio.Task
    _input_callback: Any = PrivateAttr(...)

    _trace_this = False

    async def _get_user_input(self, message: UserRequest, **kwargs) -> None:
        """Get user input from the UI"""
        raise NotImplementedError

    async def handle_control_message(
        self,
        message: UserRequest,
    ) -> None:
        """Ask the user for confirmation."""
        if isinstance(message, UserRequest):
            await self._get_user_input(message)
        raise ValueError(f"Unknown message type: {type(message)}")

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def cleanup(self) -> None:
        """Clean up resources"""
