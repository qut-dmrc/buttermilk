import asyncio
from typing import Any

from pydantic import PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    ManagerMessage,
    UserRequest,
    UserResponse,
)


class UIAgent(Agent):
    _input_task: asyncio.Task
    _input_callback: Any = PrivateAttr(...)

    _trace_this = False

    async def _get_user_input(self, message: UserRequest, **kwargs) -> UserResponse:
        """Get user input from the UI"""
        raise NotImplementedError

    async def handle_control_message(
        self,
        message: UserRequest,
    ) -> ManagerMessage | UserResponse:
        """Ask the user for confirmation."""
        if isinstance(message, UserRequest):
            result = await self._get_user_input(message)
            return result
        raise ValueError(f"Unknown message type: {type(message)}")

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def cleanup(self) -> None:
        """Clean up resources"""
