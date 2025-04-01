import asyncio
from typing import Any

from pydantic import PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    ManagerMessage,
    ManagerRequest,
)


class UIAgent(Agent):
    _input_task: asyncio.Task
    _input_callback: Any = PrivateAttr(...)

    _trace_this = False

    async def _request_user_input(self, message: ManagerRequest, **kwargs) -> str:
        """Get user input from the UI"""
        raise NotImplementedError

    async def handle_control_message(
        self,
        message: ManagerMessage | ManagerRequest,
    ) -> ManagerMessage | ManagerRequest | None:
        """Process control messages for agent coordination.

        Args:
            message: The control message to process

        Returns:
            Optional response to the control message

        Agents generally do not listen in to control messages,
        but user interfaces do.

        """
        # Ask the user for confirmation
        if isinstance(message, (ManagerRequest, AgentInput)):
            await self._request_user_input(message)
            return None
        if isinstance(message, ManagerMessage):
            # Just output these messages to the UI
            await self.receive_output(message)
            return None
        raise ValueError(f"Unknown message type: {type(message)}")

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def cleanup(self) -> None:
        """Clean up resources"""
