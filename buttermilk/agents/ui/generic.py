import asyncio
from distutils.util import strtobool
from typing import Any

from pydantic import PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    ManagerMessage,
    UserConfirm,
)


class UIAgent(Agent):
    _input_task: asyncio.Task
    _input_callback: Any = PrivateAttr(...)

    _trace_this = False

    async def handle_control_message(
        self,
        message: ManagerMessage | UserConfirm,
    ) -> ManagerMessage | UserConfirm:
        """Ask the user for confirmation."""
        result = await self._process(
            input_data=AgentInput(
                agent_id=self.name,
                content=message.content,
            ),
        )
        if isinstance(message, UserConfirm):
            try:
                confirm = bool(strtobool(result.content))
            except ValueError:
                confirm = False

            return UserConfirm(confirm=confirm, **result.model_dump())
        if result:
            return ManagerMessage(**result.model_dump())
        return ManagerMessage()

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def cleanup(self) -> None:
        """Clean up resources"""
