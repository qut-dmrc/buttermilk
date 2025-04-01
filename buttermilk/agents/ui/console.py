import asyncio
from typing import Any

from aioconsole import ainput
from pydantic import PrivateAttr
from rich.console import Console
from rich.markdown import Markdown

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    GroupchatMessages,
    ManagerRequest,
    UserInstructions,
)
from buttermilk.agents.ui.generic import UIAgent


class CLIUserAgent(UIAgent):
    _input_callback: Any = PrivateAttr(...)

    async def receive_output(
        self,
        message: GroupchatMessages,
        **kwargs,
    ) -> None:
        """Send output to the user interface"""
        if isinstance(message, UserInstructions):
            return
        console = Console(highlight=True)
        console.print(Markdown(f"### {message.role}: \n{message.content}\n"))

    async def _process(
        self,
        input_data: AgentInput | ManagerRequest,
        **kwargs,
    ) -> None:
        """Request input from the user interface"""
        Console(highlight=True).print(
            Markdown(f"Input requested: {input_data.content}"),
        )

    async def _poll_input(
        self,
    ) -> None:
        """Continuously poll for user input in the background"""
        while True:
            try:
                user_input = await ainput()
                await self._input_callback(UserInstructions(content=user_input))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unable to poll input: {e}")
                raise

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface and start input polling"""
        self._input_callback = input_callback

        self._input_task = asyncio.create_task(self._poll_input())
