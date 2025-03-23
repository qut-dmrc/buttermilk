import asyncio
from distutils.util import strtobool
from typing import Any

from aioconsole import ainput
from pydantic import PrivateAttr
from rich.console import Console
from rich.markdown import Markdown

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    AgentMessages,
    ManagerMessage,
    UserConfirm,
)


class UIAgent(Agent):
    _input_task: asyncio.Task
    _input_callback: Any = PrivateAttr(...)

    def __init__(self, input_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if input_callback:
            self._input_callback = input_callback
            asyncio.create_task(
                self.initialize(
                    input_callback=input_callback,
                ),
            )

    async def handle_control_message(
        self,
        message: ManagerMessage | UserConfirm,
    ) -> ManagerMessage | UserConfirm:
        """Ask the user for confirmation."""
        result = await self.process(
            input_data=AgentInput(
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

    async def initialize(self, **kwargs) -> None:
        """Initialize the interface"""

    async def cleanup(self) -> None:
        """Clean up resources"""


class CLIUserAgent(UIAgent):
    _input_callback: Any = PrivateAttr(...)

    async def receive_output(
        self,
        message: AgentMessages,
        source: str,
        **kwargs,
    ) -> None:
        """Send output to the user interface"""
        console = Console(highlight=True)
        console.print(Markdown(f"### {source}: \n{message.content}\n"))

    async def process(
        self,
        input_data: AgentMessages,
        **kwargs,
    ) -> AgentMessages | None:
        """Request input from the user interface"""
        Console(highlight=True).print(
            Markdown(f"Input requested: {input_data.content}"),
        )
        return None

    async def _poll_input(
        self,
    ) -> None:
        """Continuously poll for user input in the background"""
        while True:
            try:
                user_input = await ainput()
                await self._input_callback(user_input)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unable to poll input: {e}")
                raise

    async def initialize(self, **kwargs) -> None:
        """Initialize the interface and start input polling"""
        # Store the input_callback from kwargs
        if "input_callback" in kwargs:
            self._input_callback = kwargs["input_callback"]

        Console(highlight=True).print(
            Markdown(
                "# Console activated\n\nHit return to continue with the next step, or enter a prompt when you are ready.",
            ),
        )
        self._input_task = asyncio.create_task(self._poll_input())
