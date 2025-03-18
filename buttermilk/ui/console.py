from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.ui import IOInterface


class CLIUserAgent(IOInterface):
    async def get_input(self, message: Any, source: str = "") -> str:
        """Request input from the user interface"""
        user_input = input(
            message or "Enter your message: ",
        )
        Console().print(Markdown(f"### User: \n{user_input}"))

        return user_input

    async def send_output(self, message: Any, source: str = "") -> None:
        """Send output to the user interface"""
        Console(highlight=True).print(
            Markdown(f"### {source}: \n{message}"),
        )

    async def initialize(self) -> None:
        """Initialize the interface"""

    async def cleanup(self) -> None:
        """Clean up resources"""
