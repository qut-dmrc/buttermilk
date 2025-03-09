from rich.console import Console
from rich.markdown import Markdown

from buttermilk.runner.chat import (
    GroupChatMessage,
    IOInterface,
)


class CLIUserAgent(IOInterface):
    def __init__(self, description: str = None, **kwargs):
        description = description or "The human in the loop"
        super().__init__(description=description, **kwargs)

    async def get_input(self, prompt: str = "") -> GroupChatMessage:
        """Retrieve input from the user interface"""
        user_input = input(
            prompt or "Enter your message: ",
        )
        Console().print(Markdown(f"### User: \n{user_input}"))
        reply = GroupChatMessage(
            content=user_input,
            step="User",
        )
        return reply

    async def send_output(self, message: GroupChatMessage, source: str = "") -> None:
        """Send output to the user interface"""
        Console().print(
            Markdown(f"### {source}: \n{message.content}"),
        )

    async def initialize(self) -> None:
        """Initialize the interface"""

    async def cleanup(self) -> None:
        """Clean up resources"""
