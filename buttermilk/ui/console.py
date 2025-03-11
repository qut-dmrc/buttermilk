from rich.console import Console
from rich.markdown import Markdown

from buttermilk.runner.chat import (
    Answer,
    GroupChatMessage,
    IOInterface,
    RequestToSpeak,
)


class CLIUserAgent(IOInterface):
    async def query(self, request: RequestToSpeak) -> GroupChatMessage:
        """Retrieve input from the user interface"""
        user_input = input(
            request.content or "Enter your message: ",
        )
        Console().print(Markdown(f"### User: \n{user_input}"))
        reply = Answer(
            agent_id=self.id.type,
            role="user",
            content=user_input,
            step="User",
            config=self.config,
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
