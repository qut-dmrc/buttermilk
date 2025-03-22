from distutils.util import strtobool

from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    AgentMessages,
    AgentOutput,
    ManagerMessage,
    UserConfirm,
)


class UIAgent(Agent):
    async def handle_control_message(self, message: ManagerMessage) -> ManagerMessage:
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
        return ManagerMessage(**result.model_dump())

    async def initialize(self) -> None:
        """Initialize the interface"""

    async def cleanup(self) -> None:
        """Clean up resources"""


class CLIUserAgent(UIAgent):
    async def receive_output(
        self,
        message: AgentMessages,
        source: str,
        **kwargs,
    ) -> None:
        """Send output to the user interface"""
        Console(highlight=True).print(
            Markdown(f"### {source}: \n{message.content}"),
        )

    async def process(self, input_data: AgentMessages, **kwargs) -> AgentMessages:
        """Request input from the user interface"""
        user_input = input(
            input_data.content or "Do you want to proceed? y/n \n",
        )
        Console().print(Markdown(f"### User: \n{user_input}"))
        return AgentOutput(content=user_input, agent=self.name)
