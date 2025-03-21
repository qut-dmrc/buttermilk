from distutils.util import strtobool

from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    AgentMessages,
    AgentOutput,
    ManagerMessage,
)


class UIAgent(Agent):
    async def confirm(self, message: ManagerMessage) -> ManagerMessage:
        """Ask the user for confirmation."""
        prompt = message.content or "Would you like to proceed? y/n"
        result = await self.process(
            input_data=AgentInput(
                prompt=prompt,
            ),
        )
        try:
            confirm = bool(strtobool(result.content))
        except ValueError:
            confirm = False

        response = ManagerMessage(content=result.content, confirm=confirm)
        return response

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
            input_data.prompt or "Do you want to proceed? y/n \n",
        )
        Console().print(Markdown(f"### User: \n{user_input}"))
        return AgentOutput(content=user_input, agent=self.name)
