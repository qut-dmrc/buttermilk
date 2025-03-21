from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentMessages, AgentOutput


class CLIUserAgent(Agent):
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
            input_data.content or "Enter your message: \n",
        )
        Console().print(Markdown(f"### User: \n{user_input}"))
        return AgentOutput(content=user_input, agent=self.name)

    async def initialize(self) -> None:
        """Initialize the interface"""

    async def cleanup(self) -> None:
        """Clean up resources"""
