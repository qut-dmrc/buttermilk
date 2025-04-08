import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import regex as re
from aioconsole import ainput
from autogen_core import CancellationToken, MessageContext
from pydantic import PrivateAttr
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pretty_repr

from buttermilk import logger
from buttermilk._core.contract import (
    AgentOutput,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerRequest,
    ManagerResponse,
    UserInstructions,
)
from buttermilk.agents.ui.generic import UIAgent


class CLIUserAgent(UIAgent):
    _input_callback: Any = PrivateAttr(...)
    _console: Console = PrivateAttr(default_factory=lambda: Console(highlight=True, markup=True))

    def _fmt_msg(self, message: FlowMessage) -> Markdown:
        """Format a message for display in the console."""
        output = []
        output.append(f"### {message.source} ({message.role})")

        if message.outputs:
            output.append(pretty_repr(message.outputs, max_string=8000).replace("\\n", ""))
        else:
             output.append(message.content)
        output = [o for o in output if o]
        return Markdown("\n".join(output))

    async def listen(self, message: GroupchatMessageTypes, ctx: MessageContext = None, **kwargs):
        """Send output to the user interface."""
        if isinstance(message, UserInstructions):
            return

        self._console.print(self._fmt_msg(message))

        if isinstance(message, ManagerRequest):
            # self._console.print(message.description)
            self._console.print(Markdown("Input requested:\n"))

    async def _process(
        self,
        message: FlowMessage,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput | None, None]:
        """Do nothing at this stage."""
        yield  # Required for async generator typing

    async def _poll_input(
        self,
    ) -> None:
        """Continuously poll for user input in the background"""
        while True:
            try:
                user_input = await ainput()
                if user_input == "exit":
                    raise KeyboardInterrupt
                
                if re.sub(r"\W", "", user_input).lower() in ["x", "n", "no", "cancel", "abort", "stop", "quit", "q"     ]:
                    # confirm negative
                    await self._input_callback(
                        ManagerResponse(
                            role=self.role,
                            source=self.id,
                            confirm=False,
                        ),
                    )

                elif not re.sub(r"\W", "", user_input):
                # treat empty string as confirmation
                    await self._input_callback(
                        ManagerResponse(
                            role=self.role,
                            source=self.id,
                            confirm=True,
                        ),
                    )
                else:
                    await self._input_callback(UserInstructions(source=self.id, role=self.role, content=user_input))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unable to poll input: {e}")
                raise

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface and start input polling"""
        self._input_callback = input_callback

        self._input_task = asyncio.create_task(self._poll_input())
