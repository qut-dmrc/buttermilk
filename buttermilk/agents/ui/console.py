import asyncio
from collections.abc import AsyncGenerator
from textwrap import indent
from typing import Any, Awaitable, Callable

import regex as re
from aioconsole import ainput
from autogen_core import CancellationToken, MessageContext
from pydantic import PrivateAttr
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pretty_repr

from buttermilk import logger
from buttermilk._core.agent import ConductorResponse, OOBMessages
from buttermilk._core.contract import (
    AgentOutput,
    FlowMessage,
    GroupchatMessageTypes,
    HeartBeat,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    UserInstructions,
)
from buttermilk.agents.ui.generic import UIAgent

class CLIUserAgent(UIAgent):
    _input_callback: Any = PrivateAttr(...)
    _console: Console = PrivateAttr(default_factory=lambda: Console(highlight=True, markup=True))

    def _fmt_msg(self, message: FlowMessage) -> Markdown:
        """Format a message for display in the console."""
        output = []
        try:
            output.append(f"## {message.source} ({message.role})")
            if isinstance(message, (AgentOutput, ConductorResponse)):
                if message.params:
                    output.append("### Parameters: " + pretty_repr(message.params, max_string=400))
                if message.outputs:
                    if reasons := message.outputs.get("reasons"):
                        output.append("### Reasons:")
                        output.extend([f"- {reason}" for reason in reasons])
                    else:
                        output.append(pretty_repr(message.outputs, max_string=8000))
                else:
                    output.append(message.content)
                # for rec in message.records:
                #     output.append(str(rec).replace("\\n", ""))
            elif isinstance(message, TaskProcessingComplete):
                output.append(f"Task {message.role} #{message.task_index} completed {'successfully' if message.is_error else 'with ERROR'}.")
            else:
                output.append(message.content)
            output = [o for o in output if o]
            return Markdown("\n".join(output))
        except Exception as e:
            logger.error(f"Unable to format message of type {type(message)}: {e}")
            return message.model_dump_json(indent=2)

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> GroupchatMessageTypes | None:
        """Send output to the user interface."""
        if isinstance(message, UserInstructions):
            return

        self._console.print(self._fmt_msg(message))

    async def _handle_control_message(
        self, message: OOBMessages, cancellation_token: CancellationToken = None, public_callback: Callable = None, message_callback: Callable = None, **kwargs
    ) -> OOBMessages:
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        self._console.print(self._fmt_msg(message))

        if isinstance(message, ManagerRequest):
            # self._console.print(message.description)
            self._console.print(Markdown("Input requested:\n"))
        return None

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
                            source=self.name,
                            role=self.role,
                            confirm=False,
                        ),
                    )

                elif not re.sub(r"\W", "", user_input):
                    # treat empty string as confirmation
                    await self._input_callback(
                        ManagerResponse(
                            source=self.name,
                            role=self.role,
                            confirm=True,
                        ),
                    )
                else:
                    await self._input_callback(UserInstructions(source=self.id, role=self.role, content=user_input))
                await self._input_callback(HeartBeat(go_next=True))
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unable to poll input: {e}")
                raise

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the interface and start input polling"""
        self._input_callback = input_callback

        self._input_task = asyncio.create_task(self._poll_input())
