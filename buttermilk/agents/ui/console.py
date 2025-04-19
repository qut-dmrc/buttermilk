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
from buttermilk._core.agent import AgentInput, ConductorResponse, OOBMessages
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
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import QualScore
from buttermilk.agents.ui.generic import UIAgent

from rich.highlighter import JSONHighlighter

console = Console(highlighter=JSONHighlighter())
class CLIUserAgent(UIAgent):
    _input_callback: Any = PrivateAttr(...)
    _console: Console = PrivateAttr(default_factory=lambda: Console(highlight=True, markup=True))

    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentOutput | None:
        """Send or receive input from the UI."""
        self._console.print(Markdown("Input requested:\n"))

    def _fmt_msg(self, message: FlowMessage, source: str) -> Markdown | None:
        """Format a message for display in the console."""
        output = [f"## {source} "]
        try:
            if isinstance(message, (AgentOutput, ConductorResponse)):
                # add call_id if we can
                if call_id := getattr(message.outputs, "call_id", None):
                    output[0] = output[0] + f" (#{call_id})"
                # Is there a score object?
                if isinstance(message.outputs, QualScore):
                    output.append(str(message.outputs))
                    return Markdown("\n".join(output))

                # Is there a Record object?
                if message.records:
                    for rec in message.records:
                        if isinstance(rec, Record):
                            output.append(f"### Record: {rec.record_id}")
                            output.append(rec.text)
                            return Markdown("\n".join(output))
                if message.inputs and message.inputs.parameters:
                    output.append("### Parameters: ")
                    output.append(pretty_repr(message.inputs.parameters, max_string=400))

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
                if message.is_error:
                    output.append(f"Task from {source} #{message.task_index} failed...")
            else:
                output.append(message.content)
        except Exception as e:
            logger.error(f"Unable to format message of type {type(message)}: {e}")
            output.append(pretty_repr(message.model_dump(), max_string=400))

        output = [o for o in output if o]
        if len(output) > 1:
            return Markdown("\n".join(output))
        return None

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken | None = None,
        source: str = "unknown",
        **kwargs,
    ) -> None:
        """Send output to the user interface."""
        if isinstance(message, UserInstructions):
            return
        if msg := self._fmt_msg(message, source=source):
            self._console.print(msg)

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken = None,
        **kwargs,
    ) -> OOBMessages:
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        if out := self._fmt_msg(message, source=kwargs.get("source", "unknown")):
            self._console.print(out)

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
                            role=self.role,
                            confirm=False,
                        ),
                    )

                elif not re.sub(r"\W", "", user_input):
                    # treat empty string as confirmation
                    await self._input_callback(
                        ManagerResponse(
                            role=self.role,
                            confirm=True,
                        ),
                    )
                else:
                    await self._input_callback(UserInstructions(content=user_input))
                await self._input_callback(
                    TaskProcessingComplete(agent_id=self.id, role=self.role, task_index=0, more_tasks_remain=False, is_error=False)
                )
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
