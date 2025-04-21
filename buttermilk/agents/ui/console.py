import asyncio
from collections.abc import AsyncGenerator
from textwrap import indent
from typing import Any, Awaitable, Callable, List, Union, Optional  # Added List, Union, Optional

import regex as re
from aioconsole import ainput
from autogen_core import CancellationToken, MessageContext
from pydantic import PrivateAttr, BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pretty_repr

from buttermilk import logger
from buttermilk._core.agent import AgentInput, ConductorResponse, OOBMessages, buttermilk_handler
from buttermilk._core.contract import (
    AgentOutput,
    FlowMessage,
    GroupchatMessageTypes,
    HeartBeat,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    UserInstructions,
    ToolOutput,
)
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import QualScore
from buttermilk.agents.ui.generic import UIAgent
import weave  # Import weave

from rich.highlighter import JSONHighlighter

console = Console(highlighter=JSONHighlighter())


# Define a Union for types _fmt_msg can handle for better type safety
FormattableMessages = Union[AgentOutput, ConductorResponse, TaskProcessingComplete, UserInstructions, ManagerRequest, ToolOutput, AgentInput]
class CLIUserAgent(UIAgent):
    _input_callback: Any = PrivateAttr(...)
    _console: Console = PrivateAttr(default_factory=lambda: Console(highlight=True, markup=True))
    _input_task: Optional[asyncio.Task] = PrivateAttr(default=None)  # Allow None

    @weave.op()
    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | None:
        """Send or receive input from the UI."""
        if msg := self._fmt_msg(inputs, source="controller"):
            self._console.print(msg)
        else:
            self._console.print(Markdown("Input requested:\n"))
        # Return None as _process in UI agents usually doesn't produce direct output for flow
        return None

    def _fmt_msg(self, message: FormattableMessages, source: str) -> Markdown | None:
        """Format a known message type for display in the console."""
        output = [f"## {source} "]
        try:
            # --- Handle AgentOutput / ConductorResponse ---
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
            elif isinstance(message, ManagerRequest):
                output.append(f"### Request from {message.role} ({message.agent_id})\n{message.content}")
            elif isinstance(message, TaskProcessingComplete):
                if message.is_error:
                    output.append(f"Task from {source} #{message.task_index} failed...")
            elif message.prompt:
                output.append(f"### Prompt: \n{message.prompt}")
            elif message.content:
                output.append(message.content)
        except Exception as e:
            logger.error(f"Error formatting message of type {type(message)}: {e}", exc_info=True)
            if hasattr(message, "model_dump"):
                output.append(f"Error formatting message. Raw data:\n {pretty_repr(message.model_dump(), max_string=400)}")
            else:
                output.append(f"Error formatting message: {message}")

        # Filter out None/empty strings and join
        output = [str(o) for o in output if o is not None]  # Check for None explicitly
        if len(output) > 1:
            return Markdown("\n".join(output))
        return None

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> None:
        """Send output to the user interface."""
        if msg := self._fmt_msg(message, source=source):
            self._console.print(msg)

    async def _handle_events(
        self,
        message: OOBMessages,  # This is a Union
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> OOBMessages | None:  # Return type matches base
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        # Check if the specific OOB message type is formattable
        formatable_types = (AgentOutput, ConductorResponse, TaskProcessingComplete, ManagerRequest)  # AgentOutput isn't OOB, but check anyway
        if isinstance(message, formatable_types):
            if out := self._fmt_msg(message, source=kwargs.get("source", "unknown")):
                self._console.print(out)
        else:
            logger.debug(f"ConsoleAgent received unformatted OOB message: {type(message)}")

        return None 

    async def _poll_input(
        self,
    ) -> None:
        """Continuously poll for user input in the background"""
        prompt: list[str] = []  # Use lowercase list for hint consistency
        while True:
            try:
                user_input = await ainput()
                if user_input == "exit":
                    raise KeyboardInterrupt

                if re.sub(r"\W", "", user_input).lower() in ["x", "n", "no", "cancel", "abort", "stop", "quit", "q"     ]:
                    # confirm negative
                    await self._input_callback(
                        ManagerResponse(
                            confirm=False,
                            prompt="\n".join(prompt),
                        ),
                    )
                    prompt = []

                elif not re.sub(r"\W", "", user_input):
                    # treat empty string as confirmation
                    await self._input_callback(
                        ManagerResponse(
                            confirm=True,
                            prompt="\n".join(prompt),
                        ),
                    )
                    prompt = []
                else:
                    # accept input until we get an empty string
                    prompt.append(user_input)

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
        if input_callback:  # Only start polling if a callback is provided
            self._input_task = asyncio.create_task(self._poll_input())
        else:
            logger.warning("ConsoleAgent initialized without input_callback. Input polling disabled.")
            # self._input_task is already None by default

    async def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, "_input_task") and self._input_task:
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                logger.info("Console input task cancelled.")
