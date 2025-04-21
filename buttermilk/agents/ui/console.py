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

# Removed QualScore import as it's not used directly in logic after refactor
# from buttermilk.agents.evaluators.scorer import QualScore
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
                outputs_data = getattr(message, "outputs", None)
                call_id = None
                if isinstance(outputs_data, BaseModel) and hasattr(outputs_data, "call_id"):
                    call_id = getattr(outputs_data, "call_id", None)
                elif isinstance(outputs_data, dict):
                    call_id = outputs_data.get("call_id")
                if call_id:
                    output[0] = output[0] + f" (#{call_id})"

                # Removed QualScore specific check as it wasn't imported/used safely
                # if isinstance(outputs_data, QualScore): ...

                records_data = getattr(message, "records", None)
                if records_data:
                    for rec in records_data:
                        if isinstance(rec, Record):
                            output.append(f"### Record: {rec.record_id}")
                            rec_text = getattr(rec, "text", None)
                            if rec_text:
                                output.append(rec_text)

                inputs_data = getattr(message, "inputs", None)
                if inputs_data and hasattr(inputs_data, "parameters") and inputs_data.parameters:
                    output.append("### Parameters: ")
                    output.append(pretty_repr(inputs_data.parameters, max_string=400))

                if outputs_data:
                    reasons = None
                    if isinstance(outputs_data, BaseModel) and hasattr(outputs_data, "reasons"):
                        reasons = getattr(outputs_data, "reasons", None)
                    elif isinstance(outputs_data, dict):
                        reasons = outputs_data.get("reasons")

                    if reasons and isinstance(reasons, list):
                        output.append("### Reasons:")
                        output.extend([f"- {str(reason)}" for reason in reasons])
                    else:
                        output.append(pretty_repr(outputs_data, max_string=8000))
                elif message_content := getattr(message, "content", None):
                    output.append(str(message_content))

            # --- Handle TaskProcessingComplete ---
            elif isinstance(message, TaskProcessingComplete):
                if getattr(message, "is_error", False):
                    task_index = getattr(message, "task_index", "?")
                    output.append(f"Task from {source} #{task_index} failed...")

            # --- Handle UserInstructions / ManagerRequest (Prompt) ---
            elif isinstance(message, (UserInstructions, ManagerRequest)):
                prompt_content = getattr(message, "prompt", None)
                if prompt_content:
                    output.append(f"### Prompt: \n{prompt_content}")

            # --- Handle ToolOutput ---
            elif isinstance(message, ToolOutput):
                tool_content = getattr(message, "content", None)
                if tool_content:
                    output.append("### Tool Output:")
                    output.append(str(tool_content))

            # --- Handle AgentInput (e.g., if directly passed to _fmt_msg) ---
            elif isinstance(message, AgentInput):
                prompt_content = getattr(message, "prompt", None)
                if prompt_content:
                    output.append(f"### Input Prompt: \n{prompt_content}")
                # Optionally display message.inputs?
                if hasattr(message, "inputs") and message.inputs:
                    output.append("### Input Data:")
                    output.append(pretty_repr(message.inputs, max_string=400))

            # --- Handle unknown formattable types (shouldn't happen with Union hint) ---
            else:
                logger.debug(f"Unhandled formattable type in _fmt_msg: {type(message)}")
                if hasattr(message, "model_dump"):
                    output.append(pretty_repr(message.model_dump(), max_string=400))
                else:
                    return None

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
        message: GroupchatMessageTypes,  # This is Union[AgentOutput, ToolOutput, UserInstructions]
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> None:
        """Send output to the user interface."""
        # message type is already narrowed by the Agent.__call__ method's isinstance check
        # All types in GroupchatMessageTypes should be handled by _fmt_msg
        if msg := self._fmt_msg(message, source=source):
            self._console.print(msg)
        else:
            logger.debug(f"ConsoleAgent _listen could not format message type: {type(message)}")

    @buttermilk_handler(ManagerRequest)
    async def handle_manager_request(self, message: ManagerRequest) -> ManagerResponse:
        """Handle ManagerRequest messages from Autogen runtime."""
        # Display the request to the user
        if msg := self._fmt_msg(message, source="Manager"):
            self._console.print(msg)
        
        # Wait for user input (handled by _poll_input in the background)
        # The _poll_input method will call _input_callback with the response
        # This just informs the user a response is needed
        self._console.print("\nPlease respond to this request...\n")
        
        # Return None for now - actual response will come from _poll_input
        # This approach is necessary because Autogen expects synchronous handlers
        # but user input is inherently asynchronous
        return ManagerResponse(confirm=True, prompt="Request acknowledged")
    
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

        return None  # Base implementation returns None

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

                if re.sub(r"\W", "", user_input).lower() in ["x", "n", "no", "cancel", "abort", "stop", "quit", "q"]:
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

                # Need TaskProcessingComplete and HeartBeat imports
                from buttermilk._core.contract import TaskProcessingComplete, HeartBeat

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
