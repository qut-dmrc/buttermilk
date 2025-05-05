"""Defines the CLIUserAgent for interacting with the user via the command line console.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable  # Added List, Union, Optional
from typing import Union

import regex as re
from aioconsole import ainput  # For asynchronous console input
from autogen_core import CancellationToken  # Autogen types (used by base Agent)
from pydantic import PrivateAttr
from rich.console import Console
from rich.highlighter import JSONHighlighter  # Specific highlighter for JSON
from rich.markdown import Markdown
from rich.pretty import pretty_repr  # For formatted output of complex objects

from buttermilk import logger

# Import base agent and specific message types used
from buttermilk._core.agent import AgentInput, OOBMessages
from buttermilk._core.contract import (
    AgentTrace,
    ConductorResponse,
    FlowMessage,  # Base type for messages
    GroupchatMessageTypes,  # Union type for messages in group chat
    ManagerRequest,  # Requests sent *to* the manager (this agent)
    ManagerResponse,  # Responses sent *from* the manager (this agent)
    TaskProcessingComplete,  # Status updates
    ToolOutput,  # Potentially displayable tool output
)
from buttermilk._core.types import Record  # For displaying record data
from buttermilk.agents.evaluators.scorer import QualResults, QualScore  # Specific format for scores
from buttermilk.agents.judge import JudgeReasons  # Specific format for judge reasons
from buttermilk.agents.ui.generic import UIAgent  # Base class for UI agents

# Initialize a global console instance with JSON highlighting
console = Console(highlighter=JSONHighlighter())


# Define a Union for types _fmt_msg can handle, improving type safety for the formatter.
FormattableMessages = Union[
    AgentTrace,
    ConductorResponse,
    TaskProcessingComplete,
    ManagerRequest,
    ToolOutput,
    AgentInput,
    Record,
    QualScore,
    JudgeReasons,
    FlowMessage,
]
# TODO: Add other relevant FlowMessage subtypes if needed for formatting.


class CLIUserAgent(UIAgent):
    """Represents the human user interacting via the Command Line Interface (CLI).

    Inherits from `UIAgent`. It uses `rich` to display formatted messages received
    by the agent (`_listen`, `_handle_events`) and `aioconsole` to asynchronously
    poll for user input (`_poll_input`). User input is interpreted as confirmation,
    negation, or free text, which is then sent back to the system as a `ManagerResponse`
    via the `_input_callback` provided during initialization.
    """

    # Callback function provided by the orchestrator/adapter to send ManagerResponse back.
    _input_callback: Callable[[ManagerResponse], Awaitable[None]] | None = PrivateAttr(default=None)
    # Rich console instance for formatted output.
    _console: Console = PrivateAttr(default_factory=lambda: Console(highlight=True, markup=True))
    # Background task for polling user input.
    _input_task: asyncio.Task | None = PrivateAttr(default=None)

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentTrace | None:
        """Handles direct AgentInput messages, typically displaying them as requests to the user.

        Args:
            message: The AgentInput message received.
            cancellation_token: Autogen cancellation token.
            **kwargs: Additional keyword arguments.

        Returns:
            None. UI agents typically don't return direct outputs in the main flow via _process.
                   They interact via callbacks or listening methods.

        """
        logger.debug(f"{self.agent_id}: Received direct input request via _process.")
        # Format and display the incoming message.
        if msg_markdown := self._fmt_msg(message, source="controller"):  # Use := walrus operator
            self._console.print(msg_markdown)
        else:
            # Fallback if formatting fails or returns None
            self._console.print(Markdown(f"## Input Request:\n{pretty_repr(message)}"))  # Display raw if formatting fails

        # This method primarily triggers output; input is handled by _poll_input.
        return None

    def _fmt_msg(self, message: FormattableMessages, source: str) -> Markdown | None:
        """Formats various message types into Rich Markdown for console display.

        Args:
            message: The message object to format.
            source: The identifier of the agent sending the message.

        Returns:
            A Rich Markdown object ready for printing, or None if the message is not formatted.

        """
        # TODO: This function is complex. Refactor into smaller helpers for maintainability.
        #       Consider using a dictionary dispatch pattern based on message type.
        output_lines: list[str] = []  # Use list to build output lines

        try:
            # Add specific identifiers if available
            agent_id = getattr(message, "agent_id", "unknown")
            role = getattr(message, "role", None)
            header = f"## Message from {agent_id}"

            output_lines.append(header)

            # --- Specific Type Formatting ---
            if isinstance(message, (AgentTrace, ConductorResponse)):
                outputs = getattr(message, "outputs", None)
                inputs = getattr(message, "inputs", None)  # Original inputs triggering this output
                metadata = getattr(message, "metadata", {})

                # Display specific structured outputs
                if isinstance(outputs, QualResults):
                    output_lines.append(str(outputs))  # Use QualResults's __str__
                elif isinstance(outputs, JudgeReasons):
                    output_lines.append("### Conclusion")
                    output_lines.append(f"**Prediction**: {outputs.prediction}\t\t**Confidence**: {outputs.confidence}")
                    output_lines.append(f"**Conclusion**: {outputs.conclusion}")
                    if outputs.reasons:
                        output_lines.append("### Reasons:")
                        output_lines.extend([f"- {reason}" for reason in outputs.reasons])
                # Handle list output (potentially records?)
                elif isinstance(outputs, list):
                    # Check if it's a list of Records - needs better type checking maybe
                    if outputs and isinstance(outputs[0], dict) and "record_id" in outputs[0]:
                        for i, rec_dict in enumerate(outputs):
                            output_lines.append(f"### Record {i + 1}: {rec_dict.get('record_id', 'N/A')}")
                            # Display limited fields for brevity
                            output_lines.append(f"```\n{pretty_repr(rec_dict, max_string=500)}\n```")
                    else:  # Generic list
                        output_lines.append("### Output:")
                        output_lines.append(f"```\n{pretty_repr(outputs, max_string=8000)}\n```")

                # Handle dictionary output (generic JSON/dict)
                elif isinstance(outputs, dict):
                    # Look for common patterns like 'reasons' list
                    if reasons := outputs.get("reasons"):
                        if isinstance(reasons, list):
                            output_lines.append("### Reasons:")
                            output_lines.extend([f"- {reason}" for reason in reasons])
                        else:  # If reasons is not a list, print raw dict
                            output_lines.append("### Output:")
                            output_lines.append(f"```json\n{json.dumps(outputs, indent=2)}\n```")
                    else:  # Generic dictionary
                        output_lines.append("### Output:")
                        output_lines.append(f"```json\n{json.dumps(outputs, indent=2)}\n```")

                # Handle raw string content if outputs is just a string
                elif isinstance(outputs, str):
                    output_lines.append("### Output:")
                    output_lines.append(outputs)

                # Display raw content if available and not handled above
                elif hasattr(message, "contents") and message.content and not outputs:
                    output_lines.append(message.content)

                # Optionally display inputs that led to this output
                # if inputs:
                #    output_lines.append("### (Triggering Input Parameters):")
                #    output_lines.append(f"```\n{pretty_repr(inputs.parameters if hasattr(inputs, 'parameters') else inputs, max_string=400)}\n```")

                # Display metadata (like token usage)
                if metadata:
                    output_lines.append(f"### Metadata:\n\n{metadata!s}")
                    # syntax = Syntax(str(metadata), "python", theme="default", line_numbers=False, word_wrap=False)

            elif isinstance(message, AgentInput):
                output_lines.append("### Input Request:")
                output_lines.append(f"**Prompt:** {message.prompt}" if message.prompt else "(No explicit prompt)")
                if message.inputs:
                    output_lines.append("**Inputs:**")
                    output_lines.append(f"```json\n{json.dumps(message.inputs, indent=2)}\n```")
                # Could add context/records display here if needed

            elif isinstance(message, ManagerRequest):
                output_lines.append(f"### Request:\n{message.content}")
                # Often includes a plan or question needing confirmation

            elif isinstance(message, TaskProcessingComplete):
                if message.is_error:
                    output_lines.append(f"Task {message.task_index} FAILED.")
                # else: Don't necessarily need to show successful completion signal

            elif isinstance(message, ToolOutput):
                output_lines.append(f"### Tool Output (for Tool Call {message.call_id}):")
                # TODO: Format tool output better based on its content type
                output_lines.append(f"```\n{pretty_repr(message.content, max_string=1000)}\n```")

            # Generic FlowMessage with content/prompt
            elif hasattr(message, "prompt") and message.prompt:
                output_lines.append(f"### Prompt:\n{message.prompt}")
            elif hasattr(message, "content") and message.content:
                output_lines.append(message.content)

        except Exception as e:
            logger.error(f"Error formatting message type {type(message)} from {source}: {e}")
            # Fallback to raw representation on error
            try:
                output_lines.append("_(Error formatting message. Raw data below)_")
                output_lines.append(f"```\n{pretty_repr(message, max_string=1000)}\n```")
            except Exception:  # If even pretty_repr fails
                output_lines.append("_(Error formatting message, could not represent raw data)_")

        # Filter out the header if no other content was added, or if content is just empty strings/None
        filtered_output = [line for line in output_lines if isinstance(line, str) and line.strip()]
        if len(filtered_output) > 1:  # Check if more than just the header was added
            return Markdown("\n".join(filtered_output))
        logger.debug(f"Skipping display for message type {type(message)} from {source} - no formatted content.")
        return None  # Don't print if only header remains or content was empty

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,  # Usually unused by UI agent in listen
        message_callback: Callable | None = None,  # Usually unused by UI agent in listen
        **kwargs,
    ) -> None:
        """Displays messages received from other agents on the console."""
        logger.debug(f"{self.agent_id} received message from {source} via _listen.")
        # Format and display the message using the helper function.
        if msg_markdown := self._fmt_msg(message, source=source):
            self._console.print(msg_markdown)

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        source: str = "unknown",  # Add source default
        **kwargs,
    ) -> OOBMessages | None:
        """Handles Out-Of-Band messages, displaying relevant ones."""
        logger.debug(f"{self.agent_id} received OOB message from {source}: {type(message).__name__}")
        # Check if the specific OOB message type is one we want to display.
        # AgentTrace isn't technically OOB, but might arrive here in some flows? Included defensively.
        displayable_types = (AgentTrace, ConductorResponse, TaskProcessingComplete, ManagerRequest, ToolOutput)
        if isinstance(message, displayable_types):
            if msg_markdown := self._fmt_msg(message, source=source):
                self._console.print(msg_markdown)
        else:
            # Log other OOB types at debug level if not explicitly formatted.
            logger.debug(f"ConsoleAgent ignoring unformatted OOB message type: {type(message).__name__}")

        # OOB handlers typically don't return responses unless specifically requested.
        return None

    async def _poll_input(self) -> None:
        """Continuously polls for user input from the console in a background task."""
        logger.info(f"{self.agent_id}: Starting console input polling...")
        current_prompt_lines: list[str] = []
        while True:
            try:
                # Get input asynchronously, allowing other tasks to run.
                # The prompt indicates multi-line mode (Enter blank line to send).
                display_prompt = "(Enter text, blank line to send/confirm, 'n'/'q' to cancel)> "
                user_input = await ainput(display_prompt)

                # Handle exit command
                if user_input.strip().lower() == "exit":
                    logger.info("User requested exit.")
                    # TODO: How to signal exit cleanly to the orchestrator? Raising KeyboardInterrupt might be harsh.
                    # Maybe send a specific ManagerResponse or signal?
                    # For now, simulate interrupt. Need a better mechanism.
                    # Find the main task and cancel it? Difficult from here.
                    # Send a special message via callback?
                    # await self._input_callback(ManagerResponse(confirm=False, prompt="USER_EXIT_REQUEST"))
                    raise KeyboardInterrupt  # Temporary way to stop, might need refinement

                # Handle confirmation/negation based on input
                cleaned_input = re.sub(r"\W", "", user_input).lower()
                is_negation = cleaned_input in ["x", "n", "no", "cancel", "abort", "stop", "quit", "q"]
                is_confirmation = not user_input.strip()  # Empty line confirms

                if is_negation:
                    logger.info("User input interpreted as NEGATIVE confirmation.")
                    response = ManagerResponse(confirm=False, interrupt=False, prompt="\n".join(current_prompt_lines))
                    current_prompt_lines = []  # Reset prompt buffer
                    await self._input_callback(response)
                elif is_confirmation:
                    # If we have accumulated feedback in the prompt buffer, this is a confirmation WITH feedback,
                    # which should be treated as an interruption requiring the Conductor's attention
                    has_feedback = bool(current_prompt_lines)
                    if has_feedback:
                        logger.info("User input interpreted as POSITIVE confirmation with feedback (interrupt).")
                        response = ManagerResponse(confirm=True, interrupt=True, prompt="\n".join(current_prompt_lines))
                    else:
                        logger.info("User input interpreted as POSITIVE confirmation (no feedback).")
                        response = ManagerResponse(confirm=True, interrupt=False, prompt=None)

                    current_prompt_lines = []  # Reset prompt buffer
                    await self._input_callback(response)
                else:
                    # Non-empty, non-negation input: add to multi-line buffer
                    logger.debug(f"User input added to buffer: '{user_input}'")
                    current_prompt_lines.append(user_input)
                    # Don't send response yet, wait for empty line to confirm multi-line input
                    continue  # Go back to prompt for more input

                # Optional: Signal heartbeat or task completion after processing input via callback
                # await self._input_callback(TaskProcessingComplete(...)) # If input completes a "task"
                # await self._input_callback(HeartBeat(go_next=True)) # If input allows flow to proceed

                await asyncio.sleep(0.1)  # Small sleep to prevent tight loop if needed

            except asyncio.CancelledError:
                logger.info(f"{self.agent_id}: Input polling task cancelled.")
                break  # Exit loop cleanly on cancellation
            except Exception as e:
                # Log errors during input polling but try to continue
                logger.error(f"{self.agent_id}: Error polling console input: {e}")
                # Consider adding a delay before retrying after an error
                await asyncio.sleep(1)
                # Re-raise if it's KeyboardInterrupt to allow stopping the application
                if isinstance(e, KeyboardInterrupt):
                    raise

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        # Call base class initialize if needed
        await super().initialize(**kwargs)
        """
        Initializes the agent and starts the background input polling task if a callback is provided.

        Args:
            input_callback: The async function to call when user input is received.
                            Expected signature: `async def callback(response: ManagerResponse)`
            **kwargs: Additional keyword arguments passed to the base class initializer.
        """
        logger.debug(f"Initializing {self.agent_id}...")
        self._input_callback = input_callback
        # Ensure any existing task is cancelled before starting a new one (e.g., on reset)
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass  # Expected

        if self._input_callback:
            # Start the background task to poll for console input.
            self._input_task = asyncio.create_task(self._poll_input())
            logger.debug(f"{self.agent_id}: Input polling task created.")
        else:
            # If no callback, input polling is disabled.
            logger.warning(f"{self.agent_id}: Initialized without input_callback. Console input polling disabled.")
            self._input_task = None

    async def cleanup(self) -> None:
        """Cleans up resources, primarily by cancelling the input polling task."""
        logger.debug(f"Cleaning up {self.agent_id}...")
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                # Wait for the task to acknowledge cancellation
                await self._input_task
            except asyncio.CancelledError:
                logger.info(f"{self.agent_id}: Console input task successfully cancelled.")
            except Exception as e:
                # Log if waiting for cancellation fails unexpectedly
                logger.error(f"{self.agent_id}: Error during input task cleanup: {e}")
        else:
            logger.debug(f"{self.agent_id}: No active input task to cancel.")
        # Call base class cleanup if needed
        # await super().cleanup()
