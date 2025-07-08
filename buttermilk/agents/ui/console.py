"""Defines the CLIUserAgent for interacting with the user via the command line console.
"""

import asyncio
from collections.abc import Awaitable, Callable  # Added List, Union, Optional
from datetime import datetime
from typing import Union

import regex as re
from aioconsole import ainput  # For asynchronous console input
from autogen_core import CancellationToken  # Autogen types (used by base Agent)
from pydantic import PrivateAttr
from rich.console import Console
from rich.highlighter import JSONHighlighter  # Specific highlighter for JSON
from rich.pretty import pretty_repr  # For formatted output of complex objects
from rich.text import Text

from buttermilk import logger

# Import base agent and specific message types used
from buttermilk._core.agent import AgentInput, OOBMessages
from buttermilk._core.config import FatalError
from buttermilk._core.contract import (
    AgentAnnouncement,  # Agent announcement messages
    AgentTrace,
    FlowMessage,  # Base type for messages
    GroupchatMessageTypes,  # Union type for messages in group chat
    ManagerMessage,  # Responses sent *from* the manager (this agent)
    TaskProcessingComplete,  # Status updates
    TaskProcessingStarted,  # Task start notifications (to be filtered)
    ToolOutput,  # Potentially displayable tool output
    UIMessage,  # Requests sent *to* the manager (this agent)
)
from buttermilk._core.types import Record  # For displaying record data
from buttermilk.agents.differences import Differences
from buttermilk.agents.evaluators.scorer import QualResults, QualScore  # Specific format for scores
from buttermilk.agents.judge import JudgeReasons  # Specific format for judge reasons
from buttermilk.agents.rag import ResearchResult
from buttermilk.agents.ui.generic import UIAgent  # Base class for UI agents

# Initialize a global console instance with JSON highlighting
console = Console(highlighter=JSONHighlighter())


# Model-based colors (matching frontend)
MODEL_COLORS = {
    "gpt-4": "#10a37f",      # OpenAI green
    "gpt4": "#10a37f",
    "gpt-3.5": "#1f85de",    # OpenAI blue
    "gpt3": "#1f85de",
    "o3": "#00d4aa",         # OpenAI teal for o3 series
    "claude": "#ff6b35",     # Anthropic orange
    "opus": "#ff6b35",
    "sonnet": "#ff6b35",
    "haiku": "#ff6b35",
    "gemini": "#4285f4",     # Google blue
    "llama": "#0866ff",      # Meta blue
    "falcon": "#3498db",
}

# IRC-style agent icons
AGENT_ICONS = {
    "judge": "⚖",
    "scorer": "📊",
    "assistant": "🤖",
    "describer": "📝",
    "fetch": "🔍",
    "imagegen": "🎨",
    "reasoning": "💭",
    "scraper": "🕷",
    "spy": "👁",
    "synthesiser": "⚡",
    "tool": "🔧",
    "instructions": "📋",
    "record": "📄",
    "summary": "📈",
    "researcher": "🔬",
    "system": "⚙",
    "controller": "🎛",
    "manager": "👤",
}

def get_model_color(message) -> str:
    """Get color for agent based on model (matching frontend logic)"""
    if hasattr(message, "agent_info") and message.agent_info:
        if hasattr(message.agent_info, "parameters") and message.agent_info.parameters:
            model = getattr(message.agent_info.parameters, "model", None)
            if model:
                model_lower = model.lower()
                for model_name, color in MODEL_COLORS.items():
                    if model_name in model_lower:
                        # Convert hex colors to rich color names
                        if color == "#10a37f": return "green"
                        elif color == "#1f85de": return "blue"
                        elif color == "#00d4aa": return "cyan"
                        elif color == "#ff6b35": return "bright_red"
                        elif color == "#4285f4": return "bright_blue"
                        elif color == "#0866ff": return "blue"
                        elif color == "#3498db": return "bright_cyan"

    # Fallback to role-based coloring
    if hasattr(message, "agent_info") and message.agent_info:
        agent_name = getattr(message.agent_info, "agent_name", "").lower()
        if "judge" in agent_name: return "bright_red"
        elif "scorer" in agent_name: return "bright_yellow"
        elif "assistant" in agent_name: return "bright_blue"
        elif "researcher" in agent_name: return "bright_cyan"

    return "dim white"

def get_agent_name(message, source: str) -> str:
    """Extract agent name from message, preferring agent_info.agent_name"""
    if hasattr(message, "agent_info") and message.agent_info:
        agent_name = getattr(message.agent_info, "agent_name", None)
        if agent_name:
            return agent_name

    # Fallback to agent_id or source
    agent_id = getattr(message, "agent_id", source or "unknown")
    return agent_id

def get_agent_icon(agent_id: str) -> str:
    """Get icon for agent based on name/type"""
    agent_lower = agent_id.lower()
    for agent_type, icon in AGENT_ICONS.items():
        if agent_type in agent_lower:
            return icon
    return "💬"

def get_model_tag(message) -> str:
    """Extract model identifier from message"""
    if hasattr(message, "agent_info") and message.agent_info:
        if hasattr(message.agent_info, "parameters") and message.agent_info.parameters:
            model = getattr(message.agent_info.parameters, "model", None)
            if model:
                model_lower = model.lower()
                if "gpt-4" in model_lower or "gpt4" in model_lower:
                    return "GPT4"
                elif "gpt-3" in model_lower:
                    return "GPT3"
                elif "o3" in model_lower:
                    return "O3"
                elif "sonnet" in model_lower:
                    return "SNNT"
                elif "opus" in model_lower:
                    return "OPUS"
                elif "haiku" in model_lower:
                    return "HAIK"
                elif "claude" in model_lower:
                    return "CLDE"
                elif "gemini" in model_lower:
                    return "GEMN"
                elif "llama" in model_lower:
                    return "LLMA"
    return ""

def format_timestamp() -> str:
    """Format current time as HH:MM:SS for IRC-style display"""
    return datetime.now().strftime("%H:%M:%S")


# Define a Union for types _fmt_msg can handle, improving type safety for the formatter.
FormattableMessages = Union[
    AgentTrace,
    TaskProcessingComplete,
    TaskProcessingStarted,
    UIMessage,
    ToolOutput,
    AgentInput,
    Record,
    QualScore,
    JudgeReasons,
    FlowMessage, ResearchResult,
    AgentAnnouncement,  # Added for agent announcements
]
# TODO: Add other relevant FlowMessage subtypes if needed for formatting.


class CLIUserAgent(UIAgent):
    """Represents the human user interacting via the Command Line Interface (CLI).

    Inherits from `UIAgent`. It uses `rich` to display formatted messages received
    by the agent (`_listen`, `_handle_events`) and `aioconsole` to asynchronously
    poll for user input (`_poll_input`). User input is interpreted as confirmation,
    negation, or free text, which is then sent back to the system as a `ManagerMessage`
    via the `callback_to_groupchat` provided during initialization.
    """

    # Rich console instance for formatted output.
    _console: Console = PrivateAttr(default_factory=lambda: Console(highlight=True, markup=True))
    # Background task for polling user input.
    _input_task: asyncio.Task | None = PrivateAttr(default=None)

    async def callback_to_ui(self, message, source: str = "system", **kwargs):
        if formatted_msg := self._fmt_msg(message, source=source):
            # If formatting is successful, print the message to the console.
            self._console.print(formatted_msg)

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
        logger.debug(f"{self.agent_name}: Received direct input request via _process.")
        # Format and display the incoming message.
        if formatted_msg := self._fmt_msg(message, source="controller"):  # Use := walrus operator
            self._console.print(formatted_msg)
        else:
            # Fallback if formatting fails or returns None - use IRC style for consistency
            fallback_text = Text()
            fallback_text.append(f"[{format_timestamp()}] ", style="dim")
            fallback_text.append("🎛 ", style="white")
            fallback_text.append("controller".ljust(16), style="red")
            fallback_text.append(" │ ", style="dim")
            fallback_text.append(f"INPUT REQ: {pretty_repr(message)[:100]}", style="yellow")
            self._console.print(fallback_text)

        # This method primarily triggers output; input is handled by _poll_input.
        return None

    def _fmt_msg(self, message: FormattableMessages, source: str) -> Text | None:
        """Formats various message types into IRC-style console display.

        Args:
            message: The message object to format.
            source: The identifier of the agent sending the message.

        Returns:
            A Rich Text object ready for printing, or None if the message is not formatted.
        """
        try:
            # Extract agent information
            agent_name = get_agent_name(message, source)
            agent_color = get_model_color(message)
            agent_icon = get_agent_icon(agent_name)
            timestamp = format_timestamp()

            # Get display name from agent (includes model tag if LLM agent)
            if hasattr(message, "agent_info") and message.agent_info and hasattr(message.agent_info, "get_display_name"):
                display_name = message.agent_info.get_display_name()
            else:
                # Fallback to basic name with model tag for backward compatibility
                model_tag = get_model_tag(message)
                display_name = agent_name
                if model_tag:
                    display_name = f"{display_name}[{model_tag}]"
            
            # Apply UI-specific formatting (ljust for console alignment)
            agent_display = display_name[:16].ljust(16)

            # Build the formatted message text
            result = Text()
            result.append(f"[{timestamp}] ", style="dim")
            result.append(f"{agent_icon} ", style="white")
            result.append(agent_display, style=agent_color)
            result.append(" │ ", style="dim")

            # Format content based on message type
            content_added = False

            if isinstance(message, AgentTrace) or hasattr(message, "outputs"):
                outputs = getattr(message, "outputs", None)

                if isinstance(outputs, QualResults):
                    result.append(f"Score: {outputs}", style="bright_white")
                    content_added = True

                elif isinstance(outputs, JudgeReasons) or (hasattr(outputs, "prediction") and hasattr(outputs, "conclusion")):
                    # Detailed judge output with more information
                    prediction = getattr(outputs, "prediction", None)
                    conclusion = getattr(outputs, "conclusion", "")
                    reasons = getattr(outputs, "reasons", [])

                    pred_color = "green" if prediction else "red"
                    pred_symbol = "✓" if prediction else "✗"

                    result.append(f"{pred_symbol} ", style=pred_color)
                    # Clean up whitespace in conclusion and show more text
                    clean_conclusion = " ".join(conclusion.strip().split())
                    result.append(f"{clean_conclusion[:300]}", style="white")

                    # Show first 2 reasons if available with better formatting
                    if reasons and len(reasons) > 0:
                        result.append(f"\n{' ' * 25}├ ", style="dim")  # Indent for reasons
                        clean_reason1 = " ".join(str(reasons[0]).strip().split())
                        result.append(f"{clean_reason1[:200]}", style="dim white")
                        if len(reasons) > 1:
                            result.append(f"\n{' ' * 25}├ ", style="dim")
                            clean_reason2 = " ".join(str(reasons[1]).strip().split())
                            result.append(f"{clean_reason2[:200]}", style="dim white")
                        if len(reasons) > 2:
                            result.append(f"\n{' ' * 25}└ ", style="dim")
                            result.append(f"({len(reasons) - 2} more reasons...)", style="dim")
                    content_added = True

                elif isinstance(outputs, QualResults) or (hasattr(outputs, "assessed_call_id") and hasattr(outputs, "correctness")):
                    # Compact scorer output - show call_id and overall score only
                    call_id = getattr(outputs, "assessed_call_id", "unknown")
                    correctness = getattr(outputs, "correctness", 0) or 0

                    # Extract short call ID for display
                    short_call_id = call_id[-8:] if len(call_id) > 8 else call_id

                    result.append(f"[{short_call_id}] ", style="dim")
                    score_color = "green" if correctness > 0.7 else "yellow" if correctness > 0.4 else "red"
                    result.append(f"Score: {correctness:.2f}", style=score_color)

                    # Show count of correct assessments if available
                    if hasattr(outputs, "assessments") and outputs.assessments:
                        correct_count = sum(1 for a in outputs.assessments if a.correct)
                        total_count = len(outputs.assessments)
                        result.append(f" ({correct_count}/{total_count})", style="dim")

                    content_added = True

                elif isinstance(outputs, Differences) or hasattr(outputs, "conclusion"):
                    conclusion = getattr(outputs, "conclusion", "")
                    result.append("DIFF: ", style="yellow")
                    result.append(f"{conclusion[:100]}", style="white")
                    content_added = True

                elif isinstance(outputs, Record) or hasattr(outputs, "record_id"):
                    record_id = getattr(outputs, "record_id", "unknown")
                    result.append(f"REC:{record_id} ", style="cyan")
                    # Show first 100 chars of content if available
                    if hasattr(outputs, "content") and outputs.content:
                        preview = outputs.content[:100].replace("\n", " ")
                        result.append(f"{preview}...", style="dim white")
                    content_added = True

            elif isinstance(message, AgentAnnouncement):
                # Format agent announcements
                ann_type = getattr(message, "announcement_type", "unknown")
                status = getattr(message, "status", "unknown")
                
                # Choose icon and color based on announcement type and status
                if ann_type == "initial":
                    icon = "🆕"
                    color = "green" if status == "joining" else "yellow"
                elif ann_type == "response":
                    icon = "↩️"
                    color = "bright_blue"
                elif ann_type == "update":
                    icon = "🔄"
                    color = "yellow" if status == "leaving" else "bright_blue"
                else:
                    icon = "📢"
                    color = "white"
                
                result.append(f"{icon} ", style=color)
                result.append(f"[{status.upper()}] ", style=color)
                
                # Show agent role and tools
                if hasattr(message, "agent_config") and message.agent_config:
                    role = getattr(message.agent_config, "role", "UNKNOWN")
                    result.append(f"{role} ", style="bold")
                
                # Show available tools if any
                tools = getattr(message, "available_tools", [])
                if tools:
                    tools_str = ", ".join(tools[:3])  # Show first 3 tools
                    if len(tools) > 3:
                        tools_str += f" +{len(tools)-3}"
                    result.append(f"[{tools_str}] ", style="dim cyan")
                
                # Show content
                content = getattr(message, "content", "")
                if content:
                    result.append(content[:100], style="white")
                
                content_added = True

            elif isinstance(message, UIMessage):
                # Check if this is an agent list command response
                registry = getattr(message, "agent_registry_summary", None)
                if registry:
                    # Special formatting for agent registry display
                    result.append("📋 AGENTS: ", style="bright_yellow")
                    result.append(f"{len(registry)} active", style="white")
                    
                    # If content is "!agents" or similar, show detailed list
                    content = getattr(message, "content", "")
                    if "!agents" in content.lower() or "agent" in content.lower():
                        # Add line break for detailed view
                        result.append("\n", style="")
                        
                        # Show each agent in registry
                        for agent_id, info in list(registry.items())[:5]:  # Show first 5
                            role = info.get("role", "UNKNOWN")
                            status = info.get("status", "unknown")
                            tools = info.get("tools", [])
                            model = info.get("model", "")
                            
                            # Indent for sub-items
                            result.append(" " * 25 + "├ ", style="dim")
                            
                            # Agent icon and name
                            agent_icon = get_agent_icon(role)
                            result.append(f"{agent_icon} {role}", style="bold white")
                            
                            # Status indicator
                            status_color = "green" if status == "active" else "yellow"
                            result.append(f" [{status}]", style=status_color)
                            
                            # Model tag if available
                            if model:
                                model_tag = get_model_tag(type('obj', (), {'agent_info': type('info', (), {'parameters': type('params', (), {'model': model})})})())
                                if model_tag:
                                    result.append(f" {model_tag}", style="dim")
                            
                            # Tools summary
                            if tools:
                                tools_str = f" ({len(tools)} tools)"
                                result.append(tools_str, style="dim cyan")
                            
                            result.append("\n", style="")
                        
                        if len(registry) > 5:
                            result.append(" " * 25 + f"└ ... and {len(registry)-5} more\n", style="dim")
                else:
                    # Standard UIMessage formatting
                    result.append("REQ: ", style="bright_yellow")
                    content = getattr(message, "content", "")
                    content_preview = content[:150].replace("\n", " ") if content else "No content"
                    result.append(content_preview, style="white")
                    if len(content) > 150:
                        result.append("...", style="dim")
                content_added = True

            elif isinstance(message, TaskProcessingStarted):
                # Skip all TaskProcessingStarted messages to reduce noise
                return None

            elif isinstance(message, TaskProcessingComplete) or (hasattr(message, "task_index") and hasattr(message, "is_error")):
                # Hide task progress messages to reduce noise - only show errors
                is_error = getattr(message, "is_error", False)
                if is_error:
                    task_index = getattr(message, "task_index", "?")
                    result.append(f"TASK {task_index} FAILED", style="red")
                    content_added = True
                else:
                    # Skip successful task completion messages
                    return None

            elif isinstance(message, ToolOutput):
                result.append(f"TOOL:{message.call_id} ", style="green")
                content_preview = str(message.content)[:100].replace("\n", " ") if message.content else "No output"
                result.append(content_preview, style="dim white")
                content_added = True

            elif hasattr(message, "content") and message.content:
                # Generic content - increased limit and better whitespace handling
                content_str = str(message.content)
                # Clean up excessive whitespace but preserve structure
                clean_content = " ".join(content_str.split())
                content_preview = clean_content[:300]  # Increased from 150
                result.append(content_preview, style="white")
                if len(clean_content) > 300:
                    result.append("...", style="dim")
                content_added = True

            # If no specific content was added, show the message type
            if not content_added:
                result.append(f"[{type(message).__name__}]", style="dim")

            return result

        except Exception as e:
            logger.error(f"Error formatting message type {type(message)} from {source}: {e}")
            # Fallback IRC-style error message
            error_text = Text()
            error_text.append(f"[{format_timestamp()}] ", style="dim")
            error_text.append("⚠ ", style="red")
            error_text.append("ERROR".ljust(16), style="red")
            error_text.append(" │ ", style="dim")
            error_text.append(f"Failed to format {type(message).__name__}: {str(e)[:100]}", style="red")
            return error_text

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,  # Usually unused by UI agent in listen
        **kwargs,
    ) -> None:
        """Displays messages received from other agents on the console."""
        logger.debug(f"{self.agent_name} received message from {source} via _listen.")
        # Format and display the message using the helper function.
        await self.callback_to_ui(message, source=source)

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        source: str = "unknown",  # Add source default
        **kwargs,
    ) -> OOBMessages | None:
        """Handles Out-Of-Band messages, displaying relevant ones."""
        logger.debug(f"{self.agent_name} received OOB message from {source}: {type(message).__name__}")
        # Check if the specific OOB message type is one we want to display.
        # Skip TaskProcessingStarted and successful TaskProcessingComplete messages to reduce noise
        if isinstance(message, TaskProcessingStarted):
            # Skip all TaskProcessingStarted messages
            return None
        elif isinstance(message, TaskProcessingComplete):
            if getattr(message, "is_error", False):
                # Only show failed tasks
                if formatted_msg := self._fmt_msg(message, source=source):
                    self._console.print(formatted_msg)
        elif isinstance(message, (UIMessage, ToolOutput, AgentAnnouncement)):
            if formatted_msg := self._fmt_msg(message, source=source):
                self._console.print(formatted_msg)
        else:
            # Log other OOB types at debug level if not explicitly formatted.
            logger.debug(f"ConsoleAgent ignoring unformatted OOB message type: {type(message).__name__}")

        # OOB handlers typically don't return responses unless specifically requested.
        return None

    async def _poll_input(self) -> None:
        """Continuously polls for user input from the console in a background task."""
        logger.info(f"{self.agent_name}: Starting console input polling...")
        current_prompt_lines: list[str] = []
        while True:
            try:
                # Get input asynchronously, allowing other tasks to run.
                # IRC-style prompt
                display_prompt = f"[{format_timestamp()}] 👤 user             │ "
                user_input = await ainput(display_prompt)

                # Handle special commands
                input_lower = user_input.strip().lower()
                
                # Handle exit command
                if input_lower == "exit":
                    logger.info("User requested exit.")
                    # TODO: How to signal exit cleanly to the orchestrator? Raising KeyboardInterrupt might be harsh.
                    # Maybe send a specific ManagerMessage or signal?
                    # For now, simulate interrupt. Need a better mechanism.
                    # Find the main task and cancel it? Difficult from here.
                    # Send a special message via callback?
                    # await self.callback_to_groupchat(ManagerMessage(confirm=False, prompt="USER_EXIT_REQUEST"))
                    raise KeyboardInterrupt  # Temporary way to stop, might need refinement
                
                # Handle agent list command
                elif input_lower in ["!agents", "!list", "!who"]:
                    logger.info("User requested agent list.")
                    # Send a special message requesting agent list
                    response = ManagerMessage(
                        confirm=False, 
                        interrupt=False, 
                        content="!agents",
                        request_agent_list=True  # Special flag for agent list request
                    )
                    await self.callback_to_groupchat(response)
                    continue

                # Handle confirmation/negation based on input
                cleaned_input = re.sub(r"\W", "", user_input).lower()
                is_negation = cleaned_input in ["x", "n", "no", "cancel", "abort", "stop", "quit", "q"]
                is_confirmation = not user_input.strip()  # Empty line confirms

                if is_negation:
                    logger.info("User input interpreted as NEGATIVE confirmation.")
                    response = ManagerMessage(confirm=False, interrupt=False, content="\n".join(current_prompt_lines))
                    current_prompt_lines = []  # Reset prompt buffer
                    await self.callback_to_groupchat(response)
                elif is_confirmation:
                    # If we have accumulated feedback in the prompt buffer, this is a confirmation WITH feedback,
                    # which should be treated as an interruption requiring the Conductor's attention
                    has_feedback = bool(current_prompt_lines)
                    if has_feedback:
                        logger.info("User input interpreted as POSITIVE confirmation with feedback (interrupt).")
                        response = ManagerMessage(confirm=True, interrupt=True, content="\n".join(current_prompt_lines))
                    else:
                        logger.info("User input interpreted as POSITIVE confirmation (no feedback).")
                        response = ManagerMessage(confirm=True, interrupt=False, content=None)

                    current_prompt_lines = []  # Reset prompt buffer
                    await self.callback_to_groupchat(response)
                else:
                    # Non-empty, non-negation input: add to multi-line buffer
                    logger.debug(f"User input added to buffer: '{user_input}'")
                    current_prompt_lines.append(user_input)
                    # Don't send response yet, wait for empty line to confirm multi-line input
                    continue  # Go back to prompt for more input

                # Optional: Signal heartbeat or task completion after processing input via callback
                # await self.callback_to_groupchat(TaskProcessingComplete(...)) # If input completes a "task"
                # await self.callback_to_groupchat(HeartBeat(go_next=True)) # If input allows flow to proceed

                await asyncio.sleep(0.1)  # Small sleep to prevent tight loop if needed

            except asyncio.CancelledError:
                logger.info(f"{self.agent_name}: Input polling task cancelled.")
                break  # Exit loop cleanly on cancellation
            except RuntimeError as e:
                # Happens sometimes if multiple consoles are called in parallel. Best to
                # kill the agent and hope we can go on.
                raise FatalError(f"Runtime error in CLIUserAgent: {e}")
            except Exception as e:
                # Log errors during input polling but try to continue
                logger.error(f"{self.agent_name}: Error polling console input: {e}")
                # Consider adding a delay before retrying after an error
                await asyncio.sleep(1)
                # Re-raise if it's KeyboardInterrupt to allow stopping the application
                if isinstance(e, KeyboardInterrupt):
                    raise

    async def initialize(self, callback_to_groupchat: Callable[..., Awaitable[None]], **kwargs) -> None:
        """Initializes the agent and starts the background input polling task if a callback is provided.

        Args:
            callback_to_groupchat: The async function to call when user input is received.
                            Expected signature: `async def callback(response: ManagerMessage)`
            **kwargs: Additional keyword arguments passed to the base class initializer.

        """
        # Call base class initialize if needed
        await super().initialize(ui_type="console", callback_to_groupchat=callback_to_groupchat, **kwargs)

        # Initialize the console and set up the input task.
        logger.debug(f"Initializing {self.agent_name}...")
        self.callback_to_groupchat = callback_to_groupchat
        # Ensure any existing task is cancelled before starting a new one (e.g., on reset)
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass  # Expected

        if self.callback_to_groupchat:
            # Display IRC-style welcome message
            welcome_text = Text()
            welcome_text.append(f"[{format_timestamp()}] ", style="dim")
            welcome_text.append("⚙ ", style="white")
            welcome_text.append("system".ljust(16), style="yellow")
            welcome_text.append(" │ ", style="dim")
            welcome_text.append("Console UI initialized. Commands: 'exit' to quit, '!agents' to list agents, empty line to confirm, 'n'/'q' to cancel.", style="green")
            self._console.print(welcome_text)

            # Start the background task to poll for console input.
            self._input_task = asyncio.create_task(self._poll_input())
            logger.debug(f"{self.agent_name}: Input polling task created.")
        else:
            # If no callback, input polling is disabled.
            logger.warning(f"{self.agent_name}: Initialized without callback_to_groupchat. Console input polling disabled.")
            self._input_task = None

    async def cleanup(self) -> None:
        """Cleans up resources, primarily by cancelling the input polling task."""
        logger.debug(f"Cleaning up {self.agent_name}...")
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                # Wait for the task to acknowledge cancellation
                await self._input_task
            except asyncio.CancelledError:
                logger.info(f"{self.agent_name}: Console input task successfully cancelled.")
            except Exception as e:
                # Log if waiting for cancellation fails unexpectedly
                logger.error(f"{self.agent_name}: Error during input task cleanup: {e}")
        else:
            logger.debug(f"{self.agent_name}: No active input task to cancel.")
        # Call base class cleanup if needed
        # await super().cleanup()

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
            """Resets the agent state, including cancelling the input task."""
            logger.info(f"{self.agent_name}: Resetting agent state...")
            # Cancel the existing input task if it's running
            if self._input_task and not self._input_task.done():
                self._input_task.cancel()
                try:
                    await self._input_task
                except asyncio.CancelledError:
                    logger.debug(f"{self.agent_name}: Input task cancelled during reset.")
                except Exception as e:
                    logger.error(f"{self.agent_name}: Error cancelling input task during reset: {e}")
            self._input_task = None  # Ensure the task reference is cleared
            # Note: The input task will be restarted by `initialize` if called again after reset.
            # Call base class reset if needed
            # await super().on_reset(cancellation_token)

    def make_callback(self) -> Callable[..., Awaitable[None]]:
        """Create a callback function for the UI agent.

        Returns:
            A callable that can be used as a callback function

        """
        return self.callback_to_ui
