

import json

from buttermilk._core.agent import AgentOutput, ManagerRequest, ToolOutput
from buttermilk._core.contract import (
    ConductorResponse,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
)
from buttermilk._core.types import Record

# Store the scores for aggregation
score_messages: list[float] = []

# Define agent styling
AGENT_STYLES = {
    "default": {
        "avatar": "👤",
        "color": "#6c757d",
        "background": "#f8f9fa",
        "border": "1px solid #dee2e6",
    },
    "judge": {
        "avatar": "⚖️",
        "color": "#495057",
        "background": "#e9ecef",
        "border": "1px solid #ced4da",
    },
    "scorer": {
        "avatar": "📊",
        "color": "#212529",
        "background": "#f8f9fa",
        "border": "1px solid #adb5bd",
    },
    "assistant": {
        "avatar": "🤖",
        "color": "#0d6efd",
        "background": "#e7f1ff",
        "border": "1px solid #b6d4fe",
    },
    "describer": {
        "avatar": "🔍",
        "color": "#6610f2",
        "background": "#eee6ff",
        "border": "1px solid #d0bfff",
    },
    "fetch": {
        "avatar": "🌐",
        "color": "#fd7e14",
        "background": "#fff3cd",
        "border": "1px solid #ffecb5",
    },
    "imagegen": {
        "avatar": "🎨",
        "color": "#d63384",
        "background": "#f7d6e6",
        "border": "1px solid #efadce",
    },
    "reasoning": {
        "avatar": "🧠",
        "color": "#20c997",
        "background": "#d1f2eb",
        "border": "1px solid #a3e4d7",
    },
    "scraper": {
        "avatar": "🕸️",
        "color": "#6f42c1",
        "background": "#e6d9f2",
        "border": "1px solid #d2b5e8",
    },
    "spy": {
        "avatar": "🕵️",
        "color": "#212529",
        "background": "#e2e3e5",
        "border": "1px solid #c9cccf",
    },
    "tool": {
        "avatar": "🔧",
        "color": "#198754",
        "background": "#d1e7dd",
        "border": "1px solid #badbcc",
    },
    "instructions": {
        "avatar": "📝",
        "color": "#0dcaf0",
        "background": "#cff4fc",
        "border": "1px solid #9eeaf9",
    },
}


def _get_agent_role(message) -> str:
    """Extract the agent role from the message."""
    if isinstance(message, AgentOutput):
        # Safely get role attribute if it exists
        if hasattr(message, "role"):
            return getattr(message, "role", "default").lower()
    if isinstance(message, ToolOutput):
        return "tool"
    if isinstance(message, ManagerRequest):
        return "instructions"
    if isinstance(message, Record):
        # Try to extract role information if available
        if hasattr(message, "role"):
            return getattr(message, "role", "default").lower()
    return "default"


def _is_score_message(message) -> bool:
    """Check if the message is a scoring message."""
    if isinstance(message, AgentOutput) and hasattr(message, "role"):
        role = getattr(message, "role", "").lower()
        return role == "scorer"
    return False


def _format_score_indicator(score) -> str:
    """Format a score as a colored square."""
    if isinstance(score, (int, float)):
        color = "#28a745" if score > 0.5 else "#dc3545"  # Green for high scores, red for low
        return f'<span class="score-indicator" style="display:inline-block; width:12px; height:12px; background-color:{color}; margin-left:5px;"></span>'
    return ""


def _format_message_with_style(content: str, agent_style: dict[str, str]) -> str:
    """Apply styling to a message based on agent type."""
    avatar = agent_style.get("avatar", "👤")
    color = agent_style.get("color", "#6c757d")
    background = agent_style.get("background", "#f8f9fa")
    border = agent_style.get("border", "1px solid #dee2e6")

    return f"""
    <div style="display:flex; margin-bottom:10px;">
        <div style="flex-shrink:0; margin-right:10px; font-size:1.5em;">{avatar}</div>
        <div style="flex-grow:1; padding:10px; background:{background}; color:{color}; border-radius:8px; border:{border};">
            {content}
        </div>
    </div>
    """


def _format_scores_summary() -> str:
    """Format the aggregated scores as a series of colored squares."""
    if not score_messages:
        return ""

    score_indicators = ""
    for score in score_messages:
        try:
            # Try to extract numeric score value
            score_value = float(score) if isinstance(score, (int, float, str)) else 0.5
            color = "#28a745" if score_value > 0.5 else "#dc3545"
            score_indicators += f'<span style="display:inline-block; width:12px; height:12px; background-color:{color}; margin-right:3px;"></span>'
        except:
            # If we can't extract a numeric value, use a neutral color
            score_indicators += '<span style="display:inline-block; width:12px; height:12px; background-color:#6c757d; margin-right:3px;"></span>'

    return f"""
    <div style="display:flex; justify-content:flex-end; margin-top:5px;">
        <div style="padding:5px; background:#f8f9fa; border-radius:4px; border:1px solid #dee2e6;">
            {score_indicators}
        </div>
    </div>
    """


def _extract_score(message) -> float | None:
    """Extract a numeric score from a message if possible."""
    if isinstance(message, AgentOutput):
        if isinstance(message.outputs, dict) and "score" in message.outputs:
            return message.outputs["score"]
        if isinstance(message.outputs, (int, float)):
            return float(message.outputs)
        if isinstance(message.outputs, str):
            try:
                # Try to parse JSON string
                data = json.loads(message.outputs)
                if isinstance(data, dict) and "score" in data:
                    return data["score"]
            except:
                # If it's not JSON, try to convert the string to a float
                try:
                    return float(message.outputs)
                except:
                    pass
    return None


def _format_message_for_client(message) -> str | None:
    """Format different message types for web client consumption with improved styling.
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string or None if message shouldn't be sent

    """
    # Default values
    msg_type = "message"
    content = None
    agent_role = _get_agent_role(message)

    # Ignore certain message types (don't display in chat)
    if isinstance(message, (TaskProgressUpdate, TaskProcessingStarted, TaskProcessingComplete, ConductorResponse)):
        return None

    # Check for scoring message and add to aggregation
    if _is_score_message(message):
        score = _extract_score(message)
        if score is not None:
            score_messages.append(score)

    # Format based on message type
    if isinstance(message, AgentOutput):
        msg_type = "agent_output"
        if isinstance(message.outputs, str):
            content = message.outputs
        else:
            # For complex outputs, serialize to JSON
            try:
                content = json.dumps(message.outputs, indent=2) if message.outputs else ""
            except:
                content = str(message.outputs)

    elif isinstance(message, Record):  # Add handling for Record
        content = message.text

    elif isinstance(message, ToolOutput):
        msg_type = "tool_output"
        content = message.content
        # Additional info
        tool_name = getattr(message, "name", "unknown_tool")

        # Add tool name as a badge
        content = f'<span style="display:inline-block; padding:2px 8px; background:#6c757d; color:white; border-radius:4px; margin-bottom:5px;">{tool_name}</span><br/>{content}'

    elif isinstance(message, ManagerRequest):
        content = message.content

    else:
        # Unhandled message type
        return None

    # Convert markdown to HTML in message content
    # The content may already have some HTML, so we need to handle it carefully
    # First check if content exists and is a string
    if content and isinstance(content, str):
        # Process code blocks with triple backticks
        import re

        # Process code blocks
        code_pattern = r"```(.*?)\n(.*?)```"

        def code_replace(match):
            language = match.group(1).strip() or ""
            code = match.group(2)
            return f'<pre><code class="language-{language}">{code}</code></pre>'

        content = re.sub(code_pattern, code_replace, content, flags=re.DOTALL)

        # Process inline code
        inline_code_pattern = r"`([^`]+)`"
        content = re.sub(inline_code_pattern, r"<code>\1</code>", content)

        # Process headers (# Header)
        for i in range(6, 0, -1):
            pattern = r"^{} (.+)$".format("#" * i)
            content = re.sub(pattern, rf"<h{i}>\1</h{i}>", content, flags=re.MULTILINE)

        # Process lists
        list_pattern = r"^[\*\-] (.+)$"
        content = re.sub(list_pattern, r"<li>\1</li>", content, flags=re.MULTILINE)

        # Wrap list items in ul tags (simplistic but effective for most cases)
        content = re.sub(r"(<li>.*?</li>)", r"<ul>\1</ul>", content, flags=re.DOTALL)

        # Process links [text](url)
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        content = re.sub(link_pattern, r'<a href="\2" target="_blank">\1</a>', content)

        # Process bold **text**
        bold_pattern = r"\*\*([^*]+)\*\*"
        content = re.sub(bold_pattern, r"<strong>\1</strong>", content)

        # Process italic *text*
        italic_pattern = r"\*([^*]+)\*"
        content = re.sub(italic_pattern, r"<em>\1</em>", content)

        # Convert newlines to <br> tags
        content = content.replace("\n", "<br>")

    # Get the style for this agent/message type
    agent_style = AGENT_STYLES.get(agent_role, AGENT_STYLES["default"])

    # Apply styling to the message (ensure content is not None)
    if content is None:
        content = ""  # Convert None to empty string to avoid type errors
    styled_content = _format_message_with_style(content, agent_style)

    # If it's a scorer message, don't show the content, just the score indicator
    if _is_score_message(message):
        score = _extract_score(message)
        if score is not None:
            return _format_score_indicator(score)

    # Add scores summary if we have accumulated scores
    if score_messages and not _is_score_message(message):
        styled_content += _format_scores_summary()
        # Clear scores after showing summary
        score_messages.clear()

    return styled_content
