"""Message formatting utilities for displaying messages in the frontend UI.
This module handles converting Pydantic model objects to appropriate HTML/UI representations.
"""

import hashlib
import re
from typing import Any

from buttermilk._core.agent import AgentTrace
from buttermilk._core.contract import (
    ConductorResponse,
    ManagerRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
    ToolOutput,
)
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.bm import logger

# Map to track message IDs and their scores
# Key: message_id, Value: list of QualResults objects
message_scores: dict[str, list[QualResults]] = {}

# Agent styling configuration
AGENT_STYLES = {
    "default": {"avatar": "👤", "color": "#6c757d", "background": "#f8f9fa", "border": "1px solid #dee2e6"},
    "judge": {"avatar": "⚖️", "color": "#495057", "background": "#e9ecef", "border": "1px solid #ced4da"},
    "scorer": {"avatar": "📊", "color": "#212529", "background": "#f8f9fa", "border": "1px solid #adb5bd"},
    "assistant": {"avatar": "🤖", "color": "#0d6efd", "background": "#e7f1ff", "border": "1px solid #b6d4fe"},
    "describer": {"avatar": "🔍", "color": "#6610f2", "background": "#eee6ff", "border": "1px solid #d0bfff"},
    "fetch": {"avatar": "🌐", "color": "#fd7e14", "background": "#fff3cd", "border": "1px solid #ffecb5"},
    "imagegen": {"avatar": "🎨", "color": "#d63384", "background": "#f7d6e6", "border": "1px solid #efadce"},
    "reasoning": {"avatar": "🧠", "color": "#20c997", "background": "#d1f2eb", "border": "1px solid #a3e4d7"},
    "scraper": {"avatar": "🕸️", "color": "#6f42c1", "background": "#e6d9f2", "border": "1px solid #d2b5e8"},
    "spy": {"avatar": "🕵️", "color": "#212529", "background": "#e2e3e5", "border": "1px solid #c9cccf"},
    "synthesiser": {"avatar": "🔄", "color": "#0c4128", "background": "#d0f0e0", "border": "1px solid #a0d0b0"},
    "tool": {"avatar": "🔧", "color": "#198754", "background": "#d1e7dd", "border": "1px solid #badbcc"},
    "instructions": {"avatar": "📝", "color": "#0dcaf0", "background": "#cff4fc", "border": "1px solid #9eeaf9"},
}

# Color palette for agents without specific styles
COLOR_PALETTE = [
    "#4285F4", "#EA4335", "#FBBC05", "#34A853",  # Google colors
    "#1877F2", "#E4405F", "#5865F2", "#FF9900",  # Social media blues/reds
    "#00ACC1", "#8BC34A", "#FF5722", "#7E57C2",  # Material design
    "#26A69A", "#EC407A", "#AB47BC", "#42A5F5",  # More material
]

# Judge emojis with different skin tones and variations
JUDGE_AVATARS = [
    "👨‍⚖️", "👨🏻‍⚖️", "👨🏼‍⚖️", "👨🏽‍⚖️", "👨🏾‍⚖️", "👨🏿‍⚖️",
    "👩‍⚖️", "👩🏻‍⚖️", "👩🏼‍⚖️", "👩🏽‍⚖️", "👩🏾‍⚖️", "👩🏿‍⚖️",
    "🧑‍⚖️", "🧑🏻‍⚖️", "🧑🏼‍⚖️", "🧑🏽‍⚖️", "🧑🏾‍⚖️", "🧑🏿‍⚖️",
    "⚖️",  # Traditional scales of justice (fallback)
]


def _get_message_hash(message_id: str) -> str:
    """Generate a consistent color from a message ID"""
    if not message_id:
        return "#777777"  # Default gray

    hash_val = int(hashlib.md5(message_id.encode()).hexdigest(), 16)
    return COLOR_PALETTE[hash_val % len(COLOR_PALETTE)]


def _get_score_color(score: float) -> str:
    """Get color for a score value"""
    if score > 0.8:
        return "#28a745"  # Strong green
    if score > 0.6:
        return "#5cb85c"  # Light green
    if score > 0.4:
        return "#ffc107"  # Yellow
    if score > 0.2:
        return "#ff9800"  # Orange
    return "#dc3545"  # Red


def _format_score_indicator(score: float) -> str:
    """Format a score as a colored indicator"""
    color = _get_score_color(score)
    score_percent = f"{int(score * 100)}%"

    return f"""
    <div style="display:inline-flex; align-items:center; background:#f8f9fa; padding:2px 8px; border-radius:12px; margin-left:5px; font-size:0.85em;">
        <span style="display:inline-block; width:8px; height:8px; background-color:{color}; border-radius:50%; margin-right:4px;"></span>
        <span style="font-weight:bold;">{score_percent}</span>
    </div>
    """


def _get_judge_avatar(agent_id: str) -> str:
    """Generate a consistent judge avatar based on agent ID"""
    if not agent_id:
        return "⚖️"  # Default fallback

    hash_val = int(hashlib.md5(agent_id.encode()).hexdigest(), 16)
    return JUDGE_AVATARS[hash_val % len(JUDGE_AVATARS)]


def _format_record(record: Record) -> str:
    """Format a Record object for display
    
    Args:
        record: The Record object to format
        
    Returns:
        str: HTML representation of the record

    """
    # Create a snippet version for display
    content_str = str(record.content) if record.content else ""
    snippet = content_str[:200] + "..." if len(content_str) > 200 else content_str

    # Format metadata
    metadata_parts = []
    if record.metadata:
        if record.title:
            metadata_parts.append(f"<h3 class='text-lg font-semibold'>{record.title}</h3>")
        metadata_parts.append(f"<span class='text-xs text-gray-500'>ID: {record.record_id}</span>")

        metadata_html = []
        for k, v in record.metadata.items():
            if k not in ["title", "fetch_timestamp_utc", "fetch_source_id"]:
                metadata_html.append(f"<div><span class='font-medium'>{k}:</span> {v}</div>")

        if metadata_html:
            metadata_parts.append(f"<div class='text-sm mt-1 space-y-1'>{''.join(metadata_html)}</div>")

    # Create HTML with the full content in a tooltip
    return f"""
    <div class="group relative">
        <div class="p-3 bg-white rounded-md border border-gray-200">
            {''.join(metadata_parts)}
            <div class="mt-2 text-sm text-gray-700">{snippet}</div>
        </div>
        <div class="hidden group-hover:block absolute left-0 top-full z-10 w-96 p-4 bg-gray-800 text-white text-sm rounded shadow-lg overflow-y-auto max-h-80">
            <div class="mb-2 font-bold">{record.title or 'Record Content'}</div>
            <div>{record.text}</div>
        </div>
    </div>
    """


def _format_message_with_style(content: str, agent_info: Any, message_id: str | None = None) -> str:
    """Apply styling to a message based on agent info
    
    Args:
        content: The message content
        agent_info: Information about the agent
        message_id: Optional ID for the message
        
    Returns:
        str: HTML representation of the styled message

    """
    # Get the predefined style for this agent role
    role = agent_info.role.lower() if hasattr(agent_info, "role") else "default"
    agent_style = AGENT_STYLES.get(role, AGENT_STYLES["default"])

    # Extract styling elements
    background = agent_style.get("background", "#f8f9fa")
    border = agent_style.get("border", "1px solid #dee2e6")

    # Get avatar with special handling for judge role
    if role == "judge":
        avatar = _get_judge_avatar(agent_info.id)
    else:
        avatar = agent_style.get("avatar", "👤")

    color = agent_style.get("color", "#6c757d")
    name = agent_info.name if hasattr(agent_info, "name") else role.capitalize()
    agent_color = _get_message_hash(agent_info.id if hasattr(agent_info, "id") else "default")

    # Check if there's a score for this message
    score_html = ""
    if message_id and message_id in message_scores:
        for qual_result in message_scores.get(message_id, []):
            score_color = _get_score_color(qual_result.correctness or 0)
            score_text = qual_result.score_text
            score_html += f"""
            <div class="score-badge" style="position:absolute; top:-8px; right:-8px; background:#f8f9fa; padding:2px 8px; border-radius:12px; border:1px solid #dee2e6; font-size:0.85em;">
                <span style="display:inline-block; width:8px; height:8px; background-color:{score_color}; border-radius:50%; margin-right:4px;"></span>
                <span style="font-weight:bold;">{score_text}</span>
            </div>
            """

    return f"""
    <div id="{message_id}" style="display:flex; margin-bottom:15px; position:relative;">
        <div style="flex-shrink:0; margin-right:10px; font-size:1.5em;">{avatar}</div>
        <div style="flex-grow:1;">
            <div style="font-weight:bold; margin-bottom:1px; color:{agent_color};">{name}</div>
            <div style="padding:1px; background:{background}; color:{color}; border-radius:2px; border:{border}; position:relative;">
                {content}
                {score_html}
            </div>
        </div>
    </div>
    """


def _format_agent_reasons(reasons: JudgeReasons, message: AgentTrace | None = None) -> str:
    """Format JudgeReasons using the UIService formatter
    
    Args:
        reasons: The JudgeReasons to format
        message: Optional AgentTrace containing the original message
        
    Returns:
        str: HTML representation of the judge reasons

    """
    from buttermilk.web.fastapi_frontend.services.ui_service import UIService
    return UIService.format_judge_reasons_html(reasons)


def _markdown_to_html(content: str) -> str:
    """Convert markdown content to HTML
    
    Args:
        content: Markdown content to convert
        
    Returns:
        str: HTML representation of the markdown

    """
    if not content:
        return ""

    # Process code blocks with triple backticks
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
        hashtags = "#" * i
        pattern = rf"^{hashtags} (.+)$"
        replacement = f"<h{i}>\\1</h{i}>"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content


def _format_message_for_client(message: Any) -> dict[str, Any] | str | None:
    """Format different message types for web client consumption with improved styling.
    
    Args:
        message: The message to format, which should be a Pydantic object
        
    Returns:
        Union[Dict[str, Any], str, None]: Formatted message data for the client

    """
    # Skip message types that shouldn't be displayed in the UI
    if isinstance(message, (TaskProgressUpdate, TaskProcessingStarted, TaskProcessingComplete, ConductorResponse)):
        return None

    # Handle Record objects
    if isinstance(message, Record):
        return _format_record(message)

    # Only process AgentTrace objects from here on
    if not isinstance(message, AgentTrace):
        return None

    content = None

    # Process based on the output type
    if isinstance(message.outputs, QualResults):
        logger.debug(f"Detected score message from: {message.agent_info.name}")
        if message.parent_call_id in message_scores:
            logger.debug(f"Sending score update with agent={message.outputs.agent_id}, assessor={message.outputs.assessor}")
            # Store the QualResults directly
            message_scores[message.parent_call_id].append(message.outputs)

            # Always use the direct structured format for frontend updates
            return {
                "type": "score_update",
                "agent_id": message.outputs.agent_id,
                "assessor": message.outputs.assessor,
                "score_data": {
                    "correctness": message.outputs.correctness,
                    "score_text": message.outputs.score_text,
                    "assessments": [assessment.model_dump() for assessment in message.outputs.assessments],
                },
            }

    elif isinstance(message.outputs, JudgeReasons):
        # Initialize score tracking for this message
        if message.call_id not in message_scores:
            message_scores[message.call_id] = []
        # Format the reasons
        content = _format_agent_reasons(message.outputs, message)

    elif isinstance(message.outputs, ToolOutput):
        # Format tool output with badge
        tool_name = getattr(message.outputs, "function_name", "unknown_tool")
        badge = f'<span style="display:inline-block; padding:2px 8px; background:#6c757d; color:white; border-radius:4px; margin-bottom:5px;">{tool_name}</span>'
        content = badge + "<br/>" + message.outputs.content

    elif isinstance(message.outputs, ManagerRequest):
        # Use content from ManagerRequest if available
        content = message.outputs.content if hasattr(message.outputs, "content") and message.outputs.content else None
        if not content:
            return None

    else:
        return None

    # Convert markdown in content to HTML if needed
    if content and isinstance(content, str):
        content = _markdown_to_html(content)

    # Apply styling to the message
    if content is None:
        content = ""  # Convert None to empty string to avoid errors

    styled_content = _format_message_with_style(content, message.agent_info, message.call_id)

    # Return standardized chat message format
    return {
        "type": "chat_message",
        "content": styled_content,
        "agent_info": {
            "role": message.agent_info.role,
            "name": message.agent_info.name,
            "id": message.agent_info.id,
            "message_id": message.call_id,
        },
    }
