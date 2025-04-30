import hashlib
import json
import re
from typing import Any, TypedDict

from buttermilk._core.agent import AgentOutput, ManagerRequest, ToolOutput
from buttermilk._core.contract import (
    ConductorResponse,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
)
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import AgentReasons
from buttermilk.bm import logger


# Define score data structure with required fields
class ScoreData(TypedDict):
    answer_id: str
    agent_id: str  # Add agent_id
    assessor: str  # Add assessor
    score: float
    score_text: str
    color: str
    assessments: list[Any]


# Map to track which messages have been scored
# Key: answer_id, Value: message_id (used to link scores to messages)
scored_messages: dict[str, str] = {}

# Map to track message IDs and their scores
# Key: message_id, Value: score info with specific fields
message_scores: dict[str, ScoreData] = {}

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
    "synthesiser": {
        "avatar": "🔄",
        "color": "#0c4128",
        "background": "#d0f0e0",
        "border": "1px solid #a0d0b0",
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

# Generate a few more vibrant colors for agents that don't have specific styles
COLOR_PALETTE = [
    "#4285F4", "#EA4335", "#FBBC05", "#34A853",  # Google colors
    "#1877F2", "#E4405F", "#5865F2", "#FF9900",  # Social media blues/reds
    "#00ACC1", "#8BC34A", "#FF5722", "#7E57C2",  # Material design
    "#26A69A", "#EC407A", "#AB47BC", "#42A5F5",  # More material
]


def _get_agent_info(message) -> dict:
    """Extract agent information from the message."""
    agent_info = {
        "role": "default",
        "name": "Unknown",
        "id": "",
        "color": None,
        "message_id": None,
    }

    if isinstance(message, AgentOutput):
        # Generate a stable message ID
        agent_info["message_id"] = message.call_id if hasattr(message, "call_id") else ""

        # Extract role if available
        if hasattr(message, "role"):
            agent_info["role"] = getattr(message, "role", "default").lower()

        # Try to get agent_info object
        if hasattr(message, "agent_info") and message.agent_info:
            info = message.agent_info
            agent_info["name"] = getattr(info, "name", "Agent")
            agent_info["id"] = getattr(info, "id", "")

            # If the message has a custom color in agent_info, use it
            if hasattr(info, "parameters") and isinstance(info.parameters, dict) and "color" in info.parameters:
                agent_info["color"] = info.parameters["color"]

        # For answer tracking with scores
        if hasattr(message, "call_id"):
            agent_info["answer_id"] = message.call_id

    elif isinstance(message, ToolOutput):
        agent_info["role"] = "tool"
        agent_info["name"] = getattr(message, "name", "Tool")
        agent_info["message_id"] = getattr(message, "call_id", "")

    elif isinstance(message, ManagerRequest):
        agent_info["role"] = "instructions"
        agent_info["name"] = "Instructions"

    elif isinstance(message, Record):
        # Try to extract role information if available
        if hasattr(message, "role"):
            agent_info["role"] = getattr(message, "role", "default").lower()
        agent_info["name"] = "Record"

    return agent_info


def _get_agent_role(message) -> str:
    """Extract the agent role from the message."""
    return _get_agent_info(message)["role"]


def _is_score_message(message) -> bool:
    """Check if the message is a scoring message."""
    if not isinstance(message, AgentOutput):
        return False

    # Check role first
    if hasattr(message, "role"):
        role = getattr(message, "role", "").lower()
        if role == "scorer":
            return True

    # Check if outputs is QualResults
    if hasattr(message, "outputs") and isinstance(message.outputs, QualResults):
        return True

    return False


def _get_message_hash(message_id: str) -> str:
    """Generate a consistent color from a message ID"""
    if not message_id:
        return "#777777"  # Default gray

    # Generate a hash of the ID and use it to select a color
    hash_val = int(hashlib.md5(message_id.encode()).hexdigest(), 16)
    return COLOR_PALETTE[hash_val % len(COLOR_PALETTE)]


def _get_score_color(score: float) -> str:
    """Get color for a score value."""
    if score > 0.8:
        return "#28a745"  # Strong green
    if score > 0.6:
        return "#5cb85c"  # Light green
    if score > 0.4:
        return "#ffc107"  # Yellow
    if score > 0.2:
        return "#ff9800"  # Orange
    return "#dc3545"  # Red


def _handle_score_message(score_value: float, assessor_id: str, agent_id: str, qual_data: dict[str, Any] | None = None) -> str:
    logger.debug(f"Handling score message: agent_id={agent_id}, assessor_id={assessor_id}, score={score_value}")
    """Generate JavaScript to update the score sidebar for a score message, waiting for Alpine.js."""
    if not qual_data:
        # Create minimal score data if none provided
        qual_data = {
            "score": score_value,
            "score_text": f"{int(score_value * 100)}%",
            "color": _get_score_color(score_value),
            "assessor": assessor_id,
            # Add defaults for other ScoreData fields if needed by JS
            "answer_id": "",
            "agent_id": agent_id,
            "assessments": [],
        }

    # Safely embed data into the JavaScript string
    safe_agent_id = json.dumps(agent_id)
    safe_assessor_id = json.dumps(assessor_id)
    safe_qual_data = json.dumps(qual_data)

    # Generate JavaScript that waits for Alpine.js (or DOMContentLoaded)
    # before trying to access window.scoreData
    script = f"""
    <script>
        function addScoreWhenReady() {{
            if (window.scoreData && typeof window.scoreData.addScore === 'function') {{
                console.log('Alpine component found. Adding score for agent:', {safe_agent_id});
                try {{
                    window.scoreData.addScore(
                        {safe_agent_id},
                        {safe_assessor_id},
                        {safe_qual_data}
                    );
                }} catch (e) {{
                    console.error('Error calling window.scoreData.addScore:', e);
                }}
            }} else {{
                console.warn('Alpine component (window.scoreData) not ready yet. Score update deferred.');
                // Optionally add a timeout or retry mechanism if needed,
                // but alpine:initialized or DOMContentLoaded should cover most cases.
            }}
        }}

        // Check if Alpine is already initialized
        if (window.Alpine && window.Alpine.initialized) {{
             // Use requestAnimationFrame to ensure this runs after the current JS execution cycle
             requestAnimationFrame(addScoreWhenReady);
        }} else {{
             // Wait for Alpine to initialize
            document.addEventListener('alpine:initialized', addScoreWhenReady, {{ once: true }});
            // Fallback: Wait for the whole DOM to be ready if alpine:initialized doesn't fire
            document.addEventListener('DOMContentLoaded', () => {{
                // Check again in case alpine:initialized didn't fire but the component is now ready
                 if (!window.scoreData || typeof window.scoreData.addScore !== 'function') {{
                    console.warn('Alpine component still not found after DOMContentLoaded. Trying score update anyway or logging error.');
                    // Attempt one last time or log a final error
                    if (window.scoreData && typeof window.scoreData.addScore === 'function') {{
                       addScoreWhenReady();
                    }} else {{
                       console.error('Failed to find window.scoreData.addScore after multiple attempts.');
                    }}
                 }}
            }}, {{ once: true }});
        }}
    </script>
    """
    logger.debug(f"Generated score script (length: {len(script)}): waits for Alpine/DOM")
    return script


def _format_score_indicator(score, simple=False) -> str:
    """Format a score as a colored indicator."""
    if isinstance(score, (int, float)):
        color = _get_score_color(score)

        if simple:
            return f'<span style="display:inline-block; width:10px; height:10px; background-color:{color}; border-radius:2px; margin-right:3px;"></span>'
        score_percent = f"{int(score * 100)}%"
        return f"""
            <div style="display:inline-flex; align-items:center; background:#f8f9fa; padding:2px 8px; border-radius:12px; margin-left:5px; font-size:0.85em;">
                <span style="display:inline-block; width:8px; height:8px; background-color:{color}; border-radius:50%; margin-right:4px;"></span>
                <span style="font-weight:bold;">{score_percent}</span>
            </div>
            """
    return ""


def _format_message_with_style(content: str, agent_info: dict, message_id: str | None = None) -> str:
    """Apply styling to a message based on agent info."""
    # Get the predefined style for this agent role
    role = agent_info.get("role", "default")
    agent_style = AGENT_STYLES.get(role, AGENT_STYLES["default"])

    # Extract styling elements
    avatar = agent_style.get("avatar", "👤")
    background = agent_style.get("background", "#f8f9fa")
    border = agent_style.get("border", "1px solid #dee2e6")

    # Prefer custom color from agent_info if available, otherwise use role-based color
    color = agent_info.get("color") or agent_style.get("color", "#6c757d")

    # Get agent name - use provided name or generate from role
    name = agent_info.get("name", role.capitalize())

    # Calculate agent color for the header
    agent_color = _get_message_hash(agent_info.get("id", ""))

    # Check if there's a score for this message
    score_html = ""
    if message_id and message_id in message_scores:
        score_data = message_scores[message_id]
        score_html = f"""
        <div class="score-badge" style="position:absolute; top:-8px; right:-8px; background:#f8f9fa; padding:2px 8px; border-radius:12px; border:1px solid #dee2e6; font-size:0.85em;">
            <span style="display:inline-block; width:8px; height:8px; background-color:{score_data.get('color', '#777777')}; border-radius:50%; margin-right:4px;"></span>
            <span style="font-weight:bold;">{score_data.get('score_text', '?%')}</span>
        </div>
        """

    return f"""
    <div id="{message_id}" style="display:flex; margin-bottom:15px; position:relative;">
        <div style="flex-shrink:0; margin-right:10px; font-size:1.5em;">{avatar}</div>
        <div style="flex-grow:1;">
            <div style="font-weight:bold; margin-bottom:4px; color:{agent_color};">{name}</div>
            <div style="padding:10px; background:{background}; color:{color}; border-radius:8px; border:{border}; position:relative;">
                {content}
                {score_html}
            </div>
        </div>
    </div>
    """


def _format_qual_results(qual_results: QualResults) -> ScoreData | None:
    """Process QualResults for display."""
    answer_id = getattr(qual_results, "answer_id", "")
    agent_id = getattr(qual_results, "agent_id", "")  # Extract agent_id

    if not answer_id or not agent_id:  # Ensure agent_id is also present
        logger.warning(f"QualResults missing answer_id ('{answer_id}') or agent_id ('{agent_id}'). Cannot format score.")
        return None

    score = getattr(qual_results, "correctness", 0.0)  # Default to 0.0 if missing

    # Choose color based on score value
    color = _get_score_color(score)  # Use helper function

    score_text = f"{int(score * 100)}%"

    # convert from pydantic model
    assessments_data = []
    if assessments := getattr(qual_results, "assessments", []):
        assessments_data = [x.model_dump() for x in assessments]

    # Ensure the returned dictionary matches the ScoreData TypedDict structure
    # and includes agent_id
    score_data: ScoreData = {
        "answer_id": answer_id,
        "agent_id": agent_id,  # Add agent_id here
        "score": score,
        "score_text": score_text,
        "color": color,
        "assessments": assessments_data,
        "assessor": getattr(qual_results, "assessor", "scorer"),  # Add assessor
    }
    return score_data


def _format_agent_reasons(reasons: AgentReasons) -> str:
    """Format AgentReasons output in a friendly way."""
    if not reasons:
        return ""

    conclusion = getattr(reasons, "conclusion", "")
    prediction = getattr(reasons, "prediction", False)
    confidence = getattr(reasons, "confidence", "medium").capitalize()
    reason_list = getattr(reasons, "reasons", [])

    # Format prediction nicely
    prediction_color = "#28a745" if not prediction else "#dc3545"
    prediction_text = "No" if not prediction else "Yes"
    prediction_html = f'<span style="color:{prediction_color}; font-weight:bold;">{prediction_text}</span>'

    # Format confidence level
    confidence_color = "#28a745" if confidence == "High" else "#ffc107" if confidence == "Medium" else "#dc3545"
    confidence_html = f'<span style="color:{confidence_color}; font-weight:bold;">{confidence}</span>'

    # Format reasons as a list
    reasons_html = ""
    if reason_list:
        reasons_items = [f"<li>{reason}</li>" for reason in reason_list]
        reasons_html = f"""
        <div style="margin-top:10px;">
            <strong>Reasoning:</strong>
            <ul style="margin-top:5px;">
                {"".join(reasons_items)}
            </ul>
        </div>
        """

    return f"""
    <div style="margin-bottom:5px;"><strong>Conclusion:</strong> {conclusion}</div>
    <div style="margin-bottom:5px;"><strong>Violates Policy:</strong> {prediction_html}</div>
    <div style="margin-bottom:5px;"><strong>Confidence:</strong> {confidence_html}</div>
    {reasons_html}
    """


def _extract_score(message) -> tuple[float | None, QualResults | None]:
    """Extract and process score information from a message.
    
    Returns:
        tuple: (score value, score object)

    """
    if not isinstance(message, AgentOutput):
        return None, None

    # Extract QualResults if available
    if hasattr(message, "outputs") and isinstance(message.outputs, QualResults):
        score_value = getattr(message.outputs, "correctness", None)
        return score_value, message.outputs

    # Try standard extraction methods
    if hasattr(message, "outputs"):
        outputs = message.outputs

        # Check if outputs is a dict with 'score'
        if isinstance(outputs, dict) and "score" in outputs:
            return outputs["score"], None

        # Check if outputs is a numeric value
        if isinstance(outputs, (int, float)):
            return float(outputs), None

        # Try to parse JSON string
        if isinstance(outputs, str):
            try:
                data = json.loads(outputs)
                if isinstance(data, dict) and "score" in data:
                    return data["score"], None
            except:
                # If it's not JSON, try to convert the string to a float
                try:
                    return float(outputs), None
                except:
                    pass

    return None, None


def _format_message_for_client(message) -> str | None:
    """Format different message types for web client consumption with improved styling.
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string or None if message shouldn't be sent

    """
    # Get agent info and generate a unique message ID
    agent_info = _get_agent_info(message)
    message_id = agent_info.get("message_id", "")

    # Ignore certain message types (don't display in chat)
    if isinstance(message, (TaskProgressUpdate, TaskProcessingStarted, TaskProcessingComplete, ConductorResponse)):
        return None

    # Handle score messages
    if _is_score_message(message):
        logger.debug(f"Detected score message from: {getattr(message, 'role', 'unknown')}")
        score_value, score_obj = _extract_score(message)
        logger.debug(f"Extracted score: value={score_value}, has_obj={score_obj is not None}")

        # If it's a QualResults object, process it specially
        if score_obj and isinstance(score_obj, QualResults):
            qual_data = _format_qual_results(score_obj)
            if qual_data:
                # agent_id is now correctly included in qual_data
                agent_id = qual_data.get("agent_id", "unknown")  # Should now get the correct ID
                assessor_id = qual_data.get("assessor", "scorer")  # Get assessor from qual_data

                # Store the score data in the traditional way too (for backward compatibility)
                answer_id = qual_data.get("answer_id", "")
                if answer_id and answer_id in scored_messages:
                    original_message_id = scored_messages[answer_id]
                    # Ensure the structure matches the updated ScoreData
                    message_scores[original_message_id] = qual_data

                # Return JavaScript to update scores panel - convert TypedDict to regular dict
                logger.debug(f"Processing QualResults with answer_id={answer_id}, agent_id={agent_id}")
                result = _handle_score_message(
                    score_value=qual_data.get("score", 0.0),
                    assessor_id=str(assessor_id),
                    agent_id=str(agent_id),
                    qual_data=dict(qual_data),  # Convert TypedDict to regular dict
                )
                logger.debug(f"Generated score script for QualResults: length={len(result)}")
                return result

            # If we couldn't process the QualResults but have a score value, use that
            if score_value is not None:
                # Get agent info for agent IDs
                agent_info = _get_agent_info(message)
                agent_id = agent_info.get("id", "unknown")

                # Use a generic assessor ID
                assessor_id = getattr(message, "role", "scorer")

                # Return JavaScript to update scores panel
                return _handle_score_message(score_value, str(assessor_id), str(agent_id))

            # Don't show anything for this message
            return None

        if score_value is not None:
            # Simple score - just update scores panel
            agent_info = _get_agent_info(message)
            agent_id = str(agent_info.get("id", "unknown"))
            assessor_id = str(getattr(message, "role", "scorer"))

            return _handle_score_message(score_value, assessor_id, agent_id)

        # Don't render other scorer messages
        return None

    # Default values for normal (non-score) messages
    content = None

    # Format based on message type
    if isinstance(message, AgentOutput):
        # Store this message ID for future score references
        if hasattr(message, "call_id"):
            answer_id = message.call_id
            if answer_id:
                scored_messages[answer_id] = message_id

        # Special handling for AgentReasons
        if hasattr(message, "outputs") and isinstance(message.outputs, AgentReasons):
            content = _format_agent_reasons(message.outputs)
        elif isinstance(message.outputs, str):
            content = message.outputs
        else:
            # For complex outputs, serialize to JSON
            try:
                content = message.outputs.model_dump_json() if message.outputs else ""
            except:
                content = json.dumps(message.outputs)

    elif isinstance(message, Record):
        content = message.text

    elif isinstance(message, ToolOutput):
        content = message.content
        # Add tool name as a badge
        tool_name = getattr(message, "name", "unknown_tool")
        content = f'<span style="display:inline-block; padding:2px 8px; background:#6c757d; color:white; border-radius:4px; margin-bottom:5px;">{tool_name}</span><br/>{content}'

    elif isinstance(message, ManagerRequest):
        content = message.content

    else:
        # Unhandled message type
        return None

    # Convert markdown to HTML in message content
    if content and isinstance(content, str):
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

    # Apply styling to the message
    if content is None:
        content = ""  # Convert None to empty string to avoid type errors

    styled_content = _format_message_with_style(content, agent_info, message_id)

    return styled_content
