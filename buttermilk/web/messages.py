

import json

from buttermilk._core.agent import AgentOutput, ManagerRequest, ToolOutput
from buttermilk._core.contract import (
    ConductorResponse,
    ManagerMessage,
    ManagerResponse,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
)
from buttermilk._core.types import Record


def _format_message_for_client(message) -> str | None:
    """Format different message types for web client consumption.
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string or None if message shouldn't be sent

    """
    # Default values
    msg_type = "message"
    content = None

    # Ignore certain message types (don't display in chat)
    if isinstance(message, (StepRequest, TaskProgressUpdate, TaskProcessingStarted, TaskProcessingComplete, ConductorResponse, ManagerResponse, ManagerMessage)):
        return None
    # Format based on message type
    if isinstance(message, AgentOutput):
        msg_type = "agent_output"
        if isinstance(message.outputs, str):
            content = message.outputs
        else:
            # For complex outputs, serialize to JSON
            try:
                content = json.dumps(message.outputs) if message.outputs else ""
            except:
                content = str(message.outputs)

    elif isinstance(message, Record):  # Add handling for Record
        content = message.text

    elif isinstance(message, ToolOutput):
        msg_type = "tool_output"
        content = message.content
        # Additional info
        tool_name = getattr(message, "name", "unknown_tool")
        source = f"tool/{tool_name}"

    elif isinstance(message, ManagerRequest):
        msg_type = "instructions"
        content = message.prompt or ""

    else:
        # Unhandled message type
        return None
    return content
