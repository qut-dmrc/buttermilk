

import json
from typing import Any, Dict, Optional

from buttermilk._core.agent import AgentOutput, ManagerRequest, ToolOutput
from buttermilk._core.contract import ConductorResponse, ManagerResponse, TaskProcessingComplete
from buttermilk._core.types import Record


def _format_message_for_client( message) -> str|None:
    """
    Format different message types for web client consumption.
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string or None if message shouldn't be sent
    """
    # Default values
    msg_type = "message"
    content = None
    
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
    elif isinstance(message, ConductorResponse):
        msg_type = "conductor_response"
        content = getattr(message, "content", str(message))
    elif isinstance(message, ManagerResponse):
        content = message.model_dump_json()
    elif isinstance(message, TaskProcessingComplete):
        msg_type = "status"
        content = "Task completed" if not message.is_error else "Task failed"
        
    elif isinstance(message, ToolOutput):
        msg_type = "tool_output"
        content = message.content
        # Additional info
        tool_name = getattr(message, "name", "unknown_tool")
        source = f"tool/{tool_name}"
        
    elif isinstance(message, ManagerRequest):
        msg_type = "instructions"
        content = message.prompt if message.prompt else ""
        
    elif hasattr(message, "content") and message.content:
        # Generic message with content
        content = message.content
        
    elif hasattr(message, "prompt") and message.prompt:
        # Message with prompt
        content = message.prompt
        
    else:
        # Unhandled message type
        return str(message)
        
    