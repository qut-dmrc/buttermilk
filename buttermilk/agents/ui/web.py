"""
Defines the WebUIAgent for interacting with users via FastAPI websockets.

This agent serves as a bridge between the FastAPI application and the autogen
group chat workflow, routing messages in both directions.
"""

import asyncio
import datetime
import json
from collections.abc import Awaitable
from typing import Any, Callable, Dict, Optional, Union

from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk._core.contract import (
    MANAGER,
    AgentInput,
    AgentOutput,
    ConductorResponse,
    ErrorEvent,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerRequest,
    ManagerResponse,
    OOBMessages,
    TaskProcessingComplete,
    ToolOutput,
)
from buttermilk._core.types import Record


class WebUIAgent(RoutedAgent):
    """
    Represents a web-based user interface connected via mutual callbacks.
    
    This agent serves as a bridge between FastAPI or Shiny and autogen's group chat.
    It handles message routing in both directions.
    """
    
    def __init__(self, client_callback: Any, session_id: str, description: str = "Web UI Agent for handling user interactions", **kwargs):
        """
        Initialize the WebUIAgent.
        
        Args:
            description: Description of this agent's role
        """
        super().__init__(description=description)
        self._topic_id = DefaultTopicId(type=MANAGER)
        logger.debug(f"WebUIAgent initialized with topic ID: {self._topic_id}")
        self.callback_to_ui = client_callback
        self.session_id = session_id
        self.task = asyncio.create_task(self.comms())

    async def comms(self):
        # Register a new WebSocket connection from a client.
        await self.callback_to_ui.accept()
        logger.info(f"WebSocket connection registered for session {self.session_id}")
        
        # Optionally send a welcome message
        welcome_msg = {"type": "system", "content": "Connected to WebUIAgent", "session_id": self.session_id}
        await self.callback_to_ui(welcome_msg)

        # Main communication loop
        while True:
            try:
                # Wait for client message
                client_message = await self.callback_to_ui.receive_json()
                await self.callback_to_groupchat(client_message)
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from flow for session {self.session_id}")
                break
            except Exception as e:
                logger.error(f"Error in flow runner: {e}")
                await self.callback_to_ui.send_json({
                    "type": "error",
                    "content": f"Error: {str(e)}",
                    "source": "system"
                })

    
    # ===== Autogen Message Handlers =====    
    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None:
        """
        Handle messages from the group chat and forward to web clients.
        
        Args:
            message: The group chat message
            ctx: Message context
        """
        logger.debug(f"WebUIAgent received group chat message: {type(message).__name__}")
        # Format different message types for the web client
        client_msg = self._format_message_for_client(message, ctx)
        
        if client_msg:
            # Broadcast to all connected clients
            await self.callback_to_ui(client_msg['content'])
    
    def _format_message_for_client(self, message: Any, ctx: MessageContext) -> Optional[Dict[str, Any]]:
        """
        Format different message types for web client consumption.
        
        Args:
            message: The message to format
            ctx: Message context
            
        Returns:
            Formatted message dict or None if message shouldn't be sent
        """
        # Default values
        msg_type = "message"
        content = None
        source = str(ctx.sender) if ctx.sender else "system"
        
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
                    
            # Try to extract agent_id if available
            source = getattr(message, "agent_id", source)
        elif isinstance(message, Record):  # Add handling for Record
            content = message.text
            source = getattr(message, "agent_id", source)
        elif isinstance(message, ConductorResponse):
            msg_type = "conductor_response"
            content = getattr(message, "content", str(message))
            
        elif isinstance(message, TaskProcessingComplete):
            msg_type = "status"
            content = "Task completed" if not message.is_error else "Task failed"
            source = getattr(message, "agent_id", source)
            
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
            logger.debug(f"Skipping unformatted message type: {type(message).__name__}")
            return None
            
        # Create the message dict
        result = {
            "type": msg_type,
            "content": content,
            "source": source,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        # Add metadata for message types that have it
        if isinstance(message, FlowMessage) and message.metadata:
            result["metadata"] = message.metadata
            
        return result
    
    @message_handler
    async def handle_control_message(
        self,
        message: OOBMessages,  # Handles out-of-band control messages.
        ctx: MessageContext,
    ) -> OOBMessages| None:
        logger.debug(f"WebUIAgent received event message: {type(message).__name__}")
        await self.callback_to_ui(message)
        return None

    async def cleanup(self):
        """Clean up resources when the agent is no longer needed."""
        if hasattr(self, 'task') and self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass