"""Defines the WebUIAgent for interacting with users via FastAPI websockets.

This agent serves as a bridge between the FastAPI application and the autogen
group chat workflow, routing messages in both directions.
"""

import asyncio
from typing import Any

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from fastapi import WebSocketDisconnect
from pydantic import BaseModel

from buttermilk import logger
from buttermilk._core.contract import (
    MANAGER,
    AllMessages,
    ErrorEvent,
    GroupchatMessageTypes,
    OOBMessages,
)
from buttermilk.web.messages import _format_message_for_client


class WebUIAgent(RoutedAgent):
    """Represents a web-based user interface connected via mutual callbacks.
    
    This agent serves as a bridge between FastAPI or Shiny and autogen's group chat.
    It handles message routing in both directions.
    """

    def __init__(self, callback_to_ui: Any, session_id: str, description: str = "Web UI Agent for handling user interactions", **kwargs):
        """Initialize the WebUIAgent.
        
        Args:
            description: Description of this agent's role

        """
        super().__init__(description=description)
        self._topic_id = DefaultTopicId(type=MANAGER)
        logger.debug(f"WebUIAgent initialized with topic ID: {self._topic_id}")
        self.callback_to_ui = callback_to_ui

        self.session_id = session_id
        self.task = asyncio.create_task(self.comms())

    async def send_to_ui(self, message: BaseModel | dict):
        """Send a message to the UI via the websocket.

        Args:
            message: The message to send

        """
        message = _format_message_for_client(message)
        if message:
            await self.callback_to_ui(message)

    async def comms(self):
        # Monitor a websocket connection
        logger.info(f"WebSocket connection received by agent {self.id} for session {self.session_id}")

        # Optionally send a welcome message
        welcome_msg = {"type": "system", "content": "Connected to WebUIAgent", "session_id": self.session_id}

        await self.send_to_ui(welcome_msg)

        # Main communication loop
        while True:
            try:
                await asyncio.sleep(1)
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from flow for session {self.session_id}")
                break
            except Exception as e:
                logger.error(f"Error in web ui: {e}")
                await self.send_to_ui(ErrorEvent(source="web runner", content=f"Error: {e!s}"))

    # ===== Autogen Message Handlers =====
    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None:
        """Handle messages from the group chat and forward to web clients.
        
        Args:
            message: The group chat message
            ctx: Message context

        """
        logger.debug(f"WebUIAgent received group chat message: {type(message).__name__}")

        await self.send_to_ui(message)

    @message_handler
    async def handle_control_message(
        self,
        message: AllMessages,  # Handles out-of-band control messages.
        ctx: MessageContext,
    ) -> OOBMessages | None:
        logger.debug(f"WebUIAgent received event message: {type(message).__name__}")
        await self.callback_to_ui(message)
        return None

    async def cleanup(self):
        """Clean up resources when the agent is no longer needed."""
        if hasattr(self, "task") and self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
