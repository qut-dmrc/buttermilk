import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import uuid

from buttermilk._core import ManagerRequest
from buttermilk._core.agent import AgentOutput, AgentTrace
from buttermilk._core.constants import CONDUCTOR
from buttermilk._core.contract import FlowEvent, FlowMessage, ManagerResponse
from buttermilk._core.types import Record
from buttermilk.bm import logger


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal["chat_message", "record", "manager_request", "manager_response", "system_message", "user_message"]
    message_id: str | None = Field(default_factory=lambda: uuid())
    preview: str | None = Field(default="", description="Short (one-line) abstract of message")
    outputs: Any | None = Field(None, description="Message outputs")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of the message")
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent ID")


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(message: ChatMessage | Record | FlowEvent | FlowMessage) -> None | ChatMessage:
        """Pass through the message directly to the client
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            dict[str, Any] | None: The serialized message or None if not serializable

        """
        if message is None:
            return None
        if isinstance(message, ChatMessage):
            return message

        message_id = None
        message_type = None
        outputs = message.outputs if hasattr(message, "outputs") else message
        agent_id = message.agent_id if hasattr(message, "agent_id") else CONDUCTOR
        agent_name = message.agent_name if hasattr(message, "agent_name") else CONDUCTOR
        preview = message.content if hasattr(message, "content") else None
        try:
            if isinstance(message, Record):
                message_type = "record"
                preview = message.text
                outputs = message
            elif isinstance(message, (AgentTrace, AgentOutput)):
                message_type = "chat_message"
                message_id = message.call_id
                outputs = message.outputs
            elif isinstance(message, ManagerRequest):
                message_type = "manager_request"
            elif isinstance(message, ManagerResponse):
                message_type = "manager_response"
            elif isinstance(message, FlowEvent):
                message_type = "system_message"
            else:
                logger.warning(f"Unknown message type: {type(message)}")
                return None

            # Repackage
            output = ChatMessage(
                type=message_type,
                preview=preview,
                outputs=outputs,
                agent_id=agent_id, agent_name=agent_name,
                timestamp=datetime.datetime.now(),
            )
            return output

        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")

        return None
