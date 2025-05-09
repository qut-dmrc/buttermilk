import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import uuid

from buttermilk._core import AgentConfig, ManagerRequest
from buttermilk._core.agent import AgentOutput, AgentTrace
from buttermilk._core.constants import CONDUCTOR
from buttermilk._core.contract import FlowEvent, FlowMessage, ManagerMessage
from buttermilk._core.types import Record
from buttermilk.bm import logger


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal["chat_message", "record", "manager_request", "manager_response", "system_message", "user_message"]
    message_id: str | None = Field(default_factory=lambda: uuid())
    preview: str | None = Field(default="", description="Short (one-line) abstract of message")
    outputs: Any | None = Field(None, description="Message outputs")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of the message")
    agent_info: AgentConfig | None = Field(None, description="Agent information")


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(message: AgentTrace | ChatMessage | Record | FlowEvent | FlowMessage) -> None | ChatMessage:
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

        message_type = None
        outputs = message.outputs if hasattr(message, "outputs") else message
        preview = message.content if hasattr(message, "content") else None
        try:
            if hasattr(message, "agent_info"):
                agent_info = message.agent_info
            else:
                agent_info = AgentConfig(session_id=message.session_id if hasattr(message, "session_id") else str(uuid()),
                    agent_id=message.agent_id if hasattr(message, "agent_id") else CONDUCTOR,
                    name=message.agent_name if hasattr(message, "agent_name") else CONDUCTOR,
                    role=message.role if hasattr(message, "role") else CONDUCTOR,
                    description="",
                )
            if isinstance(message, Record):
                message_type = "record"
                preview = message.text
                outputs = message
            elif isinstance(message, (AgentTrace, AgentOutput)):
                message_type = "chat_message"
                outputs = message.outputs
            elif isinstance(message, ManagerRequest):
                message_type = "manager_request"
            elif isinstance(message, ManagerMessage):
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
                agent_info=agent_info,
                timestamp=datetime.datetime.now(),
            )
            return output

        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")

        return None
