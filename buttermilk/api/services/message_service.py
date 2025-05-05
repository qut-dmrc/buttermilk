import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import ShortUUID

from buttermilk._core import ManagerRequest
from buttermilk._core.agent import AgentResponse, AgentTrace
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons, Reasons
from buttermilk.bm import logger


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal["chat_message", "record"]
    message_id: str = Field(default_factory=ShortUUID.uuid)
    content: str = Field(default="", description="Short (one-line) abstract of message")
    outputs: JudgeReasons | QualResults | Record | Reasons | None = Field(None, description="Message outputs")
    timestamp: datetime.datetime | None = Field(None, description="Timestamp of the message")
    agent_id: str = Field(..., description="Agent ID")


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(message: Any) -> None | ChatMessage:
        """Pass through the message directly to the client
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            dict[str, Any] | None: The serialized message or None if not serializable

        """
        if message is None:
            return None
        try:
            if isinstance(message, (AgentTrace, AgentResponse)):
                # Repackage
                if isinstance(message.outputs, Record):
                    message_type = "record"
                else:
                    message_type = "chat_message"
                output = ChatMessage(message_id=message.call_id,
                    type=message_type,
                    content=message.content,
                    outputs=message.outputs,
                    timestamp=message.timestamp,
                    agent_id=message.agent_id,
                )
                return output
            if isinstance(message, ManagerRequest):
                return message

        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")

        return None
