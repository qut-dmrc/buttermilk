import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import ShortUUID

from buttermilk._core.agent import AgentTrace
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.bm import logger


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal["chat_message"]
    message_id: str = Field(default_factory=ShortUUID.uuid)
    content: str = Field(default="", description="Short (one-line) abstract of message")
    outputs: JudgeReasons | QualResults | None = Field(None, description="Message outputs")
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
            if isinstance(message, AgentTrace):
                # Repackage
                output = ChatMessage(message_id=message.call_id,
                    type="chat_message",
                    content=message.content,
                    outputs=message.outputs,
                    timestamp=message.timestamp,
                    agent_id=message.agent_id,
                )
                return output
        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")

        return None
