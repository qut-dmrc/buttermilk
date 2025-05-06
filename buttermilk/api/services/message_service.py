import datetime
from typing import Literal

from pydantic import BaseModel, Field
from shortuuid import ShortUUID

from buttermilk._core import ManagerRequest
from buttermilk._core.agent import AgentResponse, AgentTrace
from buttermilk._core.contract import FlowEvent, FlowMessage, ManagerResponse
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
    def format_message_for_client(message: Record | FlowEvent | FlowMessage) -> None | ChatMessage:
        """Pass through the message directly to the client
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            dict[str, Any] | None: The serialized message or None if not serializable

        """
        message_id = None
        message_type = None
        outputs = None
        agent_id = None
        content = None
        if message is None:
            return None
        try:

            agent_id = message.agent_id if hasattr(message, "agent_id") else None
            content = message.content if hasattr(message, "content") else None

            if isinstance(message, Record):
                message_type = "record"
                outputs = message
            elif isinstance(message, (AgentTrace, AgentResponse)):
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
                message_id=message_id,
                type=message_type,
                content=content,
                outputs=outputs,
                agent_id=agent_id,
                timestamp=datetime.datetime.now(),
            )
            return output

        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")

        return None
