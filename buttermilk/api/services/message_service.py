import datetime
from typing import Any, Literal

from pydantic import UUID4, BaseModel, Field
from shortuuid import uuid

from buttermilk._core import (
    AgentConfig,
    TaskProcessingComplete,
    UIMessage,
    logger,
)
from buttermilk._core.agent import AgentTrace, TaskProcessingStarted
from buttermilk._core.config import RunRequest
from buttermilk._core.contract import FlowEvent, FlowMessage, ManagerMessage
from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk.agents.differences import Differences
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons

PREVIEW_LENGTH = 200


class ChatMessage(BaseModel):
    """Chat message model"""
    
    type: Literal["chat_message", "record", "ui_message", "manager_response", "system_message", "user_message", "qual_results", "differences", "judge_reasons"] = Field(..., description="Type of message")
    message_id: str = Field(default_factory=lambda: uuid())
    preview: str | None = Field(default="", description="Short (one-line) abstract of message")
    outputs: Any | None = Field(None, description="Message outputs")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of the message")
    agent_info: AgentConfig | None = Field(None, description="Agent information")


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(message: AgentTrace | ChatMessage | Record | FlowEvent | FlowMessage) -> None | ChatMessage:
        """Format and pass the message to the client
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            dict[str, Any] | None: The serialized message or None if not serializable

        """
        try:
            if message is None:
                return None
            if isinstance(message, AgentTrace):
                if message.outputs:
                    # Send the unwrapped message instead of the AgentTrace object
                    agent_info = message.agent_info
                    message = message.outputs
                else:
                    logger.warning(f"AgentTrace object with no outputs: {message}")
                    return None

            if isinstance(message, ChatMessage):
                # Already formatted
                return message
            if isinstance(message, ManagerMessage):
                # Message from UI to chat; don't send back to UI
                return None

            message_type = None
            if isinstance(message, Record):
                message_type = "record"
            elif isinstance(message, JudgeReasons):
                message_type = "judge_reasons"
            elif isinstance(message, QualResults):
                message_type = "qual_results"
            elif isinstance(message, Differences):
                message_type = "differences"
            elif isinstance(message, UIMessage):
                message_type = "ui_message"
            elif isinstance(message, (FlowEvent, TaskProcessingComplete, TaskProcessingStarted)):
                return None
            else:
                logger.warning(f"Unknown message type: {type(message)}")
                return None

            message_id = getattr(message, "call_id", uuid())
            preview = getattr(message, "preview", None)
            agent_info = getattr(message, "agent_info", None)
            # Repackage
            output = ChatMessage(message_id=message_id,
                type=message_type,
                preview=preview,
                outputs=message,
                agent_info=agent_info,
                timestamp=datetime.datetime.now(),
            )
            return output

        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")

        return None

    @staticmethod
    async def process_message_from_ui(data: dict[str, Any]) -> FlowEvent | RunRequest | FlowMessage | TaskProcessingStarted | TaskProcessingComplete | None:
        """Process a message from a WebSocket connection.
        
        Args:
            session_id: The session ID
            message: The message to process

        Returns:
            FlowEvent | FlowMessage | None: The processed message or None if not handled

        """
        try:
            message_type = data.pop("type", None)

            match message_type:
                case "run_flow":
                    run_request = RunRequest(ui_type="web",
                        flow=data.pop("flow"),
                        record_id=data.pop("record_id", None),
                        parameters=data,
                    )
                    return run_request
                case "pull_task":
                    from buttermilk.api.job_queue import JobQueueClient
                    return await JobQueueClient().pull_single_task()
                case "pull_tox":
                    from buttermilk.api.job_queue import JobQueueClient
                    return await JobQueueClient().pull_tox_example()
                case "ui_message":
                    return UIMessage(**data)
                case "manager_response":
                    return ManagerMessage(**data)
                case "TaskProcessingComplete":
                    return TaskProcessingComplete(**data)
                case "TaskProcessingStarted":
                    return TaskProcessingStarted(**data)
            return None
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
