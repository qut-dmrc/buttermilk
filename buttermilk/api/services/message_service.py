import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import uuid

from buttermilk._core import AgentConfig, ManagerRequest, TaskProcessingComplete
from buttermilk._core.agent import AgentOutput, AgentTrace, TaskProcessingStarted
from buttermilk._core.config import RunRequest
from buttermilk._core.constants import CONDUCTOR
from buttermilk._core.contract import FlowEvent, FlowMessage, ManagerMessage
from buttermilk._core.types import Record
from buttermilk.agents.differences import Differences
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.bm import logger


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal["chat_message", "record", "manager_request", "manager_response", "system_message", "user_message", "qual_results", "differences", "judge_reasons"] = Field(..., description="Type of message")
    message_id: str | None = Field(default_factory=lambda: uuid())
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
        if message is None:
            return None
        if isinstance(message, ChatMessage):
            return message
        call_id = message.call_id if hasattr(message, "call_id") else str(uuid())
        message_type = None
        outputs = message.outputs if hasattr(message, "outputs") else message
        preview = message.content if hasattr(message, "content") else None
        try:
            if hasattr(message, "agent_info"):
                agent_info = message.agent_info
            else:
                agent_info = AgentConfig(session_id=message.session_id if hasattr(message, "session_id") else str(uuid()),
                    agent_id=message.agent_id if hasattr(message, "agent_id") else CONDUCTOR,
                    agent_name=message.agent_name if hasattr(message, "agent_name") else CONDUCTOR,
                    role=message.role if hasattr(message, "role") else CONDUCTOR,
                    description="",
                )
            if isinstance(message, Record):
                message_type = "record"
                preview = message.text
                outputs = message
            elif isinstance(message, (AgentTrace, AgentOutput)) and message.outputs:
                if isinstance(message.outputs, JudgeReasons):
                    message_type = "judge_reasons"
                    outputs = message.outputs
                elif isinstance(message.outputs, QualResults):
                    message_type = "qual_results"
                    outputs = message.outputs
                elif isinstance(message.outputs, Differences):
                    message_type = "differences"
                    outputs = message.outputs
                else:
                    message_type = "chat_message"
                    outputs = message.outputs
            elif isinstance(message, ManagerRequest):
                message_type = "manager_request"
            elif isinstance(message, ManagerMessage):
                # Message from UI to chat; don't send back to UI
                return None
            elif isinstance(message, (FlowEvent, TaskProcessingComplete, TaskProcessingStarted)):
                message_type = "system_message"
            else:
                logger.warning(f"Unknown message type: {type(message)}")
                return None

            # Repackage
            output = ChatMessage(message_id=call_id,
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
                case "manager_request":
                    return ManagerRequest(**data)
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
