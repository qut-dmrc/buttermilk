import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import uuid

from buttermilk import logger
from buttermilk._core import (
    AgentConfig,
    StepRequest,
    TaskProcessingComplete,
    UIMessage,
)
from buttermilk._core.config import RunRequest
from buttermilk._core.contract import (
    AgentTrace,
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    FlowProgressUpdate,
    ManagerMessage,
    TaskProcessingStarted,
)
from buttermilk._core.types import AssistantMessage, Record
from buttermilk.agents.differences import Differences
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.agents.rag import ResearchResult

PREVIEW_LENGTH = 200


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal[
        "chat_message",
        "record",
        "ui_message",
        "manager_response",
        "system_message",
        "system_update",
        "system_error",
        "user_message",
        "assessments",
        "research_result",
        "differences",
        "judge_reasons",
        "system_message",  # Added system_message
    ] = Field(..., description="Type of message")
    message_id: str = Field(default_factory=lambda: uuid())
    preview: str | None = Field(default="", description="Short (one-line) abstract of message")
    outputs: Any | None = Field(None, description="Message outputs")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of the message")
    agent_info: AgentConfig | None = Field(None, description="Agent information")
    tracing_link: str | None = Field(None, description="Link to the tracing information")


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(
        message: AgentTrace | ChatMessage | Record | FlowEvent | FlowMessage,
    ) -> None | ChatMessage:
        """Format and pass the message to the client

        Args:
            message: The message to format (Pydantic object)

        Returns:
            dict[str, Any] | None: The serialized message or None if not serializable

        """
        try:
            if message is None:
                return None

            # Skip messages that shouldn't be sent to UI
            if isinstance(message, (ChatMessage, ManagerMessage, StepRequest)):
                message_type = type(message).__name__
                action = "returning as-is" if isinstance(message, ChatMessage) else "not sending to UI"
                logger.debug(f"[MessageService] {message_type} received, {action}")
                return message if isinstance(message, ChatMessage) else None

            agent_info = getattr(message, "agent_info", None)
            message_id = getattr(message, "call_id", uuid())
            preview = getattr(message, "preview", None)
            tracing_link = getattr(message, "tracing_link", None)

            if isinstance(message, AgentTrace):
                if message.outputs:
                    # Send the unwrapped message instead of the AgentTrace object
                    message = message.outputs
                else:
                    logger.warning(f"[MessageService] AgentTrace object with no outputs: {message}, returning None.")
                    return None

            message_type = None
            if isinstance(message, Record):
                message_type = "record"
            elif isinstance(message, JudgeReasons):
                message_type = "judge_reasons"
            elif isinstance(message, QualResults):
                message_type = "assessments"
            elif isinstance(message, Differences):
                message_type = "differences"
            elif isinstance(message, ResearchResult):
                message_type = "research_result"
            elif isinstance(message, UIMessage):
                message_type = "ui_message"
            elif isinstance(message, AssistantMessage):
                message_type = "chat_message"
                preview = str(message.content)[:PREVIEW_LENGTH]
            elif isinstance(message, FlowProgressUpdate):
                message_type = "system_update"
            elif isinstance(message, ErrorEvent):
                message_type = "system_error"
            elif isinstance(message, FlowEvent):
                message_type = "system_message"
            elif isinstance(message, TaskProcessingComplete):
                message_type = "system_update"
            elif isinstance(message, TaskProcessingStarted):
                message_type = "system_update"
            else:
                logger.warning(f"[MessageService] Unknown message type: {type(message)}, defaulting to chat_message.")
                message_type = "chat_message"

            # Repackage
            output = ChatMessage(
                message_id=message_id,
                type=message_type,
                preview=preview,
                outputs=message,
                agent_info=agent_info,
                tracing_link=tracing_link,
                timestamp=datetime.datetime.now(),
            )
            logger.debug(f"[MessageService] Formatted message of type {output.type}, returning output. Content: {output.outputs}")
            return output

        except Exception as e:
            logger.error(f"[MessageService] Error formatting message for client: {e}")

        return None

    @staticmethod
    async def process_message_from_ui(
        data: dict[str, Any],
    ) -> FlowEvent | RunRequest | FlowMessage | TaskProcessingStarted | TaskProcessingComplete | None:
        """Process a message from a WebSocket connection.

        Args:
            session_id: The session ID
            message: The message to process

        Returns:
            FlowEvent | FlowMessage | None: The processed message or None if not handled

        """
        try:
            message_type = data.get("type", None)
            
            # If no type but has flow field, it's likely a RunRequest
            if not message_type and "flow" in data and "prompt" in data:
                logger.debug("Detected RunRequest format without type field")
                return RunRequest(**data)

            match message_type:
                case "run_flow":
                    run_request = RunRequest(
                        ui_type="web",
                        flow=data.get("flow"),
                        record_id=data.get("record_id", None),
                        parameters={k: v for k, v in data.items() if k not in ["type", "flow", "record_id"]},
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
                    # Remove 'type' field as ManagerMessage doesn't expect it
                    message_data = {k: v for k, v in data.items() if k != "type"}
                    return ManagerMessage(**message_data)
                case "TaskProcessingComplete":
                    return TaskProcessingComplete(**data)
                case "TaskProcessingStarted":
                    return TaskProcessingStarted(**data)
                case _:
                    logger.warning(f"Unknown message type received on websocket: {message_type}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
