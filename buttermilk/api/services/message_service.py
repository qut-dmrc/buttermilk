import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from shortuuid import uuid

from buttermilk import (
    AgentConfig,
    StepRequest,
    TaskProcessingComplete,
    logger,
)
from buttermilk._core.config import RunRequest
from buttermilk._core.contract import (
    AgentOutput,
    ConductorRequest,
    ErrorEvent,
    ExecutionTrace,
    FlowEvent,
    FlowMessage,
    FlowProgressUpdate,
    SystemPromptMessage,
    TaskProcessingStarted,
    UserResponseMessage,
)
from buttermilk._core.types import AssistantMessage, Record
from buttermilk.agents.differences import Differences
from buttermilk.agents.judge import JudgeReasons
from buttermilk.agents.rag import ResearchResult
from buttermilk.utils.pricing import calculate_token_cost, extract_usage_from_metadata

PREVIEW_LENGTH = 200


class ChatMessage(BaseModel):
    """Chat message model"""

    type: Literal[
        "chat_message",
        "record",
        "system_prompt",
        "user_response",
        "system_message",
        "system_update",
        "system_error",
        "assessments",
        "research_result",
        "differences",
        "judge_reasons",
        "start_flow",
    ] = Field(..., description="Type of message")
    message_id: str = Field(default_factory=lambda: uuid())
    preview: str | None = Field(
        default="", description="Short (one-line) abstract of message"
    )
    outputs: Any | None = Field(None, description="Message outputs")
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now, description="Timestamp of the message"
    )
    agent_info: AgentConfig | None = Field(None, description="Agent information")
    tracing_link: str | None = Field(
        None, description="Link to the tracing information"
    )
    prompt_tokens: int = Field(
        default=0, description="Number of prompt/input tokens used"
    )
    completion_tokens: int = Field(
        default=0, description="Number of completion/output tokens used"
    )
    cost_usd: float = Field(
        default=0.0, description="Estimated cost in USD for this message"
    )


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(
        message: ExecutionTrace | ChatMessage | Record | FlowEvent | FlowMessage,
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

            # Handle messages that need special processing
            if isinstance(message, (ChatMessage, StepRequest)):
                message_type = type(message).__name__
                action = (
                    "returning as-is"
                    if isinstance(message, ChatMessage)
                    else "not sending to UI"
                )
                logger.debug(f"[MessageService] {message_type} received, {action}")
                return message if isinstance(message, ChatMessage) else None

            # Convert UserResponseMessage to user_response for display
            if isinstance(message, UserResponseMessage):
                logger.debug(
                    "UserResponseMessage received, converting to user_response for UI"
                )
                return ChatMessage(
                    type="user_response",
                    preview=str(message.content)[:PREVIEW_LENGTH]
                    if message.content
                    else "",
                    outputs=message.content,
                    agent_info=None,
                    timestamp=datetime.datetime.now(),
                    message_id=message.message_id,  # Preserve the message_id
                )

            agent_info = getattr(message, "agent_info", None)
            message_id = getattr(message, "call_id", uuid())
            preview = getattr(message, "preview", None)
            tracing_link = getattr(message, "tracing_link", None)

            # Initialize token/cost tracking variables
            prompt_tokens = 0
            completion_tokens = 0
            cost_usd = 0.0

            if isinstance(message, ExecutionTrace) or isinstance(message, AgentOutput):
                # Extract tokens/cost from metadata if available
                usage = None
                model_for_pricing = None

                if hasattr(message, "metadata") and message.metadata:
                    # Determine model name for pricing
                    model_for_pricing = message.metadata.get("agent_model")

                    # Try to extract usage dict from heterogeneous metadata structures
                    try:
                        usage = extract_usage_from_metadata(message.metadata)
                    except Exception as e:
                        logger.debug(
                            f"[MessageService] Failed to extract usage from metadata: {e}"
                        )

                # Fall back to agent_info parameters for model name
                if not model_for_pricing and agent_info is not None:
                    try:
                        params = getattr(agent_info, "parameters", {}) or {}
                        model_for_pricing = params.get("model")
                    except Exception:
                        model_for_pricing = None

                # Calculate cost/tokens if we have model and usage information
                if model_for_pricing and usage:
                    try:
                        prompt_tokens, completion_tokens, cost_usd = (
                            calculate_token_cost(
                                model=model_for_pricing,
                                usage_dict=usage,
                            )
                        )
                        logger.debug(
                            f"[MessageService] Pricing computed: {prompt_tokens} prompt, {completion_tokens} completion, ${cost_usd:.6f}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"[MessageService] Failed to calculate token cost: {e}"
                        )

                if message.outputs:
                    # Send the unwrapped message instead of the ExecutionTrace object
                    message = message.outputs
                elif message.error:
                    # Handle error - convert to ErrorEvent if it's a list
                    if isinstance(message.error, list) and message.error:
                        message = (
                            message.error[0]
                            if isinstance(message.error[0], ErrorEvent)
                            else ErrorEvent(
                                source=agent_info.get("name", "unknown")
                                if agent_info
                                else "unknown",
                                content=str(message.error[0]),
                            )
                        )
                    else:
                        message = ErrorEvent(
                            source=agent_info.get("name", "unknown")
                            if agent_info
                            else "unknown",
                            content=str(message.error),
                        )
                else:
                    logger.warning(
                        f"[MessageService] ExecutionTrace object with no outputs: {message}, returning None."
                    )
                    return None

            message_type = None
            if isinstance(message, Record):
                message_type = "record"
            elif isinstance(message, ConductorRequest):
                return None
            elif isinstance(message, JudgeReasons):
                message_type = "judge_reasons"
            # Note: QualResults removed - scorer moved to tja project
            elif isinstance(message, Differences):
                message_type = "differences"
            elif isinstance(message, ResearchResult):
                message_type = "research_result"
            elif isinstance(message, SystemPromptMessage):
                message_type = "system_prompt"
            elif isinstance(message, AssistantMessage):
                message_type = "chat_message"
                preview = str(message.content)[:PREVIEW_LENGTH]
            elif isinstance(message, FlowProgressUpdate):
                message_type = "system_update"
            elif isinstance(message, ErrorEvent):
                message_type = "system_error"
            elif isinstance(message, FlowEvent):
                message_type = "system_update"
            elif isinstance(message, TaskProcessingComplete) or isinstance(
                message, TaskProcessingStarted
            ):
                message_type = "system_update"
            elif isinstance(message, str):
                # Handle string messages (like StructuredLLMHost summaries) as chat messages
                message_type = "chat_message"
                preview = message[:PREVIEW_LENGTH]
            else:
                return None

            # Repackage
            output = ChatMessage(
                message_id=message_id,
                type=message_type,
                preview=preview,
                outputs=message,
                agent_info=agent_info,
                tracing_link=tracing_link,
                timestamp=datetime.datetime.now(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            )
            return output

        except Exception as e:
            logger.error(f"[MessageService] Error formatting message for client: {e}")

        return None

    @staticmethod
    async def process_message_from_ui(
        data: dict[str, Any],
    ) -> (
        FlowEvent
        | RunRequest
        | FlowMessage
        | TaskProcessingStarted
        | TaskProcessingComplete
        | None
    ):
        """Process a message from a WebSocket connection.

        Args:
            session_id: The session ID
            message: The message to process

        Returns:
            FlowEvent | FlowMessage | None: The processed message or None if not handled

        """
        try:
            message_type = data.pop("type", None)

            # If no type but has flow field, it's likely a RunRequest
            if not message_type and "flow" in data and "prompt" in data:
                logger.debug("Detected RunRequest format without type field")
                return RunRequest(**data)

            match message_type:
                case "run_flow":
                    parameters = data.get("parameters", {})
                    if "criteria" in data:
                        parameters["criteria"] = data.pop("criteria")

                    run_request = RunRequest(
                        flow=data.pop("flow"),
                        parameters=parameters,
                        inputs=data,
                    )
                    return run_request
                case "pull_task":
                    from buttermilk.api.job_queue import JobQueueClient

                    task, ack_id = await JobQueueClient().pull_single_task()
                    return task
                case "pull_tox":
                    from buttermilk.api.job_queue import JobQueueClient

                    return await JobQueueClient().pull_tox_example()
                case "system_prompt":
                    return SystemPromptMessage(**data)
                case "user_response":
                    # Remove 'type' field as UserResponseMessage doesn't expect it
                    message_data = {k: v for k, v in data.items() if k != "type"}
                    return UserResponseMessage(**message_data)
                case "TaskProcessingComplete":
                    return TaskProcessingComplete(**data)
                case "TaskProcessingStarted":
                    return TaskProcessingStarted(**data)
                case _:
                    logger.warning(
                        f"Unknown message type received on websocket: {message_type}"
                    )
                    return None
            return None
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
