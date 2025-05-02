from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FlowStatus(str, Enum):
    """Enumeration for flow status values"""

    WAITING = "waiting"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class AgentInfo(BaseModel):
    """Agent information model"""

    id: str = Field(..., description="Unique identifier for the agent")
    name: str | None = Field(None, description="Name of the agent")
    role: str | None = Field(None, description="Role of the agent")


class Assessment(BaseModel):
    """Assessment model for score details"""

    correct: bool = Field(..., description="Whether the assessment is correct")
    feedback: str = Field(..., description="Feedback for the assessment")


class ScoreData(BaseModel):
    """Score data model"""

    score_text: str = Field(..., description="Text representation of the score")
    color: str = Field(..., description="Color representation of the score")
    assessments: list[Assessment] | None = Field(None, description="List of assessments")


class PredictionData(BaseModel):
    """Prediction data model"""

    agent_id: str = Field(..., description="ID of the agent making the prediction")
    agent_name: str = Field(..., description="Name of the agent making the prediction")
    violates: bool = Field(..., description="Whether the content violates criteria")
    confidence: str = Field(..., description="Confidence level of the prediction")
    conclusion: str = Field(..., description="Conclusion of the prediction")
    reasons: list[str] | None = Field(None, description="Reasons for the prediction")


class ProgressData(BaseModel):
    """Progress data model"""

    role: str | None = Field(None, description="Role associated with the progress")
    step_name: str | None = Field(None, description="Name of the current step")
    status: str = Field(..., description="Status of the progress")
    message: str | None = Field(None, description="Progress message")
    total_steps: int = Field(..., description="Total number of steps")
    current_step: int = Field(..., description="Current step number")
    pending_agents: list[str] | None = Field(None, description="List of pending agents")

    class Config:
        """Configuration for the model"""

        arbitrary_types_allowed = True


class ChatMessage(BaseModel):
    """Chat message model"""

    content: str = Field(..., description="Content of the message")
    timestamp: str | None = Field(None, description="Timestamp of the message")
    agent_info: AgentInfo | None = Field(None, description="Information about the agent")


class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages"""

    type: str = Field(..., description="Type of the message")


class ChatMessageEvent(WebSocketMessage):
    """Chat message event model"""

    type: str = "chat_message"
    content: str = Field(..., description="Content of the message")
    agent_info: AgentInfo | None = Field(None, description="Information about the agent")


class ScoreUpdateEvent(WebSocketMessage):
    """Score update event model"""

    type: str = "score_update"
    agent_id: str = Field(..., description="ID of the agent being scored")
    assessor_id: str = Field(..., description="ID of the assessor")
    score_data: ScoreData = Field(..., description="Score data")


class ProgressUpdateEvent(WebSocketMessage):
    """Progress update event model"""

    type: str = "progress_update"
    progress: ProgressData = Field(..., description="Progress data")


class FlowStateChangeEvent(WebSocketMessage):
    """Flow state change event model"""

    type: str = "flow_state_change"
    state: str = Field(..., description="New state of the flow")


class RequiresConfirmationEvent(WebSocketMessage):
    """Requires confirmation event model"""

    type: str = "requires_confirmation"


class FlowStartedEvent(WebSocketMessage):
    """Flow started event model"""

    type: str = "flow_started"
    flow: str = Field(..., description="Name of the flow that was started")
    record_id: str = Field(..., description="ID of the record being processed")


class ErrorEvent(WebSocketMessage):
    """Error event model"""

    type: str = "error"
    message: str = Field(..., description="Error message")


class RunFlowRequest(BaseModel):
    """Run flow request model"""

    type: str = "run_flow"
    flow: str = Field(..., description="Name of the flow to run")
    record_id: str = Field(..., description="ID of the record to process")
    criteria: str | None = Field(None, description="Criteria to use for evaluation")


class UserInputRequest(BaseModel):
    """User input request model"""

    type: str = "user_input"
    message: str = Field(..., description="Message input by the user")


class ConfirmRequest(BaseModel):
    """Confirm request model"""

    type: str = "confirm"


class SessionData(BaseModel):
    """Session data model"""

    messages: list[dict[str, Any]] = Field(default_factory=list, description="List of messages in the session")
    progress: dict[str, Any] = Field(
        default_factory=lambda: {
            "current_step": 0,
            "total_steps": 100,
            "status": "waiting",
        },
        description="Progress information for the session",
    )
    callback: Any | None = Field(None, description="Callback function for the session")
    outcomes_version: str | None = Field(None, description="Version of the outcomes")
