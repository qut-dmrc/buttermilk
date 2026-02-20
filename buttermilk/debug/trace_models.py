"""Pydantic models for trace analysis."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    """Summary statistics for a collection of execution traces."""

    total_traces: int = Field(..., description="Total number of traces analyzed")
    error_count: int = Field(..., description="Number of traces with errors")
    agents: list[str] = Field(..., description="List of unique agent names")
    execution_path: list[str] = Field(..., description="Sequence of execution steps")
    time_range: tuple[datetime, datetime] | None = Field(..., description="Start and end time of traces, if available")
    session_id: str | None = Field(..., description="Session identifier, if available")
    by_agent: dict[str, int] = Field(..., description="Trace count by agent name")
    by_error_status: dict[str, int] = Field(..., description="Trace count by error status")


class ErrorContext(BaseModel):
    """Error information with full execution context."""

    call_id: str = Field(..., description="Unique identifier for the failed execution")
    agent_name: str = Field(..., description="Name of the agent that encountered the error")
    agent_role: str = Field(..., description="Role of the agent that encountered the error")
    error_message: str = Field(..., description="Primary error message")
    error_details: dict[str, Any] = Field(..., description="Additional error details")
    timestamp: datetime = Field(..., description="When the error occurred")
    inputs_summary: str = Field(..., description="Summary of inputs that led to the error")
    parent_call_id: str | None = Field(None, description="Parent call ID if this was a nested call")
    preceding_trace_ids: list[str] = Field(default_factory=list, description="Call IDs of traces that preceded this error in the execution path")


class TimelineEvent(BaseModel):
    """Event in the execution timeline."""

    call_id: str = Field(..., description="Unique identifier for this execution")
    timestamp: datetime = Field(..., description="When the event occurred")
    agent_name: str = Field(..., description="Name of the agent")
    agent_role: str = Field(..., description="Role of the agent")
    event_type: Literal["start", "success", "error"] = Field(..., description="Type of event")
    duration_ms: float | None = Field(None, description="Duration in milliseconds if available")
    summary: str = Field(..., description="Brief summary of what happened")


class TraceFile(BaseModel):
    """Metadata about a loaded trace file."""

    path: Path = Field(..., description="Path to the trace file")
    traces: list[Any] = Field(..., description="List of ExecutionTrace objects")
    loaded_at: datetime = Field(default_factory=datetime.now, description="When the file was loaded")


class TraceDiff(BaseModel):
    """Differences between two execution traces."""

    trace1_id: str = Field(..., description="Call ID of the first trace")
    trace2_id: str = Field(..., description="Call ID of the second trace")
    differences: dict[str, Any] = Field(..., description="Field-level differences between traces")
    only_in_trace1: list[str] = Field(..., description="Fields present only in trace1")
    only_in_trace2: list[str] = Field(..., description="Fields present only in trace2")
