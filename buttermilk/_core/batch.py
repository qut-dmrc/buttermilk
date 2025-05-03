"""Defines batch processing models and utilities for Buttermilk.

This module contains the data models and core components for batch job configuration,
submission, and tracking in the Buttermilk system.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

import pydantic
import shortuuid
from pydantic import BaseModel, ConfigDict, Field


class BatchJobStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchErrorPolicy(str, Enum):
    """Error handling policy for batch jobs."""

    CONTINUE = "continue"  # Continue processing other jobs if one fails
    STOP = "stop"  # Stop the entire batch if any job fails


class BatchRequest(BaseModel):
    """Request to create a new batch job."""

    command: Literal["create", "list", "status"] = Field(default="create", description="Command to execute")
    flow: str = Field(..., description="The name of the flow to execute for all records in the batch")
    shuffle: bool = Field(default=True, description="Whether to shuffle record IDs before processing")
    max_records: int | None = Field(default=None, description="Maximum number of records to process (None for all)")
    concurrency: int = Field(default=1, description="Number of jobs to process concurrently")
    error_policy: BatchErrorPolicy | Literal["stop", "continue"] = Field(default=BatchErrorPolicy.CONTINUE, description="How to handle job failures")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional parameters to pass to each flow run")
    interactive: bool = Field(default=False, description="Whether this batch requires interactive input")
    wait: bool = Field(default=True, description="Whether to wait for batch completion")
    model_config = ConfigDict(
        extra="forbid",  # Disallow extra fields for strict input validation
    )

    @pydantic.field_validator("error_policy")
    @classmethod
    def _select_policy(cls, value) -> BatchErrorPolicy:
        if isinstance(value, str) and value.lower() == "stop":
            return BatchErrorPolicy.STOP
        return value


class BatchMetadata(BaseModel):
    """Metadata for a batch job."""

    id: str = Field(default_factory=lambda: f"batch-{shortuuid.uuid()[:8]}", description="Unique batch ID")
    flow: str = Field(..., description="Name of the flow being executed")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(), description="Creation timestamp")
    started_at: str | None = Field(default=None, description="Execution start timestamp")
    completed_at: str | None = Field(default=None, description="Execution completion timestamp")
    total_jobs: int = Field(..., description="Total number of jobs in this batch")
    completed_jobs: int = Field(default=0, description="Number of completed jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    status: BatchJobStatus = Field(default=BatchJobStatus.PENDING, description="Overall batch status")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Common parameters for all jobs in the batch")
    interactive: bool = Field(default=False, description="Whether this batch requires interactive input")

    @property
    def progress(self) -> float:
        """Calculate the progress percentage of the batch."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs + self.failed_jobs) / self.total_jobs
