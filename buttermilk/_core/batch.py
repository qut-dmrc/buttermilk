"""Defines batch processing models and utilities for Buttermilk.

This module contains the data models and core components for batch job configuration,
submission, and tracking in the Buttermilk system.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import shortuuid
from pydantic import BaseModel, Field


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
