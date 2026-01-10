from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from buttermilk._core.types import BaseRecord


class BatchJobStatus(StrEnum):
    """Status of a batch processing job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BatchRunResult(BaseModel):
    """Result of a batch pipeline run (returned to user)."""

    status: BatchJobStatus
    output_records: list[BaseRecord] = Field(default_factory=list)
    processed_count: int = 0
    error: str | None = None
    job_id: str | None = None
    output_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if the run completed successfully."""
        return self.status == BatchJobStatus.COMPLETED


class BatchExecutionResult(BaseModel):
    """Result from BatchExecutor (internal)."""

    status: BatchJobStatus
    output_records: list[BaseRecord] = Field(default_factory=list)
    job_id: str | None = None
    output_uri: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
