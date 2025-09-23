"""Minimal batch types for job queue functionality."""

from enum import Enum


class BatchJobStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"