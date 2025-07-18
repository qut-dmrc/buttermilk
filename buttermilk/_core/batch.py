"""Defines batch processing models and utilities for Buttermilk.

This module contains the data models and core components for batch job configuration,
submission, and tracking in the Buttermilk system.
"""

from enum import Enum


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
