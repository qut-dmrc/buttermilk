"""BatchRunner service for managing batch job execution.

This module contains the BatchRunner class, which is responsible for:
- Creating batch jobs from flow configurations
- Extracting and shuffling record IDs
- Managing job execution and tracking via Google Pub/Sub
"""

import asyncio
from typing import Any

from pydantic import BaseModel, PrivateAttr

from buttermilk._core import logger
from buttermilk._core.types import RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.runner.flowrunner import FlowRunner


class BatchRunner(BaseModel):
    """Service for managing batch job execution.

    BatchRunner is responsible for:
    - Creating batch jobs from flow configurations
    - Extracting and shuffling record IDs
    - Publishing jobs to the Pub/Sub queue
    - Maintaining batch metadata
    """

    job_queue: JobQueueClient | None = None
    _pending_jobs: dict[str, list[RunRequest]] = PrivateAttr(default_factory=dict)
    _running_jobs: set[str] = PrivateAttr(default_factory=set)
    _job_results: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _running: bool = PrivateAttr(default=False)
    _worker_task: asyncio.Task | None = PrivateAttr(default=None)

    flow_runner: FlowRunner

    def __init__(self, **data):
        """Initialize the BatchRunner."""
        super().__init__(**data)

        # Initialize job queue if not provided
        if self.job_queue is None:
            try:
                self.job_queue = JobQueueClient()
                logger.info("Initialized job queue client")
            except Exception as e:
                logger.warning(f"Failed to create job queue client: {e}. Jobs will be processed locally.")

    async def get_batch_jobs(self, batch_id: str) -> list[RunRequest]:
        """Get job definitions for a batch.

        Args:
            batch_id: The ID of the batch

        Returns:
            List of job definitions

        Raises:
            ValueError: If batch_id is not found

        """
        if batch_id not in self._active_batches:
            raise ValueError(f"Batch '{batch_id}' not found")

        # Combine pending jobs with completed/failed jobs from results
        jobs = self._pending_jobs.get(batch_id, []).copy()

        # Add any jobs that are no longer pending (completed/failed)
        if batch_id in self._job_results:
            for job_id, result in self._job_results[batch_id].items():
                job_def = result.get("job_definition")
                if job_def and job_def not in jobs:
                    jobs.append(job_def)

        return jobs

    async def get_job_result(self, batch_id: str, record_id: str) -> dict[str, Any]:
        """Get the result of a specific job.

        Args:
            batch_id: The ID of the batch
            record_id: The record ID of the job

        Returns:
            The job result data

        Raises:
            ValueError: If batch or job not found

        """
        if batch_id not in self._job_results:
            raise ValueError(f"Batch '{batch_id}' not found")

        for job_id, result in self._job_results[batch_id].items():
            job_def = result.get("job_definition")
            if job_def and job_def.record_id == record_id:
                return result

        raise ValueError(f"Task for record '{record_id}' in batch '{batch_id}' not found")  # Updated message
