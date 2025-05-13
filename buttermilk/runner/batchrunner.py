"""BatchRunner service for managing batch job execution.

This module contains the BatchRunner class, which is responsible for:
- Creating batch jobs from flow configurations
- Extracting and shuffling record IDs 
- Managing job execution and tracking via Google Pub/Sub
"""

import asyncio
import random
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, PrivateAttr

from buttermilk._core.batch import BatchJobStatus, BatchMetadata, BatchRequest
from buttermilk._core.exceptions import FatalError
from buttermilk._core.types import RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.api.services.data_service import DataService
from buttermilk.bm import logger
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.utils.utils import expand_dict


class BatchRunner(BaseModel):
    """Service for managing batch job execution.
    
    BatchRunner is responsible for:
    - Creating batch jobs from flow configurations
    - Extracting and shuffling record IDs
    - Publishing jobs to the Pub/Sub queue
    - Maintaining batch metadata
    """

    job_queue: JobQueueClient | None = None
    _active_batches: dict[str, BatchMetadata] = PrivateAttr(default_factory=dict)
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

    async def create_batch(self, batch_request: BatchRequest) -> BatchMetadata:
        """Create a new batch job from the given request.
        
        Args:
            batch_request: The batch configuration
            
        Returns:
            The created batch metadata
            
        Raises:
            ValueError: If the flow doesn't exist or record extraction fails

        """
        # Extract record IDs from the flow's data source
        try:
            # Use our helper class to get the record IDs
            record_ids = await DataService.get_records_for_flow(flow_name=batch_request.flow, flow_runner=self.flow_runner)
            logger.info(f"Extracted {len(record_ids)} record IDs for flow '{batch_request.flow}'")
        except Exception as e:
            logger.error(f"Failed to extract record IDs for flow '{batch_request.flow}': {e}")
            raise ValueError(f"Failed to extract record IDs: {e}")

        # Iterate batch parameters when asked (otherwise uses flow defaults configured on the runner)
        iteration_values = expand_dict(batch_request.parameters) or [{}]

        # Shuffle if requested
        if batch_request.shuffle:
            random.shuffle(record_ids)
            logger.debug(f"Shuffled {len(record_ids)} record IDs")

        # Apply max_records limit if specified
        if batch_request.max_records is not None:
            record_ids = record_ids[:batch_request.max_records]
            logger.debug(f"Limited to {len(record_ids)} record IDs (max_records={batch_request.max_records})")

        # Create batch metadata
        batch_metadata = BatchMetadata(
            flow=batch_request.flow,
            total_jobs=len(record_ids),
            parameters=batch_request.parameters,
            interactive=batch_request.interactive,
        )

        # Create job definitions and publish to queue if available
        job_definitions = []
        published_to_queue = False

        # Check if we can use the job queue
        use_queue = self.job_queue is not None

        # Apply iteration values
        for iteration_params in iteration_values:
            for record in record_ids:
                job = RunRequest(ui_type="batch",
                    batch_id=batch_metadata.id,
                    flow=batch_request.flow,
                    record_id=record["record_id"],
                    parameters=iteration_params,callback_to_ui=None,
                )
                job_definitions.append(job)

                # Publish to queue if available
                if use_queue and self.job_queue is not None:
                    try:
                        self.job_queue.publish_job(job)
                        published_to_queue = True
                    except Exception as e:
                        msg = f"Failed to publish job to queue: {e}"
                        use_queue = False  # Fall back to local processing
                        raise FatalError(msg) from e

        # Store batch metadata
        self._active_batches[batch_metadata.id] = batch_metadata

        logger.info(f"Created batch '{batch_metadata.id}' with {len(job_definitions)} jobs using params {batch_request.parameters} (published to queue: {published_to_queue})")

        return batch_metadata

    async def get_batch_status(self, batch_id: str) -> BatchMetadata:
        """Get status information for a batch.
        
        Args:
            batch_id: The ID of the batch to check
            
        Returns:
            The batch metadata
            
        Raises:
            ValueError: If batch_id is not found

        """
        if batch_id not in self._active_batches:
            raise ValueError(f"Batch '{batch_id}' not found")

        return self._active_batches[batch_id]

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

        raise ValueError(f"Job for record '{record_id}' in batch '{batch_id}' not found")

    async def cancel_batch(self, batch_id: str) -> BatchMetadata:
        """Cancel a batch job.
        
        Args:
            batch_id: The ID of the batch to cancel
            
        Returns:
            The updated batch metadata
            
        Raises:
            ValueError: If batch_id is not found

        """
        if batch_id not in self._active_batches:
            raise ValueError(f"Batch '{batch_id}' not found")

        batch = self._active_batches[batch_id]

        # Only allow cancelling if not completed
        if batch.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
            raise ValueError(f"Cannot cancel batch '{batch_id}' with status '{batch.status}'")

        # Update status
        batch.status = BatchJobStatus.CANCELLED
        batch.completed_at = datetime.now(UTC).isoformat()

        # Remove pending jobs
        self._pending_jobs[batch_id] = []

        logger.info(f"Cancelled batch '{batch_id}'")
        return batch
