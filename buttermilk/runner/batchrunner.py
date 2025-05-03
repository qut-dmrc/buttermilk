"""BatchRunner service for managing batch job execution.

This module contains the BatchRunner class, which is responsible for:
- Creating batch jobs from flow configurations
- Extracting and shuffling record IDs 
- Managing job execution and tracking
"""

import asyncio
import random
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, PrivateAttr

from buttermilk._core.batch import BatchErrorPolicy, BatchJobDefinition, BatchJobStatus, BatchMetadata, BatchRequest
from buttermilk.bm import logger
from buttermilk.runner.batch_helper import BatchRunnerHelper
from buttermilk.runner.flowrunner import FlowRunner


class BatchRunner(BaseModel):
    """Service for managing batch job execution.
    
    BatchRunner is responsible for:
    - Creating batch jobs from flow configurations
    - Extracting and shuffling record IDs
    - Managing job execution and tracking
    """

    flow_runner: FlowRunner
    _active_batches: dict[str, BatchMetadata] = PrivateAttr(default_factory=dict)
    _pending_jobs: dict[str, list[BatchJobDefinition]] = PrivateAttr(default_factory=dict)
    _running_jobs: set[str] = PrivateAttr(default_factory=set)
    _job_results: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _running: bool = PrivateAttr(default=False)
    _worker_task: asyncio.Task | None = PrivateAttr(default=None)

    async def create_batch(self, batch_request: BatchRequest) -> BatchMetadata:
        """Create a new batch job from the given request.
        
        Args:
            batch_request: The batch configuration
            
        Returns:
            The created batch metadata
            
        Raises:
            ValueError: If the flow doesn't exist or record extraction fails

        """
        # Validate flow exists
        if batch_request.flow not in self.flow_runner.flows:
            raise ValueError(f"Flow '{batch_request.flow}' not found. Available flows: {list(self.flow_runner.flows.keys())}")

        # Extract record IDs from the flow's data source
        try:
            # Use our helper class to get the record IDs
            record_ids = await BatchRunnerHelper.get_record_ids_for_flow(
                self.flow_runner, batch_request.flow
            )
            
            logger.info(f"Extracted {len(record_ids)} record IDs for flow '{batch_request.flow}'")
        except Exception as e:
            logger.error(f"Failed to extract record IDs for flow '{batch_request.flow}': {e}")
            raise ValueError(f"Failed to extract record IDs: {e}")

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

        # Create job definitions
        job_definitions = []
        for record in record_ids:
            job = BatchJobDefinition(
                batch_id=batch_metadata.id,
                flow=batch_request.flow,
                record_id=record["record_id"],
                parameters=batch_request.parameters,
            )
            job_definitions.append(job)

        # Store batch and jobs
        self._active_batches[batch_metadata.id] = batch_metadata
        self._pending_jobs[batch_metadata.id] = job_definitions
        self._job_results[batch_metadata.id] = {}

        logger.info(f"Created batch '{batch_metadata.id}' with {len(job_definitions)} jobs")

        # Start worker if not already running
        if not self._running:
            await self.start()

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

    async def get_batch_jobs(self, batch_id: str) -> list[BatchJobDefinition]:
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

    async def start(self) -> None:
        """Start the batch runner worker."""
        if self._running:
            logger.warning("BatchRunner already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Started BatchRunner worker")

    async def stop(self) -> None:
        """Stop the batch runner worker."""
        if not self._running:
            logger.warning("BatchRunner not running")
            return

        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        logger.info("Stopped BatchRunner worker")

    async def _worker_loop(self) -> None:
        """Main worker loop for processing batch jobs."""
        try:
            while self._running:
                # Process any pending jobs
                job = await self._get_next_job()

                if job:
                    # Process the job
                    batch_id = job.batch_id
                    record_id = job.record_id

                    job_id = f"{batch_id}:{record_id}"
                    if job_id in self._running_jobs:
                        # Job already running, skip
                        continue

                    # Mark job as running
                    self._running_jobs.add(job_id)
                    job.status = BatchJobStatus.RUNNING
                    job.started_at = datetime.now(UTC).isoformat()

                    # Update batch status if needed
                    batch = self._active_batches[batch_id]
                    if batch.status == BatchJobStatus.PENDING:
                        batch.status = BatchJobStatus.RUNNING
                        batch.started_at = datetime.now(UTC).isoformat()

                    # Run the job
                    asyncio.create_task(self._run_job(job))

                else:
                    # No pending jobs, sleep
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            self._running = False

    async def _get_next_job(self) -> BatchJobDefinition | None:
        """Get the next job to process."""
        for batch_id, jobs in self._pending_jobs.items():
            batch = self._active_batches[batch_id]

            # Skip cancelled batches
            if batch.status == BatchJobStatus.CANCELLED:
                continue

            # Check if we have any pending jobs
            pending_jobs = [j for j in jobs if j.status == BatchJobStatus.PENDING]
            if pending_jobs:
                # Get the first pending job
                job = pending_jobs[0]

                # Remove from pending list (will be added back if it fails)
                self._pending_jobs[batch_id].remove(job)

                return job

        return None

    async def _run_job(self, job: BatchJobDefinition) -> None:
        """Run a job and update its status."""
        batch_id = job.batch_id
        record_id = job.record_id
        job_id = f"{batch_id}:{record_id}"

        try:
            logger.info(f"Running job {job_id}")

            # Convert job to RunRequest
            run_request = job.to_run_request()

            # Run the flow
            callback = await self.flow_runner.run_flow(run_request)

            # Mark job as completed
            job.status = BatchJobStatus.COMPLETED
            job.completed_at = datetime.now(UTC).isoformat()

            # Store job result
            if batch_id not in self._job_results:
                self._job_results[batch_id] = {}

            self._job_results[batch_id][job_id] = {
                "job_definition": job,
                "result": callback,  # This might be a callback function, consider what you want to store
                "completed_at": job.completed_at,
            }

            # Update batch metadata
            batch = self._active_batches[batch_id]
            batch.completed_jobs += 1

            # Check if batch is complete
            if batch.completed_jobs + batch.failed_jobs >= batch.total_jobs:
                batch.status = BatchJobStatus.COMPLETED
                batch.completed_at = datetime.now(UTC).isoformat()
                logger.info(f"Batch {batch_id} completed")

        except Exception as e:
            logger.error(f"Error running job {job_id}: {e}", exc_info=True)

            # Mark job as failed
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.now(UTC).isoformat()
            job.error = str(e)

            # Store job result
            if batch_id not in self._job_results:
                self._job_results[batch_id] = {}

            self._job_results[batch_id][job_id] = {
                "job_definition": job,
                "error": str(e),
                "completed_at": job.completed_at,
            }

            # Update batch metadata
            batch = self._active_batches[batch_id]
            batch.failed_jobs += 1

            # Check if batch should be failed based on error policy
            if batch.parameters.get("error_policy") == BatchErrorPolicy.STOP:
                batch.status = BatchJobStatus.FAILED
                batch.completed_at = datetime.now(UTC).isoformat()
                logger.info(f"Batch {batch_id} failed due to error policy")

                # Remove all pending jobs
                self._pending_jobs[batch_id] = []

            # Check if batch is complete
            elif batch.completed_jobs + batch.failed_jobs >= batch.total_jobs:
                batch.status = BatchJobStatus.COMPLETED
                batch.completed_at = datetime.now(UTC).isoformat()
                logger.info(f"Batch {batch_id} completed with {batch.failed_jobs} failures")

        finally:
            # Remove from running jobs
            if job_id in self._running_jobs:
                self._running_jobs.remove(job_id)
