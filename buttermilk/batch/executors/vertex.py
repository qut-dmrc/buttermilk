from buttermilk import logger
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.managers.vertex import BatchJobManager
from buttermilk.batch.executors.base import BatchExecutor
from buttermilk.batch.result import BatchExecutionResult, BatchJobStatus


class VertexBatchExecutor(BatchExecutor):
    """Executes batch processing asynchronously on Vertex AI.

    Submits a batch job and returns immediately with PENDING status.
    Requires a processor that supports `prepare_batch_requests`
    (e.g. BatchLLMProcessor or VertexBatchProcessor).
    """

    def __init__(self, poll_interval: int = 30) -> None:
        self.poll_interval = poll_interval
        # Initialize manager lazily or here? Need client.
        # BatchJobManager requires 'client'. We can get it from 'bm.genai'.
        self._manager: BatchJobManager | None = None

    @property
    def manager(self) -> BatchJobManager:
        if self._manager is None:
            from buttermilk import bm

            self._manager = BatchJobManager(client=bm.genai, poll_interval=self.poll_interval)
        return self._manager

    async def execute(
        self,
        records: list[BaseRecord],
        processor: BatchProcessorCore,
    ) -> BatchExecutionResult:
        """Submit batch job to Vertex AI."""

        # 1. Validate Processor Compatibility
        if not hasattr(processor, "prepare_batch_requests"):
            return BatchExecutionResult(
                status=BatchJobStatus.FAILED,
                error=f"Processor {processor.name} ({type(processor).__name__}) does not support Vertex batch execution. Missing 'prepare_batch_requests'.",
            )

        # 2. Prepare Requests
        try:
            # Duck typing: Assume method exists and returns list[BatchRequest]
            requests = processor.prepare_batch_requests(records)  # type: ignore

            # Get model from processor
            model = getattr(processor, "model", None)
            if not model:
                return BatchExecutionResult(status=BatchJobStatus.FAILED, error=f"Processor {processor.name} missing 'model' attribute.")

            # 3. Submit Job
            job = await self.manager.submit_batch(
                model=model,
                requests=requests,
            )

            logger.info(f"Vertex batch job submitted: {job.name}")

            return BatchExecutionResult(status=BatchJobStatus.PENDING, job_id=job.name, metadata={"model": model, "request_count": len(requests)})

        except Exception as e:
            logger.error(f"Failed to submit Vertex batch job: {e}")
            return BatchExecutionResult(
                status=BatchJobStatus.FAILED,
                error=str(e),
            )

    async def get_status(self, job_id: str) -> BatchJobStatus:
        """Get status of the batch job."""
        try:
            # We don't have the full BatchJob object here, just ID (name).
            # BatchJobManager.wait_for_completion takes a job object.
            # We need a way to check status by name.
            # Direct client access or enhance BatchJobManager.

            # Using client directly for now to avoid modifying BatchJobManager interface too much
            # unless we refactor it.
            # Actually, BatchJobManager has 'client' public.
            try:
                job = self.manager.client.batches.get(name=job_id)
            except Exception:
                # If lookup fails, maybe it's not found or permission error
                return BatchJobStatus.FAILED

            from google.genai.types import JobState

            state_map = {
                JobState.JOB_STATE_SUCCEEDED: BatchJobStatus.COMPLETED,
                JobState.JOB_STATE_FAILED: BatchJobStatus.FAILED,
                JobState.JOB_STATE_CANCELLED: BatchJobStatus.CANCELLED,
                JobState.JOB_STATE_PAUSED: BatchJobStatus.RUNNING,  # Treat paused as running/pending for now
                JobState.JOB_STATE_RUNNING: BatchJobStatus.RUNNING,
                JobState.JOB_STATE_PENDING: BatchJobStatus.PENDING,
            }

            return state_map.get(job.state, BatchJobStatus.RUNNING)

        except Exception as e:
            logger.error(f"Error checking job status {job_id}: {e}")
            return BatchJobStatus.FAILED
