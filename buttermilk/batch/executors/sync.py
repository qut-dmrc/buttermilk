from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.base import BatchExecutor
from buttermilk.batch.result import BatchExecutionResult, BatchJobStatus


class SyncBatchExecutor(BatchExecutor):
    """Executes batch processing synchronously in the current process."""

    async def execute(
        self,
        records: list[BaseRecord],
        processor: BatchProcessorCore,
    ) -> BatchExecutionResult:
        """Execute processing synchronously.

        Simply calls processor.process_batch(records).
        """
        try:
            output_records = await processor.process_batch(records)
            return BatchExecutionResult(status=BatchJobStatus.COMPLETED, output_records=output_records, processed_count=len(output_records))
        except Exception as e:
            return BatchExecutionResult(status=BatchJobStatus.FAILED, error=str(e), processed_count=0)

    async def get_status(self, job_id: str) -> BatchJobStatus:
        """Always running or completed since it's synchronous."""
        return BatchJobStatus.COMPLETED
