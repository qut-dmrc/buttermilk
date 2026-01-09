from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.base import BatchExecutor
from buttermilk.batch.result import BatchJobStatus


class SyncBatchExecutor(BatchExecutor):
    """Executes batch processing synchronously in the current process."""

    async def execute(
        self,
        records: list[BaseRecord],
        processor: BatchProcessorCore,
    ) -> list[BaseRecord]:
        """Execute processing synchronously.

        Simply calls processor.process_batch(records).
        """
        return await processor.process_batch(records)

    async def get_status(self) -> BatchJobStatus:
        """Always running or completed since it's synchronous."""
        return BatchJobStatus.RUNNING
