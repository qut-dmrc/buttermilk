from typing import Protocol, runtime_checkable

from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.result import BatchExecutionResult, BatchJobStatus


@runtime_checkable
class BatchExecutor(Protocol):
    """Protocol for batch execution strategies."""

    async def execute(
        self,
        records: list[BaseRecord],
        processor: BatchProcessorCore,
    ) -> BatchExecutionResult:
        """Execute batch processing and return results.

        Args:
            records: List of records to process
            processor: The batch processor to use

        Returns:
            BatchExecutionResult containing records (if sync) or job ID (if async)
        """
        ...

    async def get_status(self, job_id: str) -> BatchJobStatus:
        """Get current execution status."""
        ...
