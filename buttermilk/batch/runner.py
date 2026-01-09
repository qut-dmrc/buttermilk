from typing import AsyncGenerator

from pydantic import ConfigDict, Field

from buttermilk import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import BatchProcessorCore, ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.base import BatchExecutor
from buttermilk.batch.executors.sync import SyncBatchExecutor
from buttermilk.batch.result import BatchJobStatus, BatchRunResult


class BatchPipelineRunner(ProcessorCore):
    """Orchestrates a batch processing pipeline.

    This runner:
    1. Consumes ALL records from source (or a large chunk)
    2. Runs expanders (if any) to multiply records
    3. Executes the batch_processor using the configured executor
    4. Saves results (optional)
    """

    source: list[BaseRecord] | AsyncGenerator[BaseRecord, None] = Field(..., description="Input records")
    batch_processor: BatchProcessorCore = Field(..., description="Core batch operation")
    executor: BatchExecutor = Field(default_factory=SyncBatchExecutor, description="Execution strategy (Sync/Vertex/etc)")

    expanders: list[ProcessorCore] = Field(default_factory=list, description="Pre-batch expansion/filtering processors")
    result_saver: ProcessorCore | None = Field(default=None, description="Optional result persistence")

    batch_size: int = Field(default=50, description="Batch size for execution chunks")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run(self) -> BatchRunResult:
        """Execute the batch pipeline."""
        logger.info(f"Starting BatchPipelineRunner: {self.name}", batch_processor=self.batch_processor.name)

        # 1. Collect Records
        input_records = await self._collect_records()
        logger.info(f"Collected {len(input_records)} input records")

        if not input_records:
            return BatchRunResult(status=BatchJobStatus.COMPLETED, output_records=[])

        # 2. Run Pre-batch Processors (Expanders/Filters)
        expanded_records = await self._run_expanders(input_records)
        logger.info(f"Records after expansion: {len(expanded_records)}")

        if not expanded_records:
            return BatchRunResult(status=BatchJobStatus.COMPLETED, output_records=[])

        # 3. Execute Batch
        # TODO: Chunking support if executor requires it. For now, pass all.
        try:
            output_records = await self.executor.execute(expanded_records, self.batch_processor)
            status = BatchJobStatus.COMPLETED

            # 4. Save Results
            if self.result_saver:
                # TODO: Implement saving logic
                pass

            return BatchRunResult(status=status, output_records=output_records, processed_count=len(output_records))

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            return BatchRunResult(status=BatchJobStatus.FAILED, error=str(e), processed_count=0)

    async def _collect_records(self) -> list[BaseRecord]:
        """Collect functionality handling generator or list."""
        if isinstance(self.source, list):
            return self.source

        records = []
        async for record in self.source:
            records.append(record)
        return records

    async def _run_expanders(self, records: list[BaseRecord]) -> list[BaseRecord]:
        """Run standard processors on the list of records."""
        if not self.expanders:
            return records

        current_records = records
        for proc in self.expanders:
            next_records = []
            for record in current_records:
                # Create context for each record
                # TODO: Use shared session ID
                ctx = ProcessingContext(session_id="batch_preproc", record=record)
                async for res in proc.process(ctx):
                    next_records.append(res)
            current_records = next_records

        return current_records

    # Protocol implementation (unused here but good for consistency)
    async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[BaseRecord, None]:
        yield context.record
