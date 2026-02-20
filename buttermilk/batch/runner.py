import asyncio
from typing import AsyncGenerator

from pydantic import ConfigDict, Field

from buttermilk import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import BatchProcessorCore, ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.base import BatchExecutor
from buttermilk.batch.executors.sync import SyncBatchExecutor
from buttermilk.batch.result import BatchExecutionResult, BatchJobStatus, BatchRunResult
from buttermilk.storage.base import Storage


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
    storage: Storage | None = Field(default=None, description="Optional result storage")

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
        try:
            execution_result = await self.executor.execute(expanded_records, self.batch_processor)

            # 4. Save Results (if completed)
            logger.info(f"Execution status: {execution_result.status}, storage configured: {bool(self.storage is not None)}")
            if execution_result.status == BatchJobStatus.COMPLETED and self.storage is not None:
                await self._save_results(execution_result)
            else:
                logger.info("Skipping save_results")

            return BatchRunResult(
                status=execution_result.status,
                output_records=execution_result.output_records,
                processed_count=len(execution_result.output_records),
                job_id=execution_result.job_id,
                output_uri=execution_result.output_uri,
                error=execution_result.error,
                metadata=execution_result.metadata,
            )

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            return BatchRunResult(status=BatchJobStatus.FAILED, error=str(e), processed_count=0)

    async def _save_results(self, result: BatchExecutionResult) -> None:
        """Save execution results to storage."""
        if self.storage is None:
            return

        logger.info("Saving batch results to storage")
        try:
            if result.output_uri:
                # Optimized path: Load directly from URI (e.g. GCS -> BQ)
                logger.debug(f"Loading results from URI: {result.output_uri}")
                # Note: load_from_uri is synchronous in base Storage currently, but might be async-wrapped if needed.
                # For now assuming sync call as per Storage interface.
                self.storage.load_from_uri(result.output_uri)
            elif result.output_records:
                # Standard path: Save records
                logger.debug(f"Saving {len(result.output_records)} records")
                self.storage.save(result.output_records)
            else:
                logger.warning("No records or URI to save")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # We don't raise here to allow returning the result object, but it's a significant error.
            # Depending on requirements, we might want to mark the run as partial success or failed.
            result.error = f"Result saving failed: {e}"

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

            async def _process_single(record: BaseRecord) -> list[BaseRecord]:
                """Process a single record and collect results."""
                # Create context for each record
                # TODO: Use shared session ID
                ctx = ProcessingContext(session_id="batch_preproc", record=record)
                return [res async for res in proc.process(ctx)]

            # Parallelize processing of all records for the current expander
            results = await asyncio.gather(*(_process_single(record) for record in current_records))

            # Flatten results for the next expander (or final output)
            current_records = [res for record_list in results for res in record_list]

        return current_records

    # Protocol implementation (unused here but good for consistency)
    async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[BaseRecord, None]:
        yield context.record
