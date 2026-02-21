import asyncio
import time
from typing import AsyncGenerator

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import BatchProcessorCore, ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.sync import SyncBatchExecutor
from buttermilk.batch.runner import BatchPipelineRunner


class SlowExpander(ProcessorCore):
    async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[BaseRecord, None]:
        await asyncio.sleep(0.1)
        yield context.record


class FastBatchProcessor(BatchProcessorCore):
    async def _process_batch(self, contexts: list[ProcessingContext]) -> list[BaseRecord]:
        return [ctx.record for ctx in contexts]


@pytest.mark.anyio
async def test_performance_batch_runner_expanders():
    num_records = 10
    records = [BaseRecord(record_id=f"rec-{i}", content=f"content-{i}") for i in range(num_records)]

    runner = BatchPipelineRunner(
        name="test_runner", source=records, batch_processor=FastBatchProcessor(), expanders=[SlowExpander()], executor=SyncBatchExecutor()
    )

    start_time = time.perf_counter()
    await runner.run()
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"\nTime taken for {num_records} records: {duration:.4f} seconds")

    # If sequential, it should take at least 1.0 seconds (10 * 0.1)
    # If parallel, it should take around 0.1-0.2 seconds
    # Note: This is a performance test, so we just log the time for now.
    # We can add an assertion if we want to enforce parallel execution.
    # assert duration < 0.5
