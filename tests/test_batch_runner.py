import pytest

from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.sync import SyncBatchExecutor
from buttermilk.batch.runner import BatchPipelineRunner


class MockBatchProcessor(BatchProcessorCore):
    """Simple mock batch processor that uppercases content."""

    async def _process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        results = []
        for record in records:
            if isinstance(record.content, str):
                new_content = record.content.upper()
                results.append(record.model_copy(update={"content": new_content}))
        return results


@pytest.mark.anyio
async def test_batch_pipeline_runner_sync_flow():
    """Test the basic sync execution flow."""
    # 1. Setup Source
    source_records = [
        BaseRecord(record_id="rec-1", content="alpha"),
        BaseRecord(record_id="rec-2", content="beta"),
    ]

    # 2. Setup Processor
    processor = MockBatchProcessor()

    # 3. Setup Runner (Sync)
    runner = BatchPipelineRunner(name="test_batch_run", source=source_records, batch_processor=processor, executor=SyncBatchExecutor(), batch_size=2)

    # 4. Run
    result = await runner.run()

    # 5. Verify Results
    assert result.status == "COMPLETED"
    assert len(result.output_records) == 2
    assert result.output_records[0].content == "ALPHA"
    assert result.output_records[1].content == "BETA"
    assert result.processed_count == 2
