import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.sync import SyncBatchExecutor
from buttermilk.batch.runner import BatchPipelineRunner


class MockBatchProcessor(BatchProcessorCore):
    """Simple mock batch processor that uppercases content."""

    async def _process_batch(self, contexts: list[ProcessingContext]) -> list[BaseRecord]:
        results = []
        for ctx in contexts:
            record = ctx.record
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


@pytest.mark.anyio
async def test_batch_pipeline_runner_storage_save_records():
    """Test results are saved to storage when output_records are present."""
    from buttermilk.storage.base import Storage

    # Explain Pydantic validation workaround
    class MockStorage(Storage):
        def __init__(self):
            # No super init needed for ABC usually
            self.saved_records = []
            self.loaded_uris = []

        def save(self, records):
            self.saved_records.append(records)

        def load_from_uri(self, uri):
            self.loaded_uris.append(uri)

        def __iter__(self):
            yield from []

        def get_record_by_id(self, id):
            return None

        def count(self):
            return len(self.saved_records)

    # Setup
    source = [BaseRecord(record_id="1", content="a")]
    processor = MockBatchProcessor()

    mock_storage = MockStorage()

    runner = BatchPipelineRunner(
        name="test_storage_rec", source=source, batch_processor=processor, executor=SyncBatchExecutor(), storage=mock_storage
    )

    result = await runner.run()

    # Verify
    assert result.status == "COMPLETED"
    assert len(result.output_records) == 1
    # Check that storage.save was called
    assert len(mock_storage.saved_records) == 1
    saved_records = mock_storage.saved_records[0]
    assert len(saved_records) == 1
    assert saved_records[0].content == "A"


@pytest.mark.anyio
async def test_batch_pipeline_runner_storage_load_uri():
    """Test results are loaded from URI when output_uri is present."""
    from buttermilk.batch.executors.base import BatchExecutor
    from buttermilk.batch.result import BatchExecutionResult, BatchJobStatus
    from buttermilk.storage.base import Storage

    # Setup
    source = [BaseRecord(record_id="1", content="a")]
    processor = MockBatchProcessor()

    # Mock Executor using concrete class to pass Pydantic validation
    class MockExecutor(BatchExecutor):
        async def execute(self, records, processor):
            return BatchExecutionResult(status=BatchJobStatus.COMPLETED, output_uri="gs://test-bucket/results.jsonl", job_id="test-job-123")

        async def get_status(self, job_id: str) -> BatchJobStatus:
            return BatchJobStatus.COMPLETED

    mock_executor = MockExecutor()

    # Mock Storage
    class MockStorage(Storage):
        def __init__(self):
            self.saved_records = []
            self.loaded_uris = []

        def save(self, records):
            self.saved_records.append(records)

        def load_from_uri(self, uri):
            self.loaded_uris.append(uri)

        def __iter__(self):
            yield from []

        def get_record_by_id(self, id):
            return None

        def count(self):
            return 0

    mock_storage = MockStorage()

    runner = BatchPipelineRunner(name="test_storage_uri", source=source, batch_processor=processor, executor=mock_executor, storage=mock_storage)

    result = await runner.run()

    # Verify
    assert result.status == "COMPLETED"
    assert result.output_uri == "gs://test-bucket/results.jsonl"

    # Check that storage.load_from_uri was called (precedence over save)
    assert len(mock_storage.loaded_uris) == 1
    assert mock_storage.loaded_uris[0] == "gs://test-bucket/results.jsonl"
    assert len(mock_storage.saved_records) == 0
