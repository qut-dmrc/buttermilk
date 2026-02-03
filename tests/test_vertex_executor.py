from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import BatchJob, JobState

from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.vertex import BatchExecutionResult, BatchJobStatus, VertexBatchExecutor


# Define a mock processor compatible with VertexBatchExecutor
class MockVertexProcessor(BatchProcessorCore):
    model: str = "gemini-1.5-flash-002"

    def _process_batch(self, batch):
        return []

    def prepare_batch_requests(self, records):
        return [{"request": "mock"}] * len(records)


@pytest.fixture
def vertex_executor():
    return VertexBatchExecutor(poll_interval=1)


@pytest.fixture
def mock_processor():
    return MockVertexProcessor()


@pytest.fixture
def sample_records():
    return [BaseRecord(id="1", content="test"), BaseRecord(id="2", content="test2")]


@pytest.mark.anyio
async def test_vertex_executor_submits_job(real_bm, vertex_executor, mock_processor, sample_records):
    # 1. Setup Mock BatchJobManager
    with patch("buttermilk.batch.executors.vertex.BatchJobManager") as MockManager:
        # Configure the mock manager instance
        mock_manager_instance = MockManager.return_value

        # Configure submit_batch return value
        mock_job = MagicMock(spec=BatchJob)
        mock_job.name = "projects/123/locations/us-central1/batchPredictionJobs/job-456"
        mock_manager_instance.submit_batch = AsyncMock(return_value=mock_job)

        # 2. Execute
        result = await vertex_executor.execute(sample_records, mock_processor)

        # 3. Verify Result
        assert isinstance(result, BatchExecutionResult)
        assert result.status == BatchJobStatus.PENDING
        assert result.job_id == "projects/123/locations/us-central1/batchPredictionJobs/job-456"
        assert result.metadata["model"] == "gemini-1.5-flash-002"
        assert result.metadata["request_count"] == 2

        # 4. Verify interactions
        # Verify BatchJobManager was initialized with correct args (we can't easily check client=bm.genai without access to real_bm inside patch, but we know it runs)
        MockManager.assert_called_once()
        # We can check if client passed is the real_bm.genai
        assert MockManager.call_args.kwargs["client"] == real_bm.genai

        mock_manager_instance.submit_batch.assert_called_once()
        call_kwargs = mock_manager_instance.submit_batch.call_args.kwargs
        assert call_kwargs["model"] == "gemini-1.5-flash-002"
        assert len(call_kwargs["requests"]) == 2


@pytest.mark.anyio
async def test_vertex_executor_fails_if_processor_unsupported():
    """Verify executor fails if processor lacks prepare_batch_requests."""
    processor = MagicMock()  # generic mock
    del processor.prepare_batch_requests  # ensure method doesn't exist

    executor = VertexBatchExecutor()
    result = await executor.execute([], processor)

    assert result.status == BatchJobStatus.FAILED
    assert "does not support Vertex batch execution" in result.error


@pytest.mark.anyio
async def test_vertex_executor_get_status(real_bm, vertex_executor):
    job_id = "projects/123/locations/us-central1/batchPredictionJobs/job-456"

    # 1. Setup Mock BatchJobManager and Client
    with patch("buttermilk.batch.executors.vertex.BatchJobManager") as MockManager:
        mock_manager_instance = MockManager.return_value

        # Mock the client.batches.get method
        mock_client = MagicMock()
        mock_manager_instance.client = mock_client

        mock_job = MagicMock()
        mock_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_client.batches.get.return_value = mock_job

        # Pre-initialize the manager so the property doesn't create a new one that overwrites our mocks
        # Actually, if we just patch BatchJobManager, the property access will use it.
        # But we need to make sure the instance returned by the constructor (which is called on property access)
        # is the one we configured. Since MockManager.return_value IS that instance, we are good.

        # 2. Call get_status
        # Note: the executor lazily creates manager on first access
        status = await vertex_executor.get_status(job_id)

        # 3. Verify
        assert status == BatchJobStatus.COMPLETED
        mock_client.batches.get.assert_called_with(name=job_id)
