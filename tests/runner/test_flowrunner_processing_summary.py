"""Integration tests for FlowRunner.run_batch_job with ProcessingSummary.

These tests verify that run_batch_job correctly returns ProcessingSummary
with accurate tracking of attempted, processed, failed jobs and progress bar integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from buttermilk._core.types import ProcessingSummary
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture
def mock_job_queue_client():
    """Mock JobQueueClient for testing batch processing."""
    with patch("buttermilk.runner.flowrunner.JobQueueClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_flow_runner_base(real_bm):
    """Create a FlowRunner with mocked dependencies for testing."""
    # Create a minimal FlowRunner instance
    # Note: FlowRunner requires proper initialization with real_bm
    runner = MagicMock(spec=FlowRunner)
    runner.run_flow = AsyncMock()
    runner.schedule_ack_on_completion = MagicMock()

    # Bind run_batch_job to the instance so it can access mocked methods
    async def run_batch_job_wrapper(*args, **kwargs):
        return await FlowRunner.run_batch_job(runner, *args, **kwargs)

    runner.run_batch_job = run_batch_job_wrapper

    return runner


class TestFlowRunnerProcessingSummary:
    """Test suite for FlowRunner.run_batch_job ProcessingSummary integration."""

    async def test_run_batch_job_returns_processing_summary(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that run_batch_job returns ProcessingSummary."""
        # Setup mock to return one job then None
        mock_run_request = MagicMock()
        mock_run_request.flow = "test_flow"
        mock_run_request.job_id = "test_job_123"
        mock_run_request.session_id = "session_123"

        mock_job_queue_client.pull_single_task = AsyncMock(
            side_effect=[
                (mock_run_request, "ack_123"),  # First call returns a job
                (None, None),  # Second call returns None (no more jobs)
            ]
        )

        callback = MagicMock()

        # Run batch job
        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=1,
            wait_for_completion=True,
            show_progress=False,
        )

        # Verify summary is returned and is correct type
        assert isinstance(summary, ProcessingSummary)
        assert summary.attempted >= 0
        assert summary.processed >= 0
        assert summary.failed >= 0

    async def test_run_batch_job_tracks_attempted_jobs(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that attempted counter is incremented correctly."""
        # Setup mock to return 3 jobs then None
        jobs = [
            (MagicMock(flow="test", job_id=f"job_{i}", session_id="s1"), f"ack_{i}")
            for i in range(3)
        ]
        jobs.append((None, None))  # Signal no more jobs

        mock_job_queue_client.pull_single_task = AsyncMock(side_effect=jobs)

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=3,
            wait_for_completion=True,
            show_progress=False,
        )

        # Verify 3 jobs were attempted
        assert summary.attempted == 3

    async def test_run_batch_job_tracks_processed_jobs(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that processed counter increments on successful job completion."""
        # Setup successful job execution
        mock_run_request = MagicMock()
        mock_run_request.flow = "test_flow"
        mock_run_request.job_id = "job_success"
        mock_run_request.session_id = "session_123"

        mock_job_queue_client.pull_single_task = AsyncMock(
            side_effect=[
                (mock_run_request, "ack_123"),
                (None, None),
            ]
        )

        # Mock successful run_flow execution
        mock_flow_runner_base.run_flow = AsyncMock(return_value=None)

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=1,
            wait_for_completion=True,
            show_progress=False,
        )

        # Verify job was processed successfully
        assert summary.attempted == 1
        assert summary.processed == 1
        assert summary.failed == 0

    async def test_run_batch_job_tracks_failed_jobs(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that failed counter increments when job execution fails."""
        # Setup job that will fail
        mock_run_request = MagicMock()
        mock_run_request.flow = "test_flow"
        mock_run_request.job_id = "job_fail"
        mock_run_request.session_id = "session_123"

        mock_job_queue_client.pull_single_task = AsyncMock(
            side_effect=[
                (mock_run_request, "ack_123"),
                (None, None),
            ]
        )

        # Mock run_flow to raise an exception
        mock_flow_runner_base.run_flow = AsyncMock(
            side_effect=Exception("Job execution failed")
        )

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=1,
            wait_for_completion=True,
            show_progress=False,
        )

        # Verify job was tracked as failed
        assert summary.attempted == 1
        assert summary.processed == 0
        assert summary.failed == 1

    async def test_run_batch_job_mixed_success_and_failure(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test processing with mix of successful and failed jobs."""
        # Setup 3 jobs
        jobs = []
        for i in range(3):
            mock_req = MagicMock()
            mock_req.flow = "test_flow"
            mock_req.job_id = f"job_{i}"
            mock_req.session_id = f"session_{i}"
            jobs.append((mock_req, f"ack_{i}"))
        jobs.append((None, None))

        mock_job_queue_client.pull_single_task = AsyncMock(side_effect=jobs)

        # Mock run_flow to succeed for job 0 and 2, fail for job 1
        call_count = [0]

        async def mock_run_flow(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 1:  # Second job fails
                raise Exception("Job 1 failed")
            return None

        mock_flow_runner_base.run_flow = mock_run_flow

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=3,
            wait_for_completion=True,
            show_progress=False,
        )

        # Verify mixed results
        assert summary.attempted == 3
        assert summary.processed == 2  # Jobs 0 and 2
        assert summary.failed == 1  # Job 1

    async def test_run_batch_job_success_rate_calculation(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that success rate is calculated correctly."""
        # Setup 5 jobs: 3 success, 2 fail
        jobs = []
        for i in range(5):
            mock_req = MagicMock()
            mock_req.flow = "test_flow"
            mock_req.job_id = f"job_{i}"
            mock_req.session_id = f"session_{i}"
            jobs.append((mock_req, f"ack_{i}"))
        jobs.append((None, None))

        mock_job_queue_client.pull_single_task = AsyncMock(side_effect=jobs)

        # Jobs 1 and 3 fail
        call_count = [0]

        async def mock_run_flow(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx in [1, 3]:
                raise Exception(f"Job {idx} failed")
            return None

        mock_flow_runner_base.run_flow = mock_run_flow

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=5,
            wait_for_completion=True,
            show_progress=False,
        )

        # Verify success rate (3/5 = 0.6)
        assert summary.attempted == 5
        assert summary.processed == 3
        assert summary.failed == 2
        assert abs(summary.success_rate() - 0.6) < 0.001

    async def test_run_batch_job_progress_bar_disabled(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that progress bar can be disabled with show_progress=False."""
        mock_run_request = MagicMock()
        mock_run_request.flow = "test_flow"
        mock_run_request.job_id = "test_job"
        mock_run_request.session_id = "session_123"

        mock_job_queue_client.pull_single_task = AsyncMock(
            side_effect=[
                (mock_run_request, "ack_123"),
                (None, None),
            ]
        )

        callback = MagicMock()

        # Should not raise any errors even though progress bar is disabled
        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=1,
            wait_for_completion=True,
            show_progress=False,
        )

        assert isinstance(summary, ProcessingSummary)

    async def test_run_batch_job_no_jobs_raises_fatal_error(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that FatalError is raised when no jobs are available."""
        from buttermilk._core.error_handling import FatalError

        # No jobs available
        mock_job_queue_client.pull_single_task = AsyncMock(return_value=(None, None))

        callback = MagicMock()

        # Should raise FatalError when no jobs found
        with pytest.raises(FatalError, match="No run request found"):
            await mock_flow_runner_base.run_batch_job(
                callback_to_ui=callback,
                max_jobs=1,
                wait_for_completion=True,
                show_progress=False,
            )

    async def test_run_batch_job_respects_max_jobs_limit(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that max_jobs parameter limits number of jobs processed."""
        # Setup 10 jobs but we'll only process 3
        jobs = []
        for i in range(10):
            mock_req = MagicMock()
            mock_req.flow = "test_flow"
            mock_req.job_id = f"job_{i}"
            mock_req.session_id = f"session_{i}"
            jobs.append((mock_req, f"ack_{i}"))
        jobs.append((None, None))

        mock_job_queue_client.pull_single_task = AsyncMock(side_effect=jobs)

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=3,  # Limit to 3 jobs
            wait_for_completion=True,
            show_progress=False,
        )

        # Should only process 3 jobs even though more are available
        assert summary.attempted == 3

    async def test_run_batch_job_duration_tracking(
        self, mock_flow_runner_base, mock_job_queue_client
    ):
        """Test that summary tracks duration correctly."""
        import time

        mock_run_request = MagicMock()
        mock_run_request.flow = "test_flow"
        mock_run_request.job_id = "test_job"
        mock_run_request.session_id = "session_123"

        mock_job_queue_client.pull_single_task = AsyncMock(
            side_effect=[
                (mock_run_request, "ack_123"),
                (None, None),
            ]
        )

        # Add delay to run_flow to ensure duration is tracked
        async def slow_run_flow(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return None

        import asyncio
        mock_flow_runner_base.run_flow = slow_run_flow

        callback = MagicMock()

        summary = await mock_flow_runner_base.run_batch_job(
            callback_to_ui=callback,
            max_jobs=1,
            wait_for_completion=True,
            show_progress=False,
        )

        # Duration should be at least 100ms
        assert summary.duration_ms() >= 100
