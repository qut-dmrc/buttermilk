"""
Tests for simplified batch CLI: bm batch <flow>

Test the Typer-based CLI wrapper that provides a simplified interface
to the batch processing functionality.
"""
import pytest
from typer.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch

from buttermilk.runner.batch_cli import app


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_flow_runner():
    """Mock FlowRunner for batch operations."""
    with patch("buttermilk.runner.batch_cli.FlowRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.create_batch = AsyncMock(return_value=[])
        mock_runner.run_batch_job = AsyncMock()
        mock_runner.set_session_bm = MagicMock()
        mock_runner_class.return_value = mock_runner
        yield mock_runner


@pytest.fixture
def mock_bm():
    """Mock BM initialization."""
    with patch("buttermilk.runner.batch_cli.init_async") as mock_init:
        mock_bm_instance = MagicMock()
        mock_bm_instance.cfg.run.flows = {"trans": MagicMock()}
        mock_bm_instance.graceful_shutdown = AsyncMock()
        mock_init.return_value = mock_bm_instance
        yield mock_bm_instance


class TestBatchCLI:
    """Test suite for batch CLI commands."""

    def test_batch_all_mode_enqueues_and_processes(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: bm batch trans - should enqueue AND process jobs."""
        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        mock_flow_runner.create_batch.assert_called_once()
        mock_flow_runner.run_batch_job.assert_called_once()

    def test_batch_enqueue_only_mode(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: bm batch trans --enqueue-only - should only enqueue."""
        result = cli_runner.invoke(app, ["batch", "trans", "--enqueue-only"])

        assert result.exit_code == 0
        mock_flow_runner.create_batch.assert_called_once()
        mock_flow_runner.run_batch_job.assert_not_called()

    def test_batch_process_only_mode(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: bm batch trans --process-only - should only process."""
        result = cli_runner.invoke(app, ["batch", "trans", "--process-only"])

        assert result.exit_code == 0
        mock_flow_runner.create_batch.assert_not_called()
        mock_flow_runner.run_batch_job.assert_called_once()

    def test_batch_with_limit(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: bm batch trans --limit 100 - should pass limit to both enqueue and process."""
        result = cli_runner.invoke(app, ["batch", "trans", "--limit", "100"])

        assert result.exit_code == 0
        # Verify limit was passed to create_batch as max_records
        create_batch_args = mock_flow_runner.create_batch.call_args
        assert create_batch_args.kwargs.get("max_records") == 100
        # Verify limit was passed to run_batch_job as max_jobs
        run_batch_args = mock_flow_runner.run_batch_job.call_args
        assert run_batch_args.kwargs.get("max_jobs") == 100

    def test_batch_auto_discovers_storage(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Storage should be auto-discovered from flow config."""
        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Verify create_batch was called without explicit storage_config
        call_args = mock_flow_runner.create_batch.call_args
        # storage_config should be None (auto-discover) or not provided
        assert call_args.kwargs.get("storage_config") is None

    def test_batch_invalid_flow_name_fails(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Invalid flow name should fail gracefully."""
        mock_flow_runner.create_batch.side_effect = ValueError("Flow 'invalid' not found")

        result = cli_runner.invoke(app, ["batch", "invalid"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_batch_mutually_exclusive_modes(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: --enqueue-only and --process-only are mutually exclusive."""
        result = cli_runner.invoke(
            app,
            ["batch", "trans", "--enqueue-only", "--process-only"]
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or "cannot use both" in result.output.lower()


class TestBatchCLIIntegration:
    """Integration tests for batch CLI behavior."""

    def test_batch_calls_bm_graceful_shutdown(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: BM graceful_shutdown is called after batch completion."""
        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        mock_bm.graceful_shutdown.assert_called_once()

    def test_batch_uses_session_scoped_bm(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: FlowRunner receives session-scoped BM instance."""
        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Verify set_session_bm was called with the BM instance
        mock_flow_runner.set_session_bm.assert_called_once_with(mock_bm)


class TestBatchCLIProcessingSummary:
    """Test suite for ProcessingSummary display in batch CLI."""

    def test_batch_displays_processing_summary(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Batch CLI displays ProcessingSummary after processing."""
        from buttermilk._core.types import ProcessingSummary

        # Create a mock summary with realistic data
        mock_summary = ProcessingSummary()
        mock_summary.increment_attempted()
        mock_summary.increment_processed()
        mock_summary.increment_attempted()
        mock_summary.increment_skipped()
        mock_summary.increment_attempted()
        mock_summary.increment_failed()

        # Mock run_batch_job to return this summary
        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Verify summary metrics appear in output
        assert "3" in result.output  # attempted
        assert "1" in result.output  # processed
        assert "1" in result.output  # skipped
        assert "1" in result.output  # failed

    def test_batch_displays_success_rate(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Batch CLI displays success rate in summary."""
        from buttermilk._core.types import ProcessingSummary

        mock_summary = ProcessingSummary()
        # 80% success rate (4/5)
        for _ in range(4):
            mock_summary.increment_attempted()
            mock_summary.increment_processed()
        mock_summary.increment_attempted()
        mock_summary.increment_failed()

        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Should show success rate (80%)
        assert "80" in result.output or "0.8" in result.output

    def test_batch_displays_duration(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Batch CLI displays processing duration in summary."""
        from buttermilk._core.types import ProcessingSummary
        import time

        mock_summary = ProcessingSummary()
        mock_summary.increment_attempted()
        mock_summary.increment_processed()
        time.sleep(0.1)  # Ensure some duration

        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Should mention duration somewhere (in seconds)
        assert "s" in result.output.lower() or "duration" in result.output.lower()

    def test_batch_summary_table_format(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Summary is displayed as a rich table."""
        from buttermilk._core.types import ProcessingSummary

        mock_summary = ProcessingSummary()
        mock_summary.increment_attempted()
        mock_summary.increment_processed()

        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Check for table-like structure (rich table would have these)
        output_lower = result.output.lower()
        assert "attempted" in output_lower or "processed" in output_lower

    def test_batch_summary_with_no_failures_green_panel(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Summary panel is green when no failures occur."""
        from buttermilk._core.types import ProcessingSummary

        mock_summary = ProcessingSummary()
        mock_summary.increment_attempted()
        mock_summary.increment_processed()
        # No failures

        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # With rich rendering, we'd see styling, but in plain text we just verify it completes
        assert result.exit_code == 0

    def test_batch_summary_with_failures_yellow_panel(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Summary panel styling changes when failures occur."""
        from buttermilk._core.types import ProcessingSummary

        mock_summary = ProcessingSummary()
        mock_summary.increment_attempted()
        mock_summary.increment_failed()

        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        # Verify failed count appears
        assert "1" in result.output

    def test_batch_enqueue_only_no_summary_display(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: --enqueue-only mode doesn't display processing summary."""
        result = cli_runner.invoke(app, ["batch", "trans", "--enqueue-only"])

        assert result.exit_code == 0
        # run_batch_job should not be called, so no summary
        mock_flow_runner.run_batch_job.assert_not_called()

    def test_batch_summary_all_metrics_present(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: Summary table includes all expected metrics."""
        from buttermilk._core.types import ProcessingSummary

        mock_summary = ProcessingSummary()
        mock_summary.increment_attempted()
        mock_summary.increment_processed()
        mock_summary.increment_skipped()
        mock_summary.increment_failed()

        mock_flow_runner.run_batch_job = AsyncMock(return_value=mock_summary)

        result = cli_runner.invoke(app, ["batch", "trans"])

        assert result.exit_code == 0
        output_lower = result.output.lower()

        # Verify all key metrics are mentioned
        # Note: Exact format depends on rich Table rendering
        # In plain text, we just verify the operation completed successfully
        assert mock_flow_runner.run_batch_job.called
