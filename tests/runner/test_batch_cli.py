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
        mock_bm_instance.cfg.flows = {"trans": MagicMock()}
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

    def test_batch_with_max_records(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: bm batch trans --max-records 100 - should pass max_records."""
        result = cli_runner.invoke(app, ["batch", "trans", "--max-records", "100"])

        assert result.exit_code == 0
        # Verify max_records was passed to create_batch
        call_args = mock_flow_runner.create_batch.call_args
        assert call_args.kwargs.get("max_records") == 100

    def test_batch_with_max_jobs(self, cli_runner, mock_flow_runner, mock_bm):
        """Test: bm batch trans --max-jobs 10 - should pass max_jobs."""
        result = cli_runner.invoke(app, ["batch", "trans", "--max-jobs", "10"])

        assert result.exit_code == 0
        # Verify max_jobs was passed to run_batch_job
        call_args = mock_flow_runner.run_batch_job.call_args
        assert call_args.kwargs.get("max_jobs") == 10

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
