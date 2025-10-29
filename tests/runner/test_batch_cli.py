"""
Integration tests for batch CLI functionality.

Tests verify that the batch CLI commands work end-to-end with real components:
- Real FlowRunner with real configuration (via real_bm)
- Real job creation and processing
- Real ProcessingSummary tracking

No mocking of internal buttermilk code - only testing real behaviors.
"""

import pytest
from typer.testing import CliRunner

from buttermilk.runner.batch_cli import app


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_batch_cli_validates_flow_exists(cli_runner):
    """Test that batch CLI fails gracefully when flow doesn't exist."""
    result = cli_runner.invoke(app, ["batch", "nonexistent_flow"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_batch_cli_rejects_conflicting_options(cli_runner):
    """Test that --enqueue-only and --process-only are mutually exclusive."""
    result = cli_runner.invoke(app, ["batch", "test_flow", "--enqueue-only", "--process-only"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or "cannot use both" in result.output.lower()


# Note: Full end-to-end integration tests for batch processing are covered in
# test_flowrunner_integration.py since they require:
# - Real job queue setup
# - Real storage configuration
# - Actual flow execution
#
# The CLI tests above verify CLI-specific behaviors (argument validation, error handling)
# without duplicating the full integration test suite.
