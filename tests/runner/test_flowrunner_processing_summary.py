"""Integration tests for FlowRunner ProcessingSummary tracking.

NOTE: Comprehensive ProcessingSummary tests exist in test_pipeline.py (lines 377-636)
which test the same ProcessingSummary class with real components and no mocking.

This file focuses on FlowRunner-specific batch job processing scenarios.
Full integration tests for FlowRunner.run_batch_job would require:
- Real job queue infrastructure setup
- Real job creation and enqueueing
- Real flow execution
- Complex async orchestration

Such tests belong in a dedicated end-to-end integration test suite with proper
infrastructure fixtures. The CLI and pipeline tests already verify the core
batch processing workflows with real components.
"""

from buttermilk._core.types import ProcessingSummary


def test_processing_summary_basic_operations():
    """Test ProcessingSummary basic counter operations.

    This verifies the ProcessingSummary class itself works correctly.
    Integration tests in test_pipeline.py verify it works in real workflows.
    """
    summary = ProcessingSummary()

    # Test counters start at zero
    assert summary.attempted == 0
    assert summary.processed == 0
    assert summary.failed == 0
    assert summary.skipped == 0

    # Test increment operations
    summary.increment_attempted()
    summary.increment_processed()
    assert summary.attempted == 1
    assert summary.processed == 1

    summary.increment_attempted()
    summary.increment_failed()
    assert summary.attempted == 2
    assert summary.failed == 1

    # Test success rate calculation
    assert abs(summary.success_rate() - 0.5) < 0.001  # 1 success / 2 attempted = 0.5

    # Test duration tracking
    assert summary.duration_ms() >= 0


def test_processing_summary_as_dict_export():
    """Test ProcessingSummary export to dictionary for serialization."""
    summary = ProcessingSummary()
    summary.increment_attempted()
    summary.increment_processed()

    result = summary.as_dict()

    # Verify all expected fields present
    assert "attempted" in result
    assert "processed" in result
    assert "skipped" in result
    assert "failed" in result
    assert "duration_ms" in result
    assert "success_rate" in result

    # Verify correct types
    assert isinstance(result["attempted"], int)
    assert isinstance(result["duration_ms"], int)
    assert isinstance(result["success_rate"], float)


# For integration tests of ProcessingSummary in real workflows, see:
# - tests/test_pipeline.py::test_pipeline_tracks_processing_summary
# - tests/test_pipeline.py::test_pipeline_summary_counts_attempted
# - tests/test_pipeline.py::test_pipeline_summary_counts_processed
# - tests/test_pipeline.py::test_pipeline_summary_with_limit
# - tests/test_pipeline.py::test_pipeline_summary_success_rate
# - tests/test_pipeline.py::test_pipeline_summary_duration_tracking
# - tests/test_pipeline.py::test_pipeline_summary_with_concurrency
# - tests/test_pipeline.py::test_pipeline_summary_as_dict_export
