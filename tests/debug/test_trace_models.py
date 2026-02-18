"""Tests for trace analysis models."""

from buttermilk.debug.trace_models import TraceSummary


class TestTraceSummary:
    """Test TraceSummary model."""

    def test_trace_summary_basic_fields(self):
        """TraceSummary should store basic trace statistics."""
        summary = TraceSummary(
            total_traces=10,
            error_count=2,
            agents=["JUDGE", "SCORER"],
            execution_path=["FETCH", "JUDGE", "SCORER"],
            time_range=None,
            session_id="test-session",
            by_agent={"JUDGE": 5, "SCORER": 5},
            by_error_status={"success": 8, "error": 2},
        )

        assert summary.total_traces == 10
        assert summary.error_count == 2
        assert "JUDGE" in summary.agents
        assert summary.by_error_status["error"] == 2
