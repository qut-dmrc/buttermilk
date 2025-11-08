"""Tests for weave trace formatters."""

import pytest

from buttermilk._core.tracing import EmptyTraceFilter, NoOpFormatter


class TestNoOpFormatter:
    """Tests for NoOpFormatter."""

    def test_format_returns_unchanged(self):
        """Test that NoOpFormatter returns trace data unchanged."""
        formatter = NoOpFormatter()

        # Test with various trace data structures
        test_cases = [
            {"op_name": "test_op", "output": "some output"},
            {"op_name": "message_handler", "output": None},
            {},
            {"complex": {"nested": {"data": [1, 2, 3]}}},
        ]

        for trace_data in test_cases:
            result = formatter.format(trace_data)
            assert result == trace_data
            assert result is trace_data  # Should be the same object


class TestEmptyTraceFilter:
    """Tests for EmptyTraceFilter."""

    @pytest.fixture
    def filter(self):
        """Create an EmptyTraceFilter instance."""
        return EmptyTraceFilter()

    def test_empty_message_handler_filtered(self, filter):
        """Test that empty message handler traces are filtered out."""
        # Test cases that should be filtered
        traces_to_filter = [
            # Message handler with None output
            {
                "op_name": "AutogenAdapter.handle_record.message_handler",
                "output": None,
                "inputs": {"message": {"type": "Record", "content": "test"}},
            },
            # Message handler with empty dict output
            {
                "op_name": "SomeAgent._heartbeat.message_handler",
                "output": {},
                "inputs": {"message": {"type": "HeartBeat"}},
            },
            # Message handler with empty string output
            {
                "op_name": "Agent.handle_manager_message.message_handler",
                "output": "",
                "inputs": {"message": {"type": "UserResponseMessage"}},
            },
        ]

        for trace_data in traces_to_filter:
            result = filter.format(trace_data)
            assert result is None, f"Expected trace to be filtered: {trace_data}"

    def test_meaningful_traces_preserved(self, filter):
        """Test that traces with meaningful output are preserved."""
        # Test cases that should NOT be filtered
        traces_to_keep = [
            # Message handler with actual output
            {
                "op_name": "Agent.process.message_handler",
                "output": {"result": "processed", "status": "success"},
                "inputs": {"message": {"type": "AgentInput"}},
            },
            # Non-message handler trace (even if empty)
            {"op_name": "Agent.process", "output": None, "inputs": {"data": "test"}},
            # Message handler with error
            {
                "op_name": "Agent.handle.message_handler",
                "output": None,
                "error": "Something went wrong",
                "inputs": {"message": {"type": "AgentInput"}},
            },
            # Message handler with exception
            {
                "op_name": "Agent.handle.message_handler",
                "output": None,
                "exception": {"type": "ValueError", "message": "Invalid input"},
                "inputs": {"message": {"type": "AgentInput"}},
            },
            # Message handler with non-empty string output
            {
                "op_name": "Agent.handle.message_handler",
                "output": "Success",
                "inputs": {"message": {"type": "AgentInput"}},
            },
            # Message handler with list output
            {
                "op_name": "Agent.handle.message_handler",
                "output": [],
                "inputs": {"message": {"type": "AgentInput"}},
            },
            # Message handler with numeric output
            {
                "op_name": "Agent.handle.message_handler",
                "output": 0,
                "inputs": {"message": {"type": "AgentInput"}},
            },
            # Message handler with boolean output
            {
                "op_name": "Agent.handle.message_handler",
                "output": False,
                "inputs": {"message": {"type": "AgentInput"}},
            },
        ]

        for trace_data in traces_to_keep:
            result = filter.format(trace_data)
            assert result == trace_data, f"Expected trace to be preserved: {trace_data}"
            assert result is trace_data  # Should be the same object

    def test_edge_cases(self, filter):
        """Test edge cases for the filter."""
        # Missing fields
        assert filter.format({}) == {}
        assert filter.format({"op_name": "test"}) == {"op_name": "test"}
        assert filter.format({"output": None}) == {"output": None}

        # Op name variations - "message_handler" without dot prefix is NOT filtered
        # because we specifically look for ".message_handler" suffix
        assert filter.format(
            {
                "op_name": "message_handler",  # Just "message_handler" without prefix
                "output": None,
            }
        ) == {"op_name": "message_handler", "output": None}

        assert filter.format(
            {
                "op_name": "something.message_handler.extra",  # Extra suffix
                "output": None,
            }
        ) == {"op_name": "something.message_handler.extra", "output": None}

        # Case sensitivity
        assert filter.format(
            {
                "op_name": "Agent.MESSAGE_HANDLER",  # Uppercase
                "output": None,
            }
        ) == {"op_name": "Agent.MESSAGE_HANDLER", "output": None}
