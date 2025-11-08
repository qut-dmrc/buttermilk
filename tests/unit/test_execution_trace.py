"""Unit tests for ExecutionTrace."""

from unittest.mock import AsyncMock, patch

import pytest
from autogen_core.models import SystemMessage, UserMessage

from buttermilk._core.contract import AgentOutput, ExecutionTrace
from buttermilk.utils.trace_writer import TraceWriter, get_trace_writer


class TestExecutionTrace:
    """Test suite for ExecutionTrace functionality."""

    def test_execution_trace_creation(self, real_bm):
        """Test basic ExecutionTrace creation."""
        trace = ExecutionTrace(
            agent_info={
                "component_name": "TestComponent",
                "execution_type": "processor",
                "config": {"model": "gpt-4"},
            }
        )

        assert trace.call_id is not None
        assert trace.timestamp is not None
        assert trace.agent_info["component_name"] == "TestComponent"
        assert trace.agent_info["execution_type"] == "processor"

    def test_execution_trace_with_messages(self, real_bm):
        """Test ExecutionTrace with LLM messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant", source="test"),
            UserMessage(content="Hello", source="test"),
        ]

        trace = ExecutionTrace(
            agent_info={"component_name": "LLMAgent"},
            messages=messages,
            metadata={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "template_name": "test_template",
            },
        )

        assert len(trace.messages) == 2
        assert trace.metadata["token_usage"]["prompt_tokens"] == 10
        assert trace.metadata["template_name"] == "test_template"

    def test_execution_trace_with_error(self, real_bm):
        """Test ExecutionTrace error handling."""
        trace = ExecutionTrace(
            agent_info={"component_name": "ErrorComponent"},
            error={
                "event": "Processing failed",
                "details": {"error_type": "ValueError"},
            },
        )

        assert trace.is_error is True
        assert trace.error["event"] == "Processing failed"

    def test_execution_trace_no_error(self, real_bm):
        """Test ExecutionTrace without error."""
        trace = ExecutionTrace(agent_info={"component_name": "SuccessComponent"})

        assert trace.is_error is False

    def test_execution_trace_from_output(self, real_bm):
        """Test creating ExecutionTrace from AgentOutput."""
        output = AgentOutput(
            agent_id="test_agent",
            outputs="Test result",
            messages=[UserMessage(content="Test", source="test")],
            metadata={"test_key": "test_value"},
        )

        trace = ExecutionTrace.from_output(
            output,
            agent_info={"component_name": "TestAgent", "execution_type": "agent"},
            inputs={"input": "test"},
            parameters={"param": "value"},
        )

        assert trace.outputs == "Test result"
        assert trace.inputs == {"input": "test"}
        assert trace.parameters == {"param": "value"}
        assert len(trace.messages) == 1
        assert trace.metadata["test_key"] == "test_value"

    def test_execution_trace_with_record(self, real_bm):
        """Test ExecutionTrace with record context."""
        trace = ExecutionTrace(
            agent_info={"component_name": "RecordProcessor"},
            record={
                "record_id": "rec_123",
                "dataset_name": "test_dataset",
                "split_type": "train",
            },
        )

        assert trace.record["record_id"] == "rec_123"
        assert trace.record["dataset_name"] == "test_dataset"
        assert trace.record["split_type"] == "train"

    def test_execution_trace_model_dump(self, real_bm):
        """Test ExecutionTrace serialization for BigQuery."""
        trace = ExecutionTrace(
            agent_info={"component_name": "TestComponent"},
            inputs={"test": "input"},
            outputs="test output",
            metadata={"key": "value"},
        )

        dumped = trace.model_dump()

        assert "timestamp" in dumped
        assert "call_id" in dumped
        assert "session_id" in dumped
        assert dumped["agent_info"]["component_name"] == "TestComponent"
        assert dumped["inputs"] == {"test": "input"}
        assert dumped["outputs"] == "test output"


class TestTraceWriter:
    """Test suite for TraceWriter functionality."""

    @pytest.mark.anyio
    async def test_trace_writer_singleton(self, real_bm):
        """Test TraceWriter is a singleton."""
        writer1 = TraceWriter()
        writer2 = TraceWriter()
        writer3 = get_trace_writer()

        assert writer1 is writer2
        assert writer2 is writer3

    @pytest.mark.anyio
    async def test_trace_writer_add(self, real_bm):
        """Test adding traces to TraceWriter."""
        with patch(
            "buttermilk.utils.trace_writer.AsyncDataUploader"
        ) as mock_uploader_class:
            mock_uploader = AsyncMock()
            mock_uploader_class.return_value = mock_uploader

            # Reset the singleton
            TraceWriter._instance = None
            TraceWriter._initialized = False

            writer = TraceWriter()
            writer.uploader = mock_uploader

            trace = ExecutionTrace(agent_info={"component_name": "TestComponent"})

            await writer.add(trace)

            mock_uploader.add.assert_called_once_with(trace)

    @pytest.mark.anyio
    async def test_trace_writer_no_config(self, real_bm):
        """Test TraceWriter handles missing configuration gracefully."""
        # Reset the singleton
        TraceWriter._instance = None
        TraceWriter._initialized = False

        with patch("buttermilk.utils.trace_writer.bm") as mock_bm:
            mock_bm.cfg = type("obj", (object,), {})()  # No storage config

            writer = TraceWriter()

            # Should not raise error when adding trace without config
            trace = ExecutionTrace(agent_info={"component_name": "TestComponent"})
            await writer.add(trace)  # Should handle gracefully

            # After initialization, uploader should be None
            assert writer.uploader is None
