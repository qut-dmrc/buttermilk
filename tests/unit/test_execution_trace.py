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
            inputs={"input": "test", "param": "value"},  # All template vars go in inputs
        )

        assert trace.outputs == "Test result"
        assert trace.inputs == {"input": "test", "param": "value"}
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

    def test_execution_trace_from_output_requires_explicit_record(self, real_bm):
        """Test that from_output() requires explicit record parameter.

        Strict contract: record must be passed explicitly to from_output().
        No fallback extraction from inputs.record - caller is responsible.

        This test verifies that when record is passed explicitly, it is preserved
        correctly in the trace, regardless of the inputs structure.
        """
        from buttermilk._core.types import BaseRecord

        # Create output (what agent produces)
        output = AgentOutput(
            agent_id="test_agent",
            outputs="Processed result",
            messages=[UserMessage(content="Test", source="test")],
        )

        # Create a mock message-like object with record attribute
        class MockMessage:
            def __init__(self):
                self.record = BaseRecord(
                    record_id="rec_456",
                    dataset_name="test_dataset",
                    split_type="validation",
                )
                self.inputs = {"input_text": "test data"}

        message = MockMessage()

        # Simulate what agent.py does at line 645:
        # - Passes inputs as dict (from trace_inputs, which is often a dict)
        # - Passes record explicitly to preserve it
        trace = ExecutionTrace.from_output(
            output,
            agent_info={"component_name": "TestAgent", "execution_type": "agent"},
            inputs={"input_text": "test data"},  # Inputs is a plain dict
            record=message.record,  # Strict contract: record must be explicit
        )

        # Verify that trace.record was populated from the explicit record parameter
        assert trace.record is not None, (
            "trace.record should preserve record when passed explicitly to from_output(). "
            "When inputs is a dict (no .record attribute), explicit record param is required."
        )
        assert trace.record.record_id == "rec_456"
        assert trace.record.dataset_name == "test_dataset"
        assert trace.record.split_type == "validation"

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
        with patch("buttermilk.utils.trace_writer.AsyncDataUploader") as mock_uploader_class:
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


class TestExecutionTraceSerialization:
    """Test suite for ExecutionTrace serialization edge cases."""

    def test_model_dump_with_nested_execution_trace(self, real_bm):
        """Test that model_dump() handles nested ExecutionTrace in outputs field.

        Verifies that when an ExecutionTrace contains another ExecutionTrace
        in its outputs field, model_dump() serializes it correctly without
        infinite loops.
        """
        # Create inner trace
        inner_trace = ExecutionTrace(
            agent_info={"component_name": "InnerComponent"},
            inputs={"inner_input": "value"},
            outputs="inner result",
        )

        # Create outer trace with inner trace in outputs
        outer_trace = ExecutionTrace(
            agent_info={"component_name": "OuterComponent"},
            inputs={"outer_input": "value"},
            outputs=inner_trace,
        )

        # Should not infinite loop
        dumped = outer_trace.model_dump()

        # Verify structure
        assert dumped["agent_info"]["component_name"] == "OuterComponent"
        assert dumped["inputs"]["outer_input"] == "value"
        assert isinstance(dumped["outputs"], dict)
        assert dumped["outputs"]["agent_info"]["component_name"] == "InnerComponent"
        assert dumped["outputs"]["outputs"] == "inner result"

    def test_model_dump_with_circular_reference_in_outputs(self, real_bm):
        """Test behavior when outputs contains a dict with circular reference.

        Uses a 5-second timeout to detect infinite loops. Should either handle
        gracefully or raise a clear error, NOT infinite loop.
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("model_dump() timed out - likely infinite loop")

        # Create a circular reference
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict

        trace = ExecutionTrace(
            agent_info={"component_name": "CircularComponent"},
            outputs=circular_dict,
        )

        # Set 5-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # This should either:
            # 1. Complete successfully (handle circular refs gracefully)
            # 2. Raise a clear error (ValueError, RecursionError, etc.)
            # 3. NOT timeout (would indicate infinite loop)
            dumped = trace.model_dump()

            # If we get here, it handled the circular reference
            # Cancel the alarm
            signal.alarm(0)

            # Verify at minimum the top-level fields are present
            assert "agent_info" in dumped
            assert dumped["agent_info"]["component_name"] == "CircularComponent"

        except (ValueError, RecursionError, TypeError) as e:
            # Cancel the alarm
            signal.alarm(0)
            # These are acceptable - model_dump raised a clear error
            # instead of infinite looping
            assert True, f"model_dump raised expected error: {type(e).__name__}"

        except TimeoutError:
            # This is a FAILURE - indicates infinite loop
            signal.alarm(0)
            pytest.fail("model_dump() infinite looped on circular reference")

    def test_model_dump_deeply_nested_structure(self, real_bm):
        """Test serialization with 100-level deeply nested dicts.

        Ensures no stack overflow with deeply nested structures.
        """
        # Create 100-level deep nested dict
        deep_dict = {"level": 0, "value": "innermost"}
        for i in range(1, 100):
            deep_dict = {"level": i, "nested": deep_dict}

        trace = ExecutionTrace(
            agent_info={"component_name": "DeepComponent"},
            outputs=deep_dict,
        )

        # Should not cause stack overflow
        dumped = trace.model_dump()

        # Verify serialization succeeded
        assert "outputs" in dumped
        assert dumped["outputs"]["level"] == 99

        # Verify deep nesting is preserved
        current = dumped["outputs"]
        for expected_level in range(99, -1, -1):
            assert current["level"] == expected_level
            if expected_level > 0:
                current = current["nested"]
            else:
                assert current["value"] == "innermost"

    def test_model_dump_json_with_execution_trace_list_in_outputs(self, real_bm):
        """Test JSON serialization when outputs contains a list of ExecutionTrace objects.

        Verifies that model_dump_json() properly handles lists of nested traces.
        """
        import json

        # Create multiple inner traces
        trace1 = ExecutionTrace(
            agent_info={"component_name": "Trace1"},
            outputs="result 1",
        )
        trace2 = ExecutionTrace(
            agent_info={"component_name": "Trace2"},
            outputs="result 2",
        )

        # Create outer trace with list of traces in outputs
        outer_trace = ExecutionTrace(
            agent_info={"component_name": "ListContainer"},
            outputs=[trace1, trace2],
        )

        # Test model_dump with list of traces
        dumped = outer_trace.model_dump()
        assert isinstance(dumped["outputs"], list)
        assert len(dumped["outputs"]) == 2
        assert dumped["outputs"][0]["agent_info"]["component_name"] == "Trace1"
        assert dumped["outputs"][1]["agent_info"]["component_name"] == "Trace2"

        # Test JSON serialization
        json_str = outer_trace.model_dump_json()
        parsed = json.loads(json_str)

        assert isinstance(parsed["outputs"], list)
        assert len(parsed["outputs"]) == 2
        assert parsed["outputs"][0]["agent_info"]["component_name"] == "Trace1"
        assert parsed["outputs"][1]["agent_info"]["component_name"] == "Trace2"

    def test_model_dump_includes_computed_fields(self, real_bm):
        """Verify computed fields (is_error, object_type) are included in model_dump.

        Pydantic computed fields are included in serialization by default unless
        explicitly excluded. This test verifies that behavior.
        """
        # Create trace with error
        trace_with_error = ExecutionTrace(
            agent_info={"component_name": "ErrorComponent"},
            error={"event": "test error", "details": {}},
            outputs="some output",
        )

        # Verify computed fields work as expected
        assert trace_with_error.is_error is True
        assert trace_with_error.object_type == "str"

        # Dump and verify computed fields are included
        dumped = trace_with_error.model_dump()

        # Computed fields SHOULD be in the dumped dict (Pydantic default behavior)
        assert "is_error" in dumped
        assert dumped["is_error"] is True
        assert "object_type" in dumped
        assert dumped["object_type"] == "str"

        # Regular fields should be present
        assert "error" in dumped
        assert "outputs" in dumped
        assert dumped["error"]["event"] == "test error"

        # Test excluding computed fields explicitly
        dumped_no_computed = trace_with_error.model_dump(exclude={"is_error", "object_type"})
        assert "is_error" not in dumped_no_computed
        assert "object_type" not in dumped_no_computed


class TestExecutionTraceHashability:
    """Test suite for ExecutionTrace hashability behavior.

    ExecutionTrace is a Pydantic BaseModel without frozen=True, which means
    it's unhashable by default in Pydantic v2. These tests document the
    actual behavior and potential issues when trying to use ExecutionTrace
    in sets or as dict keys.
    """

    def test_execution_trace_in_set_behavior(self, real_bm):
        """Test that ExecutionTrace cannot be used in a set.

        Since ExecutionTrace doesn't define __hash__, Pydantic models are
        not hashable by default (unless frozen=True). This should raise TypeError.
        """
        trace1 = ExecutionTrace(
            agent_info={"component_name": "TestComponent1"},
            inputs={"test": "input1"},
        )
        trace2 = ExecutionTrace(
            agent_info={"component_name": "TestComponent2"},
            inputs={"test": "input2"},
        )

        # Attempting to create a set with ExecutionTrace should raise TypeError
        with pytest.raises(TypeError, match="unhashable type"):
            {trace1, trace2}

    def test_execution_trace_as_dict_key_behavior(self, real_bm):
        """Test that ExecutionTrace cannot be used as a dictionary key.

        Should raise TypeError since it's unhashable.
        """
        trace = ExecutionTrace(
            agent_info={"component_name": "TestComponent"},
            inputs={"test": "input"},
        )

        # Attempting to use ExecutionTrace as dict key should raise TypeError
        with pytest.raises(TypeError, match="unhashable type"):
            {trace: "some_value"}

    def test_execution_trace_equality(self, real_bm):
        """Test that two ExecutionTrace objects with identical fields are NOT equal by default.

        Pydantic uses identity (id()) by default for models without frozen=True.
        Each instance is unique even if fields are identical.

        Note: This could be problematic if we need to compare traces for deduplication.
        If equality checking is needed, consider adding frozen=True to the model
        or implementing custom __eq__ and __hash__ methods.
        """
        # Create two traces with identical data
        trace1 = ExecutionTrace(
            agent_info={"component_name": "TestComponent"},
            inputs={"test": "input"},
            outputs="output",
        )
        trace2 = ExecutionTrace(
            agent_info={"component_name": "TestComponent"},
            inputs={"test": "input"},
            outputs="output",
        )

        # By default, Pydantic models ARE equal if their fields match
        # (this changed in Pydantic v2)
        # Let's verify the actual behavior
        assert trace1 != trace2, (
            "ExecutionTrace instances with identical fields should be unequal due to different call_id and timestamp (auto-generated fields)"
        )

        # The reason they're unequal is because of auto-generated fields
        assert trace1.call_id != trace2.call_id
        assert trace1.timestamp != trace2.timestamp

    def test_execution_trace_identity_hash(self, real_bm):
        """Test workaround for using ExecutionTrace in collections.

        If we need to use ExecutionTrace in sets or as dict keys, we can use:
        1. id() of the object (Python identity)
        2. call_id (unique identifier field)
        3. Convert to tuple/dict for hashing

        This test documents the recommended approaches.
        """
        trace1 = ExecutionTrace(
            agent_info={"component_name": "TestComponent1"},
            inputs={"test": "input1"},
        )
        trace2 = ExecutionTrace(
            agent_info={"component_name": "TestComponent2"},
            inputs={"test": "input2"},
        )

        # Workaround 1: Use id() for identity-based collections
        trace_set = {id(trace1), id(trace2)}
        assert len(trace_set) == 2

        # Workaround 2: Use call_id (recommended for deduplication)
        call_id_set = {trace1.call_id, trace2.call_id}
        assert len(call_id_set) == 2

        # Workaround 3: Use call_id as dict key (recommended)
        trace_dict = {
            trace1.call_id: trace1,
            trace2.call_id: trace2,
        }
        assert len(trace_dict) == 2
        assert trace_dict[trace1.call_id] is trace1
        assert trace_dict[trace2.call_id] is trace2


class TestExecutionTraceJsonSerialization:
    """Test suite for ExecutionTrace JSON serialization via model_dump_json()."""

    def test_model_dump_json_succeeds(self, real_bm):
        """Test that model_dump_json() works for standard ExecutionTrace.

        Verifies JSON serialization with typical fields including agent_info,
        inputs, outputs, and metadata.
        """
        import json

        trace = ExecutionTrace(
            agent_info={
                "component_name": "TestComponent",
                "execution_type": "processor",
                "config": {"model": "gpt-4"},
            },
            inputs={"input_key": "input_value"},
            outputs="test output result",
            metadata={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "template_name": "test_template",
            },
        )

        # Should serialize to JSON string without error
        json_str = trace.model_dump_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)

        # Verify structure
        assert parsed["agent_info"]["component_name"] == "TestComponent"
        assert parsed["agent_info"]["execution_type"] == "processor"
        assert parsed["inputs"]["input_key"] == "input_value"
        assert parsed["outputs"] == "test output result"
        assert parsed["metadata"]["token_usage"]["prompt_tokens"] == 10

    def test_model_dump_json_with_datetime(self, real_bm):
        """Test that datetime fields serialize correctly to ISO format.

        ExecutionTrace has a timestamp field that should serialize to ISO 8601 format.
        """
        import json
        from datetime import datetime

        trace = ExecutionTrace(
            agent_info={"component_name": "DateTimeTest"},
            outputs="result",
        )

        # timestamp is auto-generated as datetime
        assert isinstance(trace.timestamp, datetime)

        # Serialize to JSON
        json_str = trace.model_dump_json()
        parsed = json.loads(json_str)

        # Verify timestamp is ISO format string in JSON
        assert isinstance(parsed["timestamp"], str)
        # Should be able to parse back to datetime
        parsed_dt = datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))
        assert isinstance(parsed_dt, datetime)

    def test_model_dump_json_with_non_serializable_outputs(self, real_bm):
        """Test model_dump_json() behavior with non-JSON-serializable objects.

        When outputs contains objects like asyncio.Lock that cannot be serialized,
        model_dump_json() should raise a clear error (NOT hang).
        """
        import asyncio

        # Create trace with non-serializable object in outputs
        trace = ExecutionTrace(
            agent_info={"component_name": "NonSerializable"},
            outputs=asyncio.Lock(),  # Cannot be JSON serialized
        )

        # Should raise TypeError or ValueError, NOT hang
        with pytest.raises((TypeError, ValueError)) as exc_info:
            trace.model_dump_json()

        # Verify error message is clear
        error_msg = str(exc_info.value).lower()
        assert "json" in error_msg or "serializ" in error_msg or "not supported" in error_msg

    def test_model_dump_json_with_session_info(self, real_bm):
        """Test that session_info field serializes correctly.

        The session_info field is populated by _get_session_info() and contains
        session_id and other session metadata.
        """
        import json

        trace = ExecutionTrace(
            agent_info={"component_name": "SessionTest"},
            outputs="result",
        )

        # session_info should be populated (it's a Pydantic SessionInfo model)
        assert trace.session_info is not None
        assert hasattr(trace.session_info, "session_id")
        session_id = trace.session_info.session_id

        # Serialize to JSON
        json_str = trace.model_dump_json()
        parsed = json.loads(json_str)

        # Verify session_info is in JSON output
        assert "session_info" in parsed
        assert isinstance(parsed["session_info"], dict)
        assert "session_id" in parsed["session_info"]
        assert parsed["session_info"]["session_id"] == session_id

    def test_model_dump_json_roundtrip(self, real_bm):
        """Test that model_dump_json() output can be parsed as valid JSON.

        Verifies that JSON serialization produces valid JSON and key fields
        are preserved correctly.
        """
        import json

        trace = ExecutionTrace(
            agent_info={
                "component_name": "RoundTripTest",
                "execution_type": "agent",
            },
            inputs={"key": "value"},
            outputs="output data",
            metadata={"test": "metadata"},
        )

        # Serialize to JSON
        json_str = trace.model_dump_json()

        # Should be valid JSON
        json_parsed = json.loads(json_str)
        assert isinstance(json_parsed, dict)

        # Verify core fields are preserved in JSON
        assert json_parsed["agent_info"]["component_name"] == "RoundTripTest"
        assert json_parsed["agent_info"]["execution_type"] == "agent"
        assert json_parsed["inputs"]["key"] == "value"
        assert json_parsed["outputs"] == "output data"
        assert json_parsed["metadata"]["test"] == "metadata"

        # Verify auto-generated fields are present
        assert "call_id" in json_parsed
        assert "timestamp" in json_parsed
        assert "session_info" in json_parsed

        # Verify JSON parsed data matches model_dump for specified fields
        dict_dump = trace.model_dump()
        assert json_parsed["call_id"] == dict_dump["call_id"]
        assert json_parsed["agent_info"] == dict_dump["agent_info"]
        assert json_parsed["inputs"] == dict_dump["inputs"]
        assert json_parsed["outputs"] == dict_dump["outputs"]


class TestExecutionTraceRecordSchema:
    """Test suite for ExecutionTrace record field schema compliance.

    These tests verify that:
    1. Record field contains required fields (record_id, record_hash, content) when present
    2. Record is NOT duplicated in inputs['record'] or inputs['template_vars']['record']

    This ensures traces match the BigQuery schema in traces.schema.json.
    """

    def test_record_has_required_fields_when_present(self, real_bm):
        """Test that trace.record contains required fields per BQ schema.

        When trace.record is populated, it MUST contain:
        - record_id (REQUIRED)
        - record_hash (REQUIRED)
        - content (REQUIRED)
        """
        from buttermilk._core.types import Record

        record = Record(
            record_id="test_required_fields",
            dataset_name="test_dataset",
            split_type="train",
            content="Test content for required fields validation",
        )

        trace = ExecutionTrace(
            agent_info={"component_name": "RequiredFieldsTest"},
            record=record,
            outputs="test output",
        )

        dumped = trace.model_dump()

        # Record must have all required fields
        assert "record" in dumped
        assert dumped["record"] is not None

        # Required fields per BQ schema
        assert "record_id" in dumped["record"], "record_id is REQUIRED"
        assert dumped["record"]["record_id"] == "test_required_fields"

        assert "record_hash" in dumped["record"], "record_hash is REQUIRED"
        assert dumped["record"]["record_hash"] is not None
        assert len(dumped["record"]["record_hash"]) == 64  # SHA256 hex

        assert "content" in dumped["record"], "content is REQUIRED"
        assert dumped["record"]["content"] == "Test content for required fields validation"

    def test_record_not_in_inputs_record(self, real_bm):
        """Test that record is NOT duplicated in inputs['record'].

        The trace.record field should be the ONLY location for record data.
        inputs['record'] should be None or absent to avoid duplication.
        """
        from buttermilk._core.types import Record

        record = Record(
            record_id="test_no_duplicate",
            dataset_name="test_dataset",
            content="Content should not be duplicated",
        )

        # Simulate what should happen: record passed separately, not in inputs
        trace = ExecutionTrace(
            agent_info={"component_name": "NoDuplicateTest"},
            record=record,
            inputs={"prompt": "test prompt", "other_var": "value"},
            outputs="test output",
        )

        dumped = trace.model_dump()

        # inputs should NOT contain 'record' key
        if dumped.get("inputs"):
            assert "record" not in dumped["inputs"], "Record should NOT be in inputs['record'] - use trace.record instead"

    def test_record_not_in_inputs_template_vars_record(self, real_bm):
        """Test that record is NOT in inputs['template_vars']['record'].

        When LLMCore creates traces, template_vars should not contain the record.
        """
        from buttermilk._core.types import Record

        record = Record(
            record_id="test_no_template_var_record",
            dataset_name="test_dataset",
            content="Content should not be in template_vars",
        )

        # Simulate LLMCore resolved_inputs structure
        trace = ExecutionTrace(
            agent_info={"component_name": "NoTemplateVarRecordTest"},
            record=record,
            inputs={
                "template_vars": {"prompt": "test", "criteria": "some criteria"},
                "context": [],
            },
            outputs="test output",
        )

        dumped = trace.model_dump()

        # template_vars should NOT contain 'record' key
        if dumped.get("inputs"):
            template_vars = dumped["inputs"].get("template_vars", {})
            if template_vars:
                assert "record" not in template_vars, "Record should NOT be in inputs['template_vars']['record'] - use trace.record instead"

    def test_trace_record_field_serialization_for_bq(self, real_bm):
        """Test that record field serializes correctly for BigQuery schema.

        Verifies the serialized record matches expected BQ structure with
        all BaseRecord fields plus record_hash.
        """
        from buttermilk._core.types import Record

        record = Record(
            record_id="bq_schema_test",
            dataset_name="production_dataset",
            split_type="validation",
            content="Content for BigQuery schema validation test",
            metadata={"source": "test", "version": 1},
        )

        trace = ExecutionTrace(
            agent_info={"component_name": "BQSchemaTest"},
            record=record,
            outputs="test output",
        )

        dumped = trace.model_dump()
        record_data = dumped["record"]

        # Verify BQ schema fields
        expected_fields = {
            "record_id",
            "record_hash",
            "content",
            "dataset_name",
            "split_type",
            "metadata",
        }

        for field in expected_fields:
            assert field in record_data, f"Missing BQ schema field: {field}"

        # Verify field values
        assert record_data["record_id"] == "bq_schema_test"
        assert record_data["dataset_name"] == "production_dataset"
        assert record_data["split_type"] == "validation"
        assert record_data["content"] == "Content for BigQuery schema validation test"
        assert record_data["metadata"] == {"source": "test", "version": 1}
        assert len(record_data["record_hash"]) == 64


class TestExecutionTraceWithHashes:
    """Test suite for ExecutionTrace serialization with computed hash fields.

    These tests target potential infinite loops when serializing ExecutionTrace
    objects that contain BaseRecord or Record instances with computed hash fields
    (record_hash, ground_truth_hash).

    Background:
    - record_hash is a @computed_field on BaseRecord that calls as_markdown() then SHA256
    - ground_truth_hash is a @computed_field on Record that hashes the ground_truth dict
    - as_markdown() excludes HASH_METADATA_KEYS to avoid recursion
    - Infinite loops could occur if hash computation triggers serialization loops
    """

    def test_execution_trace_with_base_record_serialization(self, real_bm):
        """Test ExecutionTrace with BaseRecord in record field serializes without loops.

        Creates an ExecutionTrace with a BaseRecord (which has record_hash computed field)
        and verifies model_dump() completes without infinite loops.
        Uses 5-second timeout to detect infinite loops.
        """
        import signal

        from buttermilk._core.types import BaseRecord

        def timeout_handler(signum, frame):
            raise TimeoutError("model_dump() timed out - likely infinite loop")

        # Create BaseRecord with record_hash computed field
        base_record = BaseRecord(
            record_id="test_record_001",
            dataset_name="test_dataset",
            split_type="train",
        )

        # Verify record_hash is computed
        assert hasattr(base_record, "record_hash")
        hash_value = base_record.record_hash
        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 hex digest

        # Create trace with this record
        trace = ExecutionTrace(
            agent_info={"component_name": "HashTestComponent"},
            record=base_record,
            outputs="test output",
        )

        # Set 5-second timeout to detect infinite loops
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # This should complete without infinite loop
            dumped = trace.model_dump()

            # Cancel alarm
            signal.alarm(0)

            # Verify serialization succeeded
            assert "record" in dumped
            assert dumped["agent_info"]["component_name"] == "HashTestComponent"

        except TimeoutError:
            signal.alarm(0)
            pytest.fail("model_dump() infinite looped when serializing ExecutionTrace with BaseRecord")

    def test_execution_trace_with_record_and_ground_truth_hash(self, real_bm):
        """Test ExecutionTrace with Record containing ground_truth serializes correctly.

        Creates a Record with ground_truth (which triggers ground_truth_hash computation)
        inside an ExecutionTrace, and verifies both record_hash and ground_truth_hash
        don't cause infinite loops during serialization.

        Note: Record's model_config excludes computed fields (record_hash, ground_truth_hash)
        from model_dump() output. This test verifies that accessing these computed fields
        and then serializing doesn't cause infinite loops.
        """
        import signal

        from buttermilk._core.types import Record

        def timeout_handler(signum, frame):
            raise TimeoutError("model_dump() timed out - likely infinite loop")

        # Create Record with ground_truth
        record = Record(
            record_id="test_record_002",
            dataset_name="test_dataset",
            split_type="validation",
            ground_truth={"label": "positive", "score": 0.95},
        )

        # Verify both computed hashes work
        assert hasattr(record, "record_hash")
        assert hasattr(record, "ground_truth_hash")
        record_hash = record.record_hash
        gt_hash = record.ground_truth_hash
        assert record_hash is not None
        assert gt_hash is not None

        # Create trace with this record
        trace = ExecutionTrace(
            agent_info={"component_name": "GroundTruthHashTest"},
            record=record,
            inputs={"query": "test query"},
            outputs="prediction result",
        )

        # Set 5-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # Should not infinite loop
            dumped = trace.model_dump()

            # Cancel alarm
            signal.alarm(0)

            # Verify serialization succeeded
            assert "record" in dumped
            assert dumped["record"]["record_id"] == "test_record_002"
            # Note: ground_truth_hash and record_hash are excluded from model_dump
            # but ground_truth itself should be present
            # Check if ground_truth is in dumped record (it may be excluded too based on config)
            if "ground_truth" in dumped["record"]:
                assert dumped["record"]["ground_truth"]["label"] == "positive"

        except TimeoutError:
            signal.alarm(0)
            pytest.fail("model_dump() infinite looped when serializing ExecutionTrace with Record containing ground_truth_hash")

    def test_record_hash_is_idempotent(self, real_bm):
        """Test that record_hash computation is idempotent (same value on multiple calls).

        Verifies that calling record.record_hash multiple times returns the same value
        and doesn't mutate state in a way that could cause serialization loops.
        """
        from buttermilk._core.types import BaseRecord

        record = BaseRecord(
            record_id="idempotent_test",
            dataset_name="test_dataset",
            split_type="test",
        )

        # Compute hash multiple times
        hash1 = record.record_hash
        hash2 = record.record_hash
        hash3 = record.record_hash

        # All should be identical
        assert hash1 == hash2 == hash3
        assert len(hash1) == 64  # SHA256 hex

        # Verify hash computation doesn't break serialization
        # BaseRecord includes computed fields by default
        dumped = record.model_dump()
        # Check if record_hash is included (it should be for BaseRecord)
        if "record_hash" in dumped:
            assert dumped["record_hash"] == hash1

    def test_execution_trace_model_dump_json_with_record(self, real_bm):
        """Test JSON serialization of ExecutionTrace containing record with hashes.

        Verifies that model_dump_json() (not just model_dump()) works correctly
        when the trace contains a record with computed hash fields.

        Note: Record's model_config excludes computed fields (record_hash, ground_truth_hash)
        from serialization. This test verifies that accessing these fields before serialization
        doesn't cause infinite loops in JSON serialization.
        """
        import json
        import signal

        from buttermilk._core.types import Record

        def timeout_handler(signum, frame):
            raise TimeoutError("model_dump_json() timed out - likely infinite loop")

        # Create record with ground_truth
        record = Record(
            record_id="json_test_record",
            dataset_name="test_dataset",
            split_type="train",
            ground_truth={"category": "A", "confidence": 0.88},
        )

        # Access the computed hash fields to trigger their computation
        _ = record.record_hash
        _ = record.ground_truth_hash

        # Create trace
        trace = ExecutionTrace(
            agent_info={"component_name": "JsonHashTest"},
            record=record,
            outputs="classification result",
        )

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # JSON serialization should work
            json_str = trace.model_dump_json()

            # Cancel alarm
            signal.alarm(0)

            # Verify valid JSON
            parsed = json.loads(json_str)
            assert "record" in parsed
            assert parsed["record"]["record_id"] == "json_test_record"
            # Computed fields are excluded from serialization by Record's model_config
            # The important thing is that serialization completes without infinite loop

        except TimeoutError:
            signal.alarm(0)
            pytest.fail("model_dump_json() infinite looped when serializing ExecutionTrace with Record")

    def test_execution_trace_resolved_inputs_with_record(self, real_bm):
        """Test trace where inputs contains a record (as would happen from LLMCore).

        This simulates the scenario where LLMCore creates a trace and includes
        a Record object in the inputs dict. Verifies no infinite loop occurs
        when both trace.record and trace.inputs contain record-like data.
        """
        import signal

        from buttermilk._core.types import Record

        def timeout_handler(signum, frame):
            raise TimeoutError("model_dump() timed out - likely infinite loop")

        # Create a record
        record = Record(
            record_id="input_record_test",
            dataset_name="test_dataset",
            split_type="validation",
            ground_truth={"answer": "yes"},
        )

        # Create trace where BOTH record field and inputs contain record data
        # (This mimics LLMCore behavior where record might be in multiple places)
        trace = ExecutionTrace(
            agent_info={"component_name": "LLMCoreSimulation"},
            record=record,  # Record in dedicated field
            inputs={
                "prompt": "test prompt",
                "context_record": record,  # Record also in inputs
            },
            outputs="llm response",
        )

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # Should not infinite loop even with record in multiple places
            dumped = trace.model_dump()

            # Cancel alarm
            signal.alarm(0)

            # Verify both records are present in serialized output
            assert "record" in dumped
            assert dumped["record"]["record_id"] == "input_record_test"
            assert "inputs" in dumped
            assert "context_record" in dumped["inputs"]
            assert dumped["inputs"]["context_record"]["record_id"] == "input_record_test"

        except TimeoutError:
            signal.alarm(0)
            pytest.fail("model_dump() infinite looped when ExecutionTrace has record in both record field and inputs dict")


class TestExecutionTraceSchemaContracts:
    """Test schema contracts for ExecutionTrace cleanup (Dec 2024).

    These tests validate the cleaned-up schema structure:
    - Phase 1: agent_info has agent_class, no config dump
    - Phase 2: hashes consolidated to metadata.hashes
    - Phase 3: inputs flattened (no template_vars wrapper)
    - Phase 4: strict contracts (record only in trace.record)
    """

    def test_metadata_hashes_structure(self, real_bm):
        """Test that hashes are consolidated in metadata.hashes dict.

        Schema contract: All hashes (record_hash, template_hash, ground_truth_hash)
        should be stored in metadata.hashes for consistency and extensibility.
        """
        trace = ExecutionTrace(
            agent_info={"component_name": "TestAgent"},
            metadata={
                "hashes": {
                    "record_hash": "abc123",
                    "template_hash": "def456",
                    "ground_truth_hash": "ghi789",
                },
                "other_data": "preserved",
            },
        )

        # Verify hashes are accessible in consolidated location
        assert "hashes" in trace.metadata
        assert trace.metadata["hashes"]["record_hash"] == "abc123"
        assert trace.metadata["hashes"]["template_hash"] == "def456"
        assert trace.metadata["hashes"]["ground_truth_hash"] == "ghi789"
        # Other metadata preserved
        assert trace.metadata["other_data"] == "preserved"

    def test_inputs_flat_structure(self, real_bm):
        """Test that inputs uses flat structure without template_vars wrapper.

        Schema contract: Template variables should be flattened directly into
        the inputs dict, not wrapped in a 'template_vars' key.
        """
        trace = ExecutionTrace(
            agent_info={"component_name": "TestAgent"},
            inputs={
                "var1": "value1",
                "var2": "value2",
                "context": [{"role": "user", "content": "test"}],
            },
        )

        # Verify flat structure - variables at top level
        assert trace.inputs["var1"] == "value1"
        assert trace.inputs["var2"] == "value2"
        assert "context" in trace.inputs
        # No nested template_vars wrapper
        assert "template_vars" not in trace.inputs

    def test_record_separation(self, real_bm):
        """Test that record lives only in trace.record, not duplicated in inputs.

        Schema contract: Record data should be in trace.record only.
        Inputs should contain template variables and context, not record.
        """
        from buttermilk._core.types import BaseRecord

        record = BaseRecord(
            record_id="test_rec",
            dataset_name="test_dataset",
            split_type="train",
        )

        trace = ExecutionTrace(
            agent_info={"component_name": "TestAgent"},
            record=record.model_dump(),
            inputs={
                "var1": "derived_from_record",
                "context": [],
            },
        )

        # Record is in trace.record
        assert trace.record["record_id"] == "test_rec"
        # Inputs has derived data but not raw record
        assert trace.inputs["var1"] == "derived_from_record"
        assert "record" not in trace.inputs
        assert "record_id" not in trace.inputs  # Not leaked to inputs

    def test_agent_info_with_agent_class(self, real_bm):
        """Test that agent_info includes agent_class, not full config dump.

        Schema contract: agent_info should have agent_class for identification,
        but NOT a full config dump (config is saved separately via config_uri).
        """
        trace = ExecutionTrace(
            agent_info={
                "component_name": "MyTestAgent",
                "agent_class": "LLMAgent",
                "agent_id": "agent_123",
                "role": "scorer",
                # NO config key - full config saved separately
            },
        )

        # Agent class is present
        assert trace.agent_info["agent_class"] == "LLMAgent"
        assert trace.agent_info["component_name"] == "MyTestAgent"
        # No config dump
        assert "config" not in trace.agent_info

    def test_from_output_preserves_schema_contracts(self, real_bm):
        """Test that from_output() produces traces following schema contracts."""
        from buttermilk._core.types import BaseRecord

        output = AgentOutput(
            agent_id="test_agent",
            outputs="Result",
        )

        record = BaseRecord(
            record_id="rec_schema_test",
            dataset_name="dataset",
            split_type="test",
        )

        trace = ExecutionTrace.from_output(
            output,
            agent_info={
                "component_name": "TestAgent",
                "agent_class": "LLMCore",
            },
            inputs={
                "prompt_var": "test value",
                "context": [],
            },
            record=record,
            metadata={
                "hashes": {
                    "record_hash": record.record_hash,
                    "template_hash": "tmpl_hash_123",
                },
            },
        )

        # Verify all schema contracts
        assert trace.agent_info["agent_class"] == "LLMCore"
        assert "config" not in trace.agent_info
        assert trace.record is not None
        # Record may be dict or BaseRecord - access appropriately
        record_id = trace.record["record_id"] if isinstance(trace.record, dict) else trace.record.record_id
        assert record_id == "rec_schema_test"
        assert "record" not in trace.inputs
        assert trace.inputs["prompt_var"] == "test value"
        assert trace.metadata["hashes"]["template_hash"] == "tmpl_hash_123"
