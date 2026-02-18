"""Unit tests for JMESPathTransform processor."""

import pytest
from pydantic import ValidationError

from buttermilk._core.types import BaseRecord
from buttermilk.processors.jmespath_transform import JMESPathTransform


class TestJMESPathTransform:
    """Test suite for JMESPathTransform processor."""

    @pytest.mark.anyio
    async def test_simple_field_extraction(self):
        """Test extraction of a single field from record."""
        # Create a record with nested output data
        record = BaseRecord(record_id="test_1", metadata={"outputs": {"result": "test_value"}})

        # Define mapping to extract result field
        processor = JMESPathTransform(mappings={"answer": "metadata.outputs.result"})

        # Process the record
        results = []
        async for result in processor.process(record, processor_stage="transform"):
            results.append(result)

        # Should yield one result
        assert len(results) == 1
        result_record = results[0]

        # Check that answer field was added
        assert hasattr(result_record, "answer")
        assert result_record.answer == "test_value"

        # Check that original metadata outputs are preserved (not the entire metadata due to record_hash)
        assert result_record.metadata["outputs"] == record.metadata["outputs"]

    @pytest.mark.anyio
    async def test_nested_object_construction(self):
        """Test building a new object from multiple parts."""
        # Create an ExecutionTrace-like record
        record = BaseRecord(
            record_id="trace_1",
            metadata={
                "agent_info": {"agent_id": "agent_123"},
                "outputs": {"conclusion": "test conclusion"},
                "call_id": "call_456",
            },
        )

        # Define mapping to construct nested object
        processor = JMESPathTransform(
            mappings={"answers": "{agent_id: metadata.agent_info.agent_id, result: metadata.outputs, answer_id: metadata.call_id}"}
        )

        # Process the record
        results = []
        async for result in processor.process(record, processor_stage="transform"):
            results.append(result)

        assert len(results) == 1
        result_record = results[0]

        # Check that answers field was constructed
        assert hasattr(result_record, "answers")
        assert isinstance(result_record.answers, dict)
        assert result_record.answers["agent_id"] == "agent_123"
        assert result_record.answers["result"]["conclusion"] == "test conclusion"
        assert result_record.answers["answer_id"] == "call_456"

    @pytest.mark.anyio
    async def test_missing_field_handling(self):
        """Test graceful handling of missing fields."""
        # Create a record without the expected field
        record = BaseRecord(record_id="test_2", metadata={"other_field": "value"})

        # Define mapping to non-existent field
        processor = JMESPathTransform(mappings={"value": "metadata.nonexistent.field"})

        # Process the record
        results = []
        async for result in processor.process(record, processor_stage="transform"):
            results.append(result)

        assert len(results) == 1
        result_record = results[0]

        # Check that value field was NOT added (missing field)
        assert not hasattr(result_record, "value")

        # Original metadata should be preserved
        assert result_record.metadata == record.metadata

    @pytest.mark.anyio
    async def test_multiple_mappings(self):
        """Test applying multiple transformations."""
        # Create a record with multiple fields
        record = BaseRecord(
            record_id="test_3",
            metadata={
                "data": {
                    "name": "test_name",
                    "value": 42,
                    "nested": {"key": "nested_value"},
                }
            },
        )

        # Define multiple mappings
        processor = JMESPathTransform(
            mappings={
                "extracted_name": "metadata.data.name",
                "extracted_value": "metadata.data.value",
                "extracted_nested": "metadata.data.nested.key",
            }
        )

        # Process the record
        results = []
        async for result in processor.process(record, processor_stage="transform"):
            results.append(result)

        assert len(results) == 1
        result_record = results[0]

        # Check that all mappings were applied
        assert hasattr(result_record, "extracted_name")
        assert result_record.extracted_name == "test_name"

        assert hasattr(result_record, "extracted_value")
        assert result_record.extracted_value == 42

        assert hasattr(result_record, "extracted_nested")
        assert result_record.extracted_nested == "nested_value"

    @pytest.mark.anyio
    async def test_preserves_original_fields(self):
        """Test that original fields remain intact."""
        # Create a record with existing fields
        record = BaseRecord(
            record_id="test_4",
            dataset_name="test_dataset",
            split_type="train",
            metadata={"source": "test", "data": {"value": "extracted"}},
        )

        # Define mapping to add new field
        processor = JMESPathTransform(mappings={"new_field": "metadata.data.value"})

        # Process the record
        results = []
        async for result in processor.process(record, processor_stage="transform"):
            results.append(result)

        assert len(results) == 1
        result_record = results[0]

        # Check that new field was added
        assert hasattr(result_record, "new_field")
        assert result_record.new_field == "extracted"

        # Check that original fields are preserved
        assert result_record.record_id == "test_4"
        assert result_record.dataset_name == "test_dataset"
        assert result_record.split_type == "train"
        assert result_record.metadata["source"] == "test"

    @pytest.mark.anyio
    async def test_invalid_jmespath_expression(self):
        """Test that invalid JMESPath expressions raise ValidationError during initialization."""
        with pytest.raises(ValidationError) as exc_info:
            JMESPathTransform(
                mappings={"result": "[[[invalid"}  # Invalid JMESPath syntax
            )

        # Verify the error message contains helpful information
        assert "Invalid JMESPath expression" in str(exc_info.value)
