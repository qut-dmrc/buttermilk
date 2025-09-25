"""Unit tests for LLMProcessor pipeline component."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from buttermilk.processors.llm import LLMProcessor
from buttermilk._core.llm_core import LLMResult
from buttermilk._core.types import BaseRecord
from buttermilk._core.contract import ErrorEvent
from buttermilk._core.exceptions import ProcessingError


class StructuredOutputForTesting(BaseModel):
    """Test model for structured output."""
    category: str
    confidence: float


class TestLLMProcessor:
    """Test suite for LLMProcessor functionality."""

    def test_init_default_fields(self):
        """Test LLMProcessor initialization with default fields."""
        params = {
            "model": "gpt-4",
            "template": "summarize"
        }

        processor = LLMProcessor(parameters=params)

        assert processor.parameters == params
        assert processor.input_field == "content"
        assert processor.output_field == "processed"
        assert processor.output_model is None
        assert hasattr(processor, "llm_core")

    def test_init_custom_fields(self):
        """Test LLMProcessor initialization with custom fields."""
        params = {
            "model": "claude-3",
            "template": "analyze"
        }

        processor = LLMProcessor(
            parameters=params,
            input_field="full_text",
            output_field="analysis",
            output_model=StructuredOutputForTesting
        )

        assert processor.input_field == "full_text"
        assert processor.output_field == "analysis"
        assert processor.output_model == StructuredOutputForTesting
        assert processor.llm_core.output_model == StructuredOutputForTesting

    @pytest.mark.asyncio
    async def test_process_basic_record(self):
        """Test processing a basic record with string content."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"},
            input_field="text",
            output_field="summary"
        )

        # Create test record
        record = BaseRecord(
            record_id="test-001",
            dataset_name="test",
            split_type="test"
        )
        record.text = "This is test content to process."
        record.metadata = {}

        # Mock LLM core processing
        mock_result = LLMResult(
            content="Processed summary",
            metadata={"usage": {"tokens": 50}},
            trace_id="trace-123",
            template_metadata={"template_name": "test"}
        )

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            # Process the record
            results = []
            async for result in processor.process(record):
                results.append(result)

            assert len(results) == 1
            processed_record = results[0]

            # Verify the record was updated
            assert processed_record.summary == "Processed summary"
            assert "llm_summary" in processed_record.metadata
            assert processed_record.metadata["llm_summary"]["trace_id"] == "trace-123"

            # Verify LLM core was called correctly
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[1]["inputs"] == {"text": "This is test content to process."}
            assert call_args[1]["records"] == [record]

    @pytest.mark.asyncio
    async def test_process_record_with_dict_data(self):
        """Test processing a record that stores data in a dict."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"},
            input_field="content",
            output_field="result"
        )

        # Create record with data dict
        record = BaseRecord(
            record_id="test-002",
            dataset_name="test",
            split_type="test"
        )
        record.data = {"content": "Input text from data dict"}
        record.metadata = {}

        mock_result = LLMResult(
            content="Processed content",
            metadata={},
            trace_id="trace-456"
        )

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            results = []
            async for result in processor.process(record):
                results.append(result)

            processed_record = results[0]
            assert processed_record.data["result"] == "Processed content"

    @pytest.mark.asyncio
    async def test_process_structured_output(self):
        """Test processing with structured output model."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "classify"},
            output_model=StructuredOutputForTesting
        )

        record = BaseRecord(
            record_id="test-003",
            dataset_name="test",
            split_type="test"
        )
        record.content = "Text to classify"
        record.metadata = {}

        # Mock structured output
        structured_obj = StructuredOutputForTesting(
            category="technical",
            confidence=0.95
        )
        mock_result = LLMResult(
            content=structured_obj,
            metadata={"model": "gpt-4"},
            trace_id="trace-789"
        )

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            results = []
            async for result in processor.process(record):
                results.append(result)

            processed_record = results[0]
            assert isinstance(processed_record.processed, StructuredOutputForTesting)
            assert processed_record.processed.category == "technical"
            assert processed_record.processed.confidence == 0.95

    @pytest.mark.asyncio
    async def test_process_missing_input_field(self):
        """Test error handling when input field is missing."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"},
            input_field="missing_field"
        )

        record = BaseRecord(
            record_id="test-004",
            dataset_name="test",
            split_type="test"
        )
        record.metadata = {}

        results = []
        async for result in processor.process(record):
            results.append(result)

        # Should yield record with error
        assert len(results) == 1
        error_record = results[0]
        assert hasattr(error_record, "error")
        assert len(error_record.error) == 1
        assert "Field 'missing_field' not found" in error_record.error[0].content
        assert error_record.metadata.get("llm_error") is not None

    @pytest.mark.asyncio
    async def test_process_with_processing_error(self):
        """Test handling of ProcessingError from LLM core."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"}
        )

        record = BaseRecord(
            record_id="test-005",
            dataset_name="test",
            split_type="test"
        )
        record.content = "Test content"
        record.metadata = {}

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.side_effect = ProcessingError("Template not found")

            results = []
            async for result in processor.process(record):
                results.append(result)

            error_record = results[0]
            assert hasattr(error_record, "error")
            assert "Template not found" in error_record.error[0].content
            assert error_record.error[0].source == "LLMProcessor"

    @pytest.mark.asyncio
    async def test_process_with_unexpected_error(self):
        """Test handling of unexpected errors."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"}
        )

        record = BaseRecord(
            record_id="test-006",
            dataset_name="test",
            split_type="test"
        )
        record.content = "Test content"
        record.metadata = {}

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.side_effect = RuntimeError("Unexpected failure")

            results = []
            async for result in processor.process(record):
                results.append(result)

            error_record = results[0]
            assert hasattr(error_record, "error")
            assert "Unexpected error: Unexpected failure" in error_record.error[0].content

    @pytest.mark.asyncio
    async def test_process_with_parent_trace_id(self):
        """Test that parent trace ID is passed through from record metadata."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"}
        )

        record = BaseRecord(
            record_id="test-007",
            dataset_name="test",
            split_type="test"
        )
        record.content = "Test"
        record.metadata = {"trace_id": "parent-trace-123"}

        mock_result = LLMResult(
            content="Result",
            metadata={},
            trace_id="child-trace-456"
        )

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            async for _ in processor.process(record):
                pass

            # Verify parent trace ID was passed
            call_args = mock_process.call_args
            assert call_args[1]["parent_trace_id"] == "parent-trace-123"

    @pytest.mark.asyncio
    async def test_process_dict_input(self):
        """Test processing when input field contains a dict."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"},
            input_field="data",
            output_field="result"  # Add output field to avoid overwriting data dict
        )

        record = BaseRecord(
            record_id="test-008",
            dataset_name="test",
            split_type="test"
        )
        record.data = {"key1": "value1", "key2": "value2"}
        record.metadata = {}

        mock_result = LLMResult(
            content="Processed dict",
            metadata={},
            trace_id="trace-dict"
        )

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            results = []
            async for result in processor.process(record):
                results.append(result)

            processed_record = results[0]
            # Verify dict was passed directly as inputs
            call_args = mock_process.call_args
            assert call_args[1]["inputs"] == {"key1": "value1", "key2": "value2"}
            # Verify output was stored correctly (should be in data dict since record has data dict)
            assert processed_record.data["result"] == "Processed dict"

    @pytest.mark.asyncio
    async def test_process_preserves_original_record(self):
        """Test that processor preserves all original record fields."""
        processor = LLMProcessor(
            parameters={"model": "gpt-4", "template": "test"},
            output_field="enhanced"
        )

        record = BaseRecord(
            record_id="test-009",
            dataset_name="test",
            split_type="test"
        )
        record.content = "Original content"
        record.custom_field = "Custom value"
        record.metadata = {"original": "metadata"}

        mock_result = LLMResult(
            content="Enhanced content",
            metadata={},
            trace_id="trace-preserve"
        )

        with patch.object(processor.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            results = []
            async for result in processor.process(record):
                results.append(result)

            processed = results[0]
            # Original fields preserved
            assert processed.content == "Original content"
            assert processed.custom_field == "Custom value"
            assert processed.metadata["original"] == "metadata"
            # New field added
            assert processed.enhanced == "Enhanced content"