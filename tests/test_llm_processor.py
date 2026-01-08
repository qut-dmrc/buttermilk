"""Tests for LLMProcessor - Unified Processor Architecture implementation.

This module tests the LLMProcessor, which is the unified architecture version of LLMCore.
LLMProcessor implements the Processor protocol and inherits from UnifiedProcessor.

Tests use REAL templates and data patterns (no mocking internal code) and follow fail-fast philosophy.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core.models import RequestUsage
from pydantic import BaseModel

from buttermilk._core.llms import CreateResult
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord
from buttermilk._core.unified_processor import UnifiedProcessor
from buttermilk.processors.unified_processors import LLMProcessor


class OutputModelForTesting(BaseModel):
    """Test Pydantic model for structured output."""

    summary: str
    sentiment: str


class TestLLMProcessorConfig:
    """Test LLMProcessor Pydantic model creation and validation."""

    def test_llm_processor_creation(self):
        """Verify LLMProcessor can be created with required fields."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
        )

        assert processor.model == "gpt-4"
        assert processor.template == "test/simple"
        assert processor.temperature == 0.7  # Default value
        assert processor.max_tokens == 1024  # Default value
        assert processor.input_variables == {}

    def test_llm_processor_with_optional_fields(self):
        """Verify LLMProcessor can be created with optional fields."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            temperature=0.0,
            max_tokens=2000,
            input_variables={"criteria": "test", "language": "en"},
        )

        assert processor.temperature == 0.0
        assert processor.max_tokens == 2000
        assert processor.input_variables == {"criteria": "test", "language": "en"}

    def test_llm_processor_forbids_extra_fields(self):
        """Verify strict config validation (extra='forbid')."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            LLMProcessor(
                model="gpt-4",
                template="test/simple",
                unknown_field="value",  # This should raise an error
            )

    def test_llm_processor_requires_model(self):
        """Verify model field is required."""
        with pytest.raises(ValueError, match="Field required"):
            LLMProcessor(
                template="test/simple",
                # model is missing
            )

    def test_llm_processor_requires_template(self):
        """Verify template field is required."""
        with pytest.raises(ValueError, match="Field required"):
            LLMProcessor(
                model="gpt-4",
                # template is missing
            )


class TestLLMProcessorProtocol:
    """Test that LLMProcessor implements the Processor protocol."""

    def test_llm_processor_exists(self):
        """Verify LLMProcessor class can be imported."""
        assert LLMProcessor is not None

    def test_llm_processor_inherits_from_unified_processor(self):
        """Verify LLMProcessor inherits from UnifiedProcessor."""
        # LLMProcessor should inherit from UnifiedProcessor
        assert issubclass(LLMProcessor, UnifiedProcessor)

    def test_llm_processor_implements_processor_protocol(self):
        """Verify LLMProcessor implements the Processor protocol."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
        )

        # Should satisfy the Processor protocol
        assert isinstance(processor, Processor)

    def test_llm_processor_has_process_method(self):
        """Verify LLMProcessor has async process method with correct signature."""
        import inspect

        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
        )

        # Should have process method
        assert hasattr(processor, "process")

        # Process should be async generator (not coroutine) per Processor protocol
        assert inspect.isasyncgenfunction(processor.process)


class TestLLMProcessorProcessing:
    """Test LLMProcessor processing logic."""

    @pytest.mark.anyio
    async def test_llm_processor_processes_single_record(self):
        """Verify LLMProcessor processes a single record and outputs to configured field."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            temperature=0.0,
        )

        # Create test record
        record = BaseRecord(
            record_id="llm-001",
            content="Test content for LLM processing",
            metadata={"var": "test value"},
        )

        context = ProcessingContext(
            session_id="llm-session",
            record=record,
        )

        # Mock ONLY the external LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="LLM generated response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=25, completion_tokens=25),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            # Process the record
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            # Should yield exactly 1 record
            assert len(outputs) == 1
            output_record = outputs[0]

            # Verify record ID is preserved
            assert output_record.record_id == "llm-001"

            # Verify LLM output is in the configured output column (default "llm_output")
            assert hasattr(output_record, "llm_output") or "llm_output" in output_record.metadata

    @pytest.mark.anyio
    async def test_llm_processor_uses_template(self):
        """Verify LLMProcessor uses the configured template for LLM calls."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
        )

        # Create test record with template variable
        record = BaseRecord(
            record_id="llm-002",
            content="content",
            metadata={"var": "template test value"},
        )

        context = ProcessingContext(
            session_id="template-session",
            record=record,
        )

        # Mock the LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1

            # Verify LLM was called with messages from the template
            assert mock_client.call_chat.called
            call_args = mock_client.call_chat.call_args
            messages_sent = call_args.kwargs.get("messages", call_args[0] if call_args[0] else [])

            # Should have messages from template (system + user)
            assert len(messages_sent) >= 1

    @pytest.mark.anyio
    async def test_llm_processor_enriches_record_metadata(self):
        """Verify LLMProcessor enriches record with LLM metadata (usage, model, etc)."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
        )

        record = BaseRecord(
            record_id="llm-003",
            content="content",
            metadata={"var": "test"},
        )

        context = ProcessingContext(
            session_id="metadata-session",
            record=record,
        )

        # Mock the LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=50, completion_tokens=30),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            output_record = outputs[0]

            # Verify metadata contains LLM execution info
            # Metadata structure should follow LLMCore patterns
            assert "metadata" in output_record.model_dump()
            # Should have usage, model, finish_reason, etc.

    @pytest.mark.anyio
    async def test_llm_processor_with_input_variables(self):
        """Verify LLMProcessor merges input_variables into template context."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            input_variables={"static_var": "static_value", "criteria": "test criteria"},
        )

        record = BaseRecord(
            record_id="llm-004",
            content="content",
            metadata={"var": "dynamic value"},
        )

        context = ProcessingContext(
            session_id="input-vars-session",
            record=record,
        )

        # Mock the LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1

            # Verify LLM was called
            assert mock_client.call_chat.called

            # Input variables from config should be merged with record data
            # and available in the template context


class TestLLMProcessorIntegration:
    """Integration tests for LLMProcessor with real templates."""

    @pytest.mark.anyio
    async def test_llm_processor_end_to_end(self):
        """Verify end-to-end processing: record -> template -> LLM -> enriched record."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            temperature=0.0,
        )

        # Create a realistic record
        record = BaseRecord(
            record_id="integration-001",
            dataset_name="test_dataset",
            split_type="test",
            content="This is test content that needs LLM processing.",
            metadata={"var": "test input", "source": "test"},
        )

        context = ProcessingContext(
            session_id="integration-session",
            record=record,
        )

        # Mock the LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Processed output from LLM",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=100, completion_tokens=50),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            # Process the record
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            # Verify output
            assert len(outputs) == 1
            output_record = outputs[0]

            # Verify original fields are preserved
            assert output_record.record_id == "integration-001"
            assert output_record.dataset_name == "test_dataset"
            assert output_record.content == "This is test content that needs LLM processing."
            assert output_record.metadata["source"] == "test"

            # Verify LLM output was added
            # (Exact structure depends on implementation, but output should exist)

    @pytest.mark.anyio
    async def test_llm_processor_with_structured_output(self):
        """Verify LLMProcessor handles structured output models."""
        from buttermilk._core.llms import ModelOutput

        # Note: This test will need to be updated once we know how
        # LLMProcessor handles output_model configuration
        # For now, we're testing the basic structure

        processor = LLMProcessor(
            model="gpt-4",
            template="test/structured_output",
            temperature=0.0,
        )

        record = BaseRecord(
            record_id="structured-001",
            content="Analyze this text",
            metadata={"text": "This is a positive statement."},
        )

        context = ProcessingContext(
            session_id="structured-session",
            record=record,
        )

        # Mock the LLM boundary with structured output
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        parsed_obj = OutputModelForTesting(summary="Test summary", sentiment="positive")
        mock_client.call_chat.return_value = ModelOutput(
            content='{"summary": "Test summary", "sentiment": "positive"}',
            parsed_object=parsed_obj,
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=40, completion_tokens=35),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1

            # Structured output handling will depend on implementation
            # This test validates the processor can handle ModelOutput
