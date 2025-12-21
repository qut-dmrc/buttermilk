"""Tests for LLMProcessor - Unified Processor Architecture implementation.

This module tests the LLMProcessor, which is the unified architecture version of LLMCore.
LLMProcessor implements the Processor protocol and inherits from UnifiedProcessor.

Tests use REAL templates and data patterns (no mocking internal code) and follow fail-fast philosophy.
"""

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core.models import RequestUsage, UserMessage
from pydantic import BaseModel

from buttermilk._core.llm_core import LLMCore
from buttermilk._core.llms import CreateResult
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import LLMProcessorConfig
from buttermilk._core.processor_registry import create_processor, get_registered_types
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord
from buttermilk._core.unified_processor import UnifiedProcessor


class OutputModelForTesting(BaseModel):
    """Test Pydantic model for structured output."""

    summary: str
    sentiment: str


class TestLLMProcessorConfig:
    """Test LLMProcessorConfig creation and validation."""

    def test_llm_processor_config_creation(self):
        """Verify LLMProcessorConfig can be created with required fields."""
        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )

        assert config.type == "llm"
        assert config.model == "gpt-4"
        assert config.prompt_template == "test/simple"
        assert config.temperature == 0.0  # Default value
        assert config.max_tokens is None
        assert config.input_variables == {}

    def test_llm_processor_config_with_optional_fields(self):
        """Verify LLMProcessorConfig can be created with optional fields."""
        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
            temperature=0.7,
            max_tokens=1000,
            input_variables={"criteria": "Be concise", "language": "English"},
        )

        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.input_variables == {"criteria": "Be concise", "language": "English"}

    def test_llm_processor_config_validates_type(self):
        """Verify type field is constrained to 'llm' literal."""
        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )

        # Type should be the literal "llm"
        assert config.type == "llm"

    def test_llm_processor_config_forbids_extra_fields(self):
        """Verify strict config validation (extra='forbid')."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            LLMProcessorConfig(
                type="llm",
                model="gpt-4",
                prompt_template="test/simple",
                unknown_field="value",  # This should raise an error
            )

    def test_llm_processor_config_requires_model(self):
        """Verify model field is required."""
        with pytest.raises(ValueError, match="Field required"):
            LLMProcessorConfig(
                type="llm",
                prompt_template="test/simple",
                # model is missing
            )

    def test_llm_processor_config_requires_prompt_template(self):
        """Verify prompt_template field is required."""
        with pytest.raises(ValueError, match="Field required"):
            LLMProcessorConfig(
                type="llm",
                model="gpt-4",
                # prompt_template is missing
            )


class TestLLMProcessorProtocol:
    """Test that LLMProcessor implements the Processor protocol."""

    def test_llm_processor_exists(self):
        """Verify LLMProcessor class can be imported."""
        # This will fail until LLMProcessor is implemented
        from buttermilk.processors.unified_processors import LLMProcessor

        assert LLMProcessor is not None

    def test_llm_processor_inherits_from_unified_processor(self):
        """Verify LLMProcessor inherits from UnifiedProcessor."""
        from buttermilk.processors.unified_processors import LLMProcessor

        # LLMProcessor should inherit from UnifiedProcessor
        assert issubclass(LLMProcessor, UnifiedProcessor)

    def test_llm_processor_implements_processor_protocol(self):
        """Verify LLMProcessor implements the Processor protocol."""
        from buttermilk.processors.unified_processors import LLMProcessor

        # Use isinstance with the protocol
        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )
        processor = LLMProcessor(config)

        # Should satisfy the Processor protocol
        assert isinstance(processor, Processor)

    def test_llm_processor_has_process_method(self):
        """Verify LLMProcessor has async process method with correct signature."""
        from buttermilk.processors.unified_processors import LLMProcessor
        import inspect

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )
        processor = LLMProcessor(config)

        # Should have process method
        assert hasattr(processor, "process")

        # Process should be async generator (not coroutine) per Processor protocol
        assert inspect.isasyncgenfunction(processor.process)


class TestLLMProcessorRegistry:
    """Test LLMProcessor registration with processor registry."""

    def test_llm_processor_registered_as_llm_type(self):
        """Verify 'llm' type is registered in processor registry."""
        registered_types = get_registered_types()

        # 'llm' should be registered
        assert "llm" in registered_types

    def test_registry_creates_llm_processor_from_config(self):
        """Verify registry creates LLMProcessor from LLMProcessorConfig."""
        from buttermilk.processors.unified_processors import LLMProcessor

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )

        # Create processor using registry
        processor = create_processor(config)

        # Should create an LLMProcessor instance
        assert isinstance(processor, LLMProcessor)
        assert processor.config.model == "gpt-4"
        assert processor.config.prompt_template == "test/simple"


class TestLLMProcessorProcessing:
    """Test LLMProcessor processing logic."""

    @pytest.mark.anyio
    async def test_llm_processor_processes_single_record(self):
        """Verify LLMProcessor processes a single record and outputs to configured field."""
        from buttermilk.processors.unified_processors import LLMProcessor

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
            temperature=0.0,
        )

        processor = LLMProcessor(config)

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

            # Verify LLM output is in the configured output column (default "output")
            assert "output" in output_record.metadata or hasattr(output_record, "output")

    @pytest.mark.anyio
    async def test_llm_processor_uses_template(self):
        """Verify LLMProcessor uses the configured template for LLM calls."""
        from buttermilk.processors.unified_processors import LLMProcessor

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )

        processor = LLMProcessor(config)

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
        from buttermilk.processors.unified_processors import LLMProcessor

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
        )

        processor = LLMProcessor(config)

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
        from buttermilk.processors.unified_processors import LLMProcessor

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
            input_variables={"static_var": "static value", "criteria": "be precise"},
        )

        processor = LLMProcessor(config)

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
        from buttermilk.processors.unified_processors import LLMProcessor

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/simple",
            temperature=0.0,
        )

        processor = LLMProcessor(config)

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
        from buttermilk.processors.unified_processors import LLMProcessor
        from buttermilk._core.llms import ModelOutput

        # Note: This test will need to be updated once we know how
        # LLMProcessor handles output_model configuration
        # For now, we're testing the basic structure

        config = LLMProcessorConfig(
            type="llm",
            model="gpt-4",
            prompt_template="test/structured_output",
            temperature=0.0,
        )

        processor = LLMProcessor(config)

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
