"""Tests for LLMProcessor - Unified Processor Architecture implementation.

This module tests the LLMProcessor, which is the unified architecture version of LLMCore.
LLMProcessor implements the Processor protocol and inherits from UnifiedProcessor.

Tests use REAL templates and data patterns (no mocking internal code) and follow fail-fast philosophy.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from buttermilk._core.llms import CreateResult
from buttermilk._core.messages import RequestUsage
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord
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

    def test_llm_processor_inherits_from_processor_core(self):
        """Verify LLMProcessor inherits from ProcessorCore."""
        from buttermilk._core.processor_core import ProcessorCore

        assert issubclass(LLMProcessor, ProcessorCore)

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
        mock_bm.llms.get_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
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
            assert "llm_output" in output_record.metadata
            assert output_record.metadata["llm_output"]["content"] == "LLM generated response"

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
        mock_bm.llms.get_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1

            # Verify LLM was called with messages from the template
            assert mock_client.call_chat.called
            call_args = mock_client.call_chat.call_args
            messages_sent = call_args.kwargs["messages"]

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
        mock_bm.llms.get_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            output_record = outputs[0]

            # Verify metadata contains LLM execution info
            assert "llm_output" in output_record.metadata
            assert output_record.metadata["llm_output"]["model"] == "gpt-4"
            assert output_record.metadata["llm_output"]["template"] == "test/simple"

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
        mock_bm.llms.get_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
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
        mock_bm.llms.get_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
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
            assert "llm_output" in output_record.metadata
            assert output_record.metadata["llm_output"]["content"] == "Processed output from LLM"
            assert output_record.metadata["llm_output"]["model"] == "gpt-4"

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
        mock_bm.llms.get_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            output_record = outputs[0]

            # Structured output is stored in metadata.llm_output.content
            assert "llm_output" in output_record.metadata
            # Original record fields preserved
            assert output_record.record_id == "structured-001"


class TestLLMProcessorInputs:
    """Test JMESPath-based inputs resolution for per-record config overrides."""

    def test_inputs_field_defaults_to_empty(self):
        """Verify inputs defaults to empty dict."""
        processor = LLMProcessor(model="gpt-4", template="test/simple")
        assert processor.inputs == {}

    def test_inputs_field_accepted_in_config(self):
        """Verify inputs can be set via config."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            inputs={"model": "record.metadata.model"},
        )
        assert processor.inputs == {"model": "record.metadata.model"}

    @pytest.mark.anyio
    async def test_inputs_overrides_model_per_record(self):
        """Verify inputs can override model from record metadata."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            inputs={"model": "record.metadata.model"},
        )

        record = BaseRecord(
            record_id="inputs-001",
            content="Test content",
            metadata={"var": "test", "model": "claude-3-opus"},
        )

        context = ProcessingContext(session_id="inputs-session", record=record)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response from overridden model",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            # The LLM client should have been fetched with the overridden model
            mock_bm.llms.get_client.assert_called_with("claude-3-opus")
            # Enriched metadata should reflect the resolved model
            assert outputs[0].metadata["llm_output"]["model"] == "claude-3-opus"

    @pytest.mark.anyio
    async def test_inputs_overrides_template_per_record(self):
        """Verify inputs can override template from record metadata."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            inputs={"template": "record.metadata.template_name"},
        )

        record = BaseRecord(
            record_id="inputs-002",
            content="Test content",
            metadata={"var": "test", "text": "Some text to analyze", "template_name": "test/structured_output"},
        )

        context = ProcessingContext(session_id="inputs-session", record=record)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            # Enriched metadata should reflect the resolved template
            assert outputs[0].metadata["llm_output"]["template"] == "test/structured_output"

    @pytest.mark.anyio
    async def test_inputs_injects_extra_template_vars(self):
        """Verify non-config inputs are injected as template variables."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            inputs={"country": "record.metadata.country"},
        )

        record = BaseRecord(
            record_id="inputs-003",
            content="Test content",
            metadata={"var": "test", "country": "Australia"},
        )

        context = ProcessingContext(session_id="inputs-session", record=record)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            # Model stays as default since we didn't override it
            assert outputs[0].metadata["llm_output"]["model"] == "gpt-4"

    @pytest.mark.anyio
    async def test_inputs_falls_back_to_default_when_missing(self):
        """Verify that when a JMESPath path resolves to None, the default is used."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            inputs={"model": "record.metadata.model"},
        )

        # Record has no 'model' in metadata
        record = BaseRecord(
            record_id="inputs-004",
            content="Test content",
            metadata={"var": "test"},
        )

        context = ProcessingContext(session_id="inputs-session", record=record)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            # Should fall back to configured default
            mock_bm.llms.get_client.assert_called_with("gpt-4")
            assert outputs[0].metadata["llm_output"]["model"] == "gpt-4"

    @pytest.mark.anyio
    async def test_no_inputs_behaves_identically_to_before(self):
        """Verify that a processor with no inputs works the same as before the refactor."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
        )

        record = BaseRecord(
            record_id="inputs-005",
            content="Test content",
            metadata={"var": "test"},
        )

        context = ProcessingContext(session_id="inputs-session", record=record)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            assert len(outputs) == 1
            assert outputs[0].metadata["llm_output"]["model"] == "gpt-4"
            assert outputs[0].metadata["llm_output"]["template"] == "test/simple"

    def test_resolve_inputs_with_top_level_record_fields(self):
        """Verify _resolve_inputs can access top-level record fields."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/simple",
            inputs={"content_text": "record.content", "ds_name": "record.dataset_name"},
        )

        record = BaseRecord(
            record_id="resolve-001",
            content="The actual content",
            dataset_name="my_dataset",
            metadata={},
        )

        context = ProcessingContext(session_id="test", record=record)
        resolved = processor._resolve_inputs(context)

        assert resolved["content_text"] == "The actual content"
        assert resolved["ds_name"] == "my_dataset"


class TestLLMProcessorVariantParamsTemplateVars:
    """Regression tests for issue #373: variant_params must be merged into template vars.

    ParameterExpansionProcessor stores criteria/model/etc in record.metadata["_variant_params"].
    The pipeline promotes these to context.variant_params.  LLMProcessor must merge the
    non-config fields (e.g. criteria) into the template variables so templates can render them.
    """

    @pytest.mark.anyio
    async def test_variant_params_non_config_fields_reach_template(self):
        """Non-config variant_params (e.g. criteria) must be passed to the template.

        Regression: #364 dropped non-config variant_params from template rendering,
        causing "unfilled parameters: criteria" errors for every record.
        """
        processor = LLMProcessor(
            model="gpt-4",
            template="test/with_unfilled_vars",
            fail_on_unfilled_parameters=False,  # allow missing_var; required_var comes from variant_params
        )

        record = BaseRecord(record_id="reg-373", content="some content")
        context = ProcessingContext(
            session_id="test",
            record=record,
            variant_params={"required_var": "from_variant"},
        )

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="ok",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=5, completion_tokens=5),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

        # If variant_params weren't merged, required_var would be unfilled and
        # process_with_llm would raise ProcessingError; getting here means the fix works.
        assert len(outputs) == 1

    @pytest.mark.anyio
    async def test_variant_params_config_fields_override_model_and_template(self):
        """Config-level variant_params (model, template) still override processor defaults."""
        processor = LLMProcessor(
            model="default-model",
            template="test/simple",
        )

        record = BaseRecord(record_id="reg-373b", content="content", metadata={"var": "test"})
        context = ProcessingContext(
            session_id="test",
            record=record,
            variant_params={"model": "override-model", "template": "test/simple"},
        )

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="ok",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=5, completion_tokens=5),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

        assert len(outputs) == 1
        assert mock_bm.llms.get_client.call_args[0][0] == "override-model"

    @pytest.mark.anyio
    async def test_inputs_jmespath_takes_priority_over_variant_params(self):
        """JMESPath inputs override variant_params for the same key."""
        processor = LLMProcessor(
            model="gpt-4",
            template="test/with_unfilled_vars",
            inputs={"required_var": "record.metadata.from_record"},
            fail_on_unfilled_parameters=False,
        )

        record = BaseRecord(
            record_id="reg-373c",
            content="content",
            metadata={"from_record": "record_wins"},
        )
        context = ProcessingContext(
            session_id="test",
            record=record,
            variant_params={"required_var": "variant_loses"},
        )

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="ok",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=5, completion_tokens=5),
            cached=False,
        )
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            # Spy on _build_llm_core to verify the right extra_template_vars are used
            original_build = processor._build_llm_core
            build_calls = []

            def spy_build(**kwargs):
                build_calls.append(kwargs)
                return original_build(**kwargs)

            processor._build_llm_core = spy_build
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

        # JMESPath "record_wins" should override variant_params "variant_loses"
        assert len(build_calls) == 1
        assert build_calls[0]["extra_template_vars"].get("required_var") == "record_wins"
