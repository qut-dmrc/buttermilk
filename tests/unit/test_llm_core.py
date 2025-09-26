"""Unit tests for LLMCore shared functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core.models import SystemMessage, UserMessage
from pydantic import BaseModel

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llm_core import LLMCore, LLMResult
from buttermilk._core.llms import CreateResult, ModelOutput
from buttermilk._core.types import BaseRecord


class OutputModelForTesting(BaseModel):
    """Test Pydantic model for structured output."""
    summary: str
    sentiment: str


class TestLLMCore:
    """Test suite for LLMCore functionality."""

    def test_init_valid_params(self):
        """Test LLMCore initialization with valid parameters."""
        params = {
            "model": "gpt-4",
            "template": "test_template",
            "temperature": 0.7,
            "fail_on_unfilled_parameters": False
        }

        core = LLMCore(**params)

        assert core._model == "gpt-4"
        assert core._template == "test_template"
        assert core._fail_on_unfilled_parameters is False
        assert core.output_model is None
        assert core.tools == []

    def test_init_missing_model(self):
        """Test LLMCore initialization fails without model."""
        params = {
            "template": "test_template"
        }

        with pytest.raises(ValueError, match="'model' is required"):
            LLMCore(**params)

    def test_init_missing_template(self):
        """Test LLMCore initialization fails without template."""
        params = {
            "model": "gpt-4"
        }

        with pytest.raises(ValueError, match="'template' is required"):
            LLMCore(**params)

    def test_init_with_output_model(self):
        """Test LLMCore initialization with structured output model."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }

        core = LLMCore(**params, output_model=OutputModelForTesting)

        assert core.output_model == OutputModelForTesting

    @pytest.mark.asyncio
    async def test_fill_template_basic(self):
        """Test template filling with basic inputs."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        # Mock the template loading
        with patch("buttermilk._core.llm_core.load_template") as mock_load:
            mock_load.return_value = (
                "Template content with {{var}}",
                set(),  # No unfilled vars
                "template_hash_123"
            )

            with patch("buttermilk._core.llm_core.make_messages") as mock_make:
                mock_make.return_value = [SystemMessage(content="System prompt"), UserMessage(content="User prompt", source="test")]

                messages = await core._fill_template(
                    inputs={"var": "value"},
                    context=[],
                    records=[]
                )

                assert len(messages) == 2
                assert isinstance(messages[0], SystemMessage)
                assert isinstance(messages[1], UserMessage)
                assert core._template_metadata["template_name"] == "test_template"
                assert core._template_metadata["template_hash"] == "template_hash_123"

    @pytest.mark.asyncio
    async def test_fill_template_with_unfilled_vars_strict(self):
        """Test template filling fails with unfilled vars in strict mode."""
        params = {
            "model": "gpt-4",
            "template": "test_template",
            "fail_on_unfilled_parameters": True
        }
        core = LLMCore(**params)

        with patch("buttermilk._core.llm_core.load_template") as mock_load:
            mock_load.return_value = (
                "Template with {{missing_var}}",
                {"missing_var"},  # Unfilled var
                "hash"
            )

            with patch("buttermilk._core.llm_core.make_messages") as mock_make:
                mock_make.return_value = [UserMessage(content="Test", source="test")]

                with pytest.raises(ProcessingError, match="unfilled parameters: missing_var"):
                    await core._fill_template(
                        inputs={},
                        context=[],
                        records=[]
                    )

    @pytest.mark.asyncio
    async def test_fill_template_with_unfilled_vars_lenient(self):
        """Test template filling continues with unfilled vars in lenient mode."""
        params = {
            "model": "gpt-4",
            "template": "test_template",
            "fail_on_unfilled_parameters": False
        }
        core = LLMCore(**params)

        with patch("buttermilk._core.llm_core.load_template") as mock_load:
            mock_load.return_value = (
                "Template with {{missing_var}}",
                {"missing_var"},
                "hash"
            )

            with patch("buttermilk._core.llm_core.make_messages") as mock_make:
                mock_make.return_value = [UserMessage(content="Test", source="test")]

                messages = await core._fill_template(
                    inputs={},
                    context=[],
                    records=[]
                )

                assert len(messages) == 1
                assert core._template_metadata["unfilled_vars"] == ["missing_var"]

    @pytest.mark.asyncio
    async def test_call_llm_with_trace_success(self):
        """Test successful LLM call with tracing."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        # Mock the BM and LLM client
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        from autogen_core.models import RequestUsage
        mock_result = CreateResult(
            content="LLM response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=50, completion_tokens=50),
            cached=False
        )
        with patch("buttermilk._core.llm_core.bm", mock_bm):
            result = await core._call_llm_with_trace(
                messages=[UserMessage(content="Test", source="test")],
                cancellation_token=None,
                parent_trace_id="parent_123"
            )

            assert result == mock_result
            mock_client.call_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_with_trace_failure(self):
        """Test LLM call failure handling."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.side_effect = Exception("API error")
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            with pytest.raises(ProcessingError, match="LLM call to 'gpt-4' failed"):
                await core._call_llm_with_trace(
                    messages=[UserMessage(content="Test", source="test")],
                    cancellation_token=None,
                    parent_trace_id=None
                )

    @pytest.mark.asyncio
    async def test_process_with_llm_success(self):
        """Test full LLM processing pipeline success."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        # Mock template filling
        with patch.object(core, "_fill_template") as mock_fill:
            mock_fill.return_value = [SystemMessage(content="System"), UserMessage(content="User", source="test")]

            # Mock LLM call
            with patch.object(core, "_call_llm_with_trace") as mock_call:
                from autogen_core.models import RequestUsage
                mock_call.return_value = CreateResult(
                    content="Response text",
                    finish_reason="stop",
                    usage=RequestUsage(prompt_tokens=25, completion_tokens=25),
                    cached=False
                )

                result = await core.process_with_llm(
                    inputs={"text": "input"},
                    context=None,
                    records=None,
                    parent_trace_id="trace_123"
                )

                assert isinstance(result, LLMResult)
                assert result.content == "Response text"
                assert result.metadata["model"] == "gpt-4"
                assert result.metadata["finish_reason"] == "stop"
                # Usage is a RequestUsage object, not a dict
                assert result.metadata["usage"].prompt_tokens == 25
                assert result.metadata["usage"].completion_tokens == 25
                assert result.error is None

    @pytest.mark.asyncio
    async def test_process_with_llm_structured_output(self):
        """Test LLM processing with structured output model."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params, output_model=OutputModelForTesting)

        with patch.object(core, "_fill_template") as mock_fill:
            mock_fill.return_value = [UserMessage(content="Test", source="test")]

            with patch.object(core, "_call_llm_with_trace") as mock_call:
                parsed_obj = OutputModelForTesting(
                    summary="Test summary",
                    sentiment="positive"
                )
                from autogen_core.models import RequestUsage
                mock_call.return_value = ModelOutput(
                    content='{"summary": "Test summary", "sentiment": "positive"}',
                    parsed_object=parsed_obj,
                    finish_reason="stop",
                    usage=RequestUsage(prompt_tokens=40, completion_tokens=35),
                    cached=False
                )

                result = await core.process_with_llm(
                    inputs={"text": "analyze this"}
                )

                assert isinstance(result.content, OutputModelForTesting)
                assert result.content.summary == "Test summary"
                assert result.content.sentiment == "positive"

    @pytest.mark.asyncio
    async def test_process_with_llm_processing_error(self):
        """Test LLM processing handles ProcessingError correctly."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        with patch.object(core, "_fill_template") as mock_fill:
            mock_fill.side_effect = ProcessingError("Template error")

            with pytest.raises(ProcessingError, match="Template error"):
                await core.process_with_llm(inputs={"text": "test"})

    @pytest.mark.asyncio
    async def test_process_with_llm_unexpected_error(self):
        """Test LLM processing wraps unexpected errors."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        with patch.object(core, "_fill_template") as mock_fill:
            mock_fill.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ProcessingError, match="LLMCore processing failed"):
                await core.process_with_llm(inputs={"text": "test"})

    @pytest.mark.asyncio
    async def test_process_with_records(self):
        """Test LLM processing with BaseRecord objects."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(**params)

        # Create test records
        record1 = BaseRecord(
            record_id="rec1",
            dataset_name="test",
            split_type="train"
        )
        record2 = BaseRecord(
            record_id="rec2",
            dataset_name="test",
            split_type="train"
        )

        with patch.object(core, "_fill_template") as mock_fill:
            mock_fill.return_value = [UserMessage(content="Test", source="test")]

            with patch.object(core, "_call_llm_with_trace") as mock_call:
                from autogen_core.models import RequestUsage
                mock_call.return_value = CreateResult(
                    content="Processed records",
                    finish_reason="stop",
                    usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
                    cached=False
                )

                result = await core.process_with_llm(
                    inputs={},
                    records=[record1, record2]
                )

                # Verify records were passed to template filling
                mock_fill.assert_called_once()
                call_args = mock_fill.call_args
                assert call_args[1]["records"] == [record1, record2]

                assert result.content == "Processed records"
