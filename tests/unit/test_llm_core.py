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

        core = LLMCore(
            model=params["model"],
            template=params["template"],
            fail_on_unfilled_parameters=params["fail_on_unfilled_parameters"]
        )

        assert core._model == "gpt-4"
        assert core._template == "test_template"
        assert core._fail_on_unfilled_parameters is False
        assert core.output_model is None
        assert core.tools == []

    def test_init_missing_model(self):
        """Test LLMCore initialization works without model (uses empty string)."""
        # LLMCore now accepts empty model - no longer raises error
        core = LLMCore(model="", template="test_template")
        assert core._model == ""

    def test_init_missing_template(self):
        """Test LLMCore initialization works without template (uses empty string)."""
        # LLMCore now accepts empty template - no longer raises error
        core = LLMCore(model="gpt-4", template="")
        assert core._template == ""

    def test_init_with_output_model(self):
        """Test LLMCore initialization with structured output model."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }

        core = LLMCore(
            model=params["model"],
            template=params["template"],
            output_model=OutputModelForTesting
        )

        assert core.output_model == OutputModelForTesting

    @pytest.mark.asyncio
    async def test_fill_template_basic(self):
        """Test template filling with basic inputs."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
                    inputs={"var": "value", "context": [], "records": []}
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
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
                        inputs={"context": [], "records": []}
                    )

    @pytest.mark.asyncio
    async def test_fill_template_with_unfilled_vars_lenient(self):
        """Test template filling continues with unfilled vars in lenient mode."""
        params = {
            "model": "gpt-4",
            "template": "test_template",
            "fail_on_unfilled_parameters": False
        }
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", ""),
            fail_on_unfilled_parameters=params.get("fail_on_unfilled_parameters", True)
        )

        with patch("buttermilk._core.llm_core.load_template") as mock_load:
            mock_load.return_value = (
                "Template with {{missing_var}}",
                {"missing_var"},
                "hash"
            )

            with patch("buttermilk._core.llm_core.make_messages") as mock_make:
                mock_make.return_value = [UserMessage(content="Test", source="test")]

                messages = await core._fill_template(
                    inputs={"context": [], "records": []}
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
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
        # Set up the mock chain
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client
        mock_client.call_chat.return_value = mock_result

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
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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

    def test_combine_inputs_with_dict_and_kwargs(self):
        """Test that _combine_inputs properly merges dict inputs with kwargs."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(
            model=params["model"],
            template=params["template"]
        )

        # Test dict inputs + kwargs
        dict_inputs = {'my_var': 'from_dict', 'context': []}
        kwargs = {'my_var': 'from_kwargs', 'other_var': 'kwargs_only'}

        combined = core._combine_inputs(dict_inputs, kwargs)

        # kwargs should take precedence
        assert combined['my_var'] == 'from_kwargs'
        assert combined['other_var'] == 'kwargs_only'
        assert combined['context'] == []

    def test_combine_inputs_with_none_and_kwargs(self):
        """Test that _combine_inputs handles None inputs with kwargs."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(
            model=params["model"],
            template=params["template"]
        )

        kwargs = {'my_var': 'from_kwargs', 'other_var': 'kwargs_only'}
        combined = core._combine_inputs(None, kwargs)

        assert combined['my_var'] == 'from_kwargs'
        assert combined['other_var'] == 'kwargs_only'

    def test_combine_inputs_with_agentinput_and_kwargs(self):
        """Test that _combine_inputs properly handles AgentInput objects with kwargs."""
        from buttermilk._core.contract import AgentInput
        from buttermilk._core.types import Record

        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(
            model=params["model"],
            template=params["template"]
        )

        agent_input = AgentInput(
            inputs={'agent_var': 'agent_value'},
            context=[],
            records=[Record(record_id='test', content='test')]
        )

        kwargs = {'kwargs_var': 'kwargs_value'}
        combined = core._combine_inputs(agent_input, kwargs)

        # Should contain all AgentInput fields plus kwargs
        assert combined['inputs']['agent_var'] == 'agent_value'
        assert combined['kwargs_var'] == 'kwargs_value'
        assert combined['context'] == []
        assert len(combined['records']) == 1

    def test_template_variable_treated_normally(self):
        """Test that 'template' input variable is treated like any other variable."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(
            model=params["model"],
            template=params["template"]
        )

        # Template variable in inputs should be preserved as normal variable
        inputs = {'template': 'input_template_value', 'other': 'value'}
        kwargs = {'more': 'kwargs_value'}

        combined = core._combine_inputs(inputs, kwargs)

        # Template should be treated as normal input variable
        assert combined['template'] == 'input_template_value'
        assert combined['other'] == 'value'
        assert combined['more'] == 'kwargs_value'

        # The LLMCore should still use its own template from init
        assert core._template == 'test_template'  # From constructor, not inputs

    @pytest.mark.asyncio
    async def test_process_with_llm_success(self):
        """Test full LLM processing pipeline success."""
        params = {
            "model": "gpt-4",
            "template": "test_template"
        }
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
        core = LLMCore(
            model=params["model"],
            template=params["template"],
            output_model=OutputModelForTesting
        )

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
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
        core = LLMCore(
            model=params.get("model", ""),
            template=params.get("template", "")
        )

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
                # Records are now part of the combined inputs dict passed as first argument
                inputs_dict = call_args[0][0]  # First positional argument (inputs)
                assert inputs_dict["records"] == [record1, record2]

                assert result.content == "Processed records"
