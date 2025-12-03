"""Unit tests for LLMCore shared functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core.models import RequestUsage, SystemMessage, UserMessage
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
            "fail_on_unfilled_parameters": False,
        }

        core = LLMCore(
            model=params["model"],
            template=params["template"],
            fail_on_unfilled_parameters=params["fail_on_unfilled_parameters"],
        )

        assert core.model == "gpt-4"
        assert core.template == "test_template"
        assert core._fail_on_unfilled_parameters is False
        assert core.output_model is None
        assert core.tools == []

    def test_init_missing_model(self):
        """Test LLMCore initialization works without model (uses empty string)."""
        # LLMCore now accepts empty model - no longer raises error
        core = LLMCore(model="", template="test_template")
        assert core.model == ""

    def test_init_missing_template(self):
        """Test LLMCore initialization works without template (uses empty string)."""
        # LLMCore now accepts empty template - no longer raises error
        core = LLMCore(model="gpt-4", template="")
        assert core.template == ""

    def test_init_with_output_model(self):
        """Test LLMCore initialization with structured output model."""
        params = {"model": "gpt-4", "template": "test_template"}

        core = LLMCore(
            model=params["model"],
            template=params["template"],
            output_model=OutputModelForTesting,
        )

        assert core.output_model == OutputModelForTesting

    @pytest.mark.anyio
    async def test_fill_template_basic(self):
        """Test template filling with basic template_vars using real template file."""
        core = LLMCore(model="gpt-4", template="test/simple")

        messages = await core._fill_template(
            template_vars={"var": "test value", "context": [], "records": []}
        )

        # Should have system and user messages
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], UserMessage)
        # User message should contain the filled variable
        assert "test value" in messages[1].content

        # Template metadata should be set
        assert core.template_metadata["template_name"] == "test/simple"
        assert "template_hash" in core.template_metadata
        assert core.template_metadata["unfilled_vars"] == []

    @pytest.mark.anyio
    async def test_fill_template_with_unfilled_vars_strict(self):
        """Test template filling fails with unfilled vars in strict mode using real template."""
        core = LLMCore(
            model="gpt-4",
            template="test/with_unfilled_vars",
            fail_on_unfilled_parameters=True,
        )

        # Only provide required_var, leave missing_var undefined
        with pytest.raises(ProcessingError, match="unfilled parameters"):
            await core._fill_template(
                template_vars={"required_var": "value", "context": [], "records": []}
            )

    @pytest.mark.anyio
    async def test_fill_template_with_unfilled_vars_lenient(self):
        """Test template filling continues with unfilled vars in lenient mode using real template."""
        core = LLMCore(
            model="gpt-4",
            template="test/with_unfilled_vars",
            fail_on_unfilled_parameters=False,
        )

        # Only provide required_var, leave missing_var undefined
        messages = await core._fill_template(
            template_vars={"required_var": "value", "context": [], "records": []}
        )

        # Should return messages even with unfilled vars
        assert len(messages) >= 1
        # Unfilled vars should be recorded in metadata
        assert "missing_var" in core.template_metadata["unfilled_vars"]

    @pytest.mark.anyio
    async def test_call_llm_with_trace_success(self):
        """Test successful LLM call with tracing - mocking only at boundary."""
        core = LLMCore(model="gpt-4", template="test/simple")

        # Mock ONLY the external boundary (bm.llms)
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_result = CreateResult(
            content="LLM response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=50, completion_tokens=50),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client
        mock_client.call_chat.return_value = mock_result

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            result = await core._call_llm_with_trace(
                messages=[UserMessage(content="Test", source="test")],
                cancellation_token=None,
                parent_trace_id="parent_123",
            )

            assert result == mock_result
            mock_client.call_chat.assert_called_once()

    @pytest.mark.anyio
    async def test_call_llm_with_trace_failure(self):
        """Test LLM call failure handling - mocking only at boundary."""
        core = LLMCore(model="gpt-4", template="test/simple")

        # Mock ONLY the external boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.side_effect = Exception("API error")
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            with pytest.raises(ProcessingError, match="LLM call to 'gpt-4' failed"):
                await core._call_llm_with_trace(
                    messages=[UserMessage(content="Test", source="test")],
                    cancellation_token=None,
                    parent_trace_id=None,
                )

    @pytest.mark.anyio
    async def test_process_with_llm_success(self):
        """Test full LLM processing pipeline - real template, boundary-only mocking."""
        core = LLMCore(model="gpt-4", template="test/simple")

        # Mock ONLY the external LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Response text",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=25, completion_tokens=25),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            result = await core.process_with_llm(
                template_vars={"var": "test input"},
                parent_trace_id="trace_123",
            )

            # Verify observable outcomes
            assert isinstance(result, LLMResult)
            assert result.content == "Response text"
            assert result.metadata["model"] == "gpt-4"
            assert result.metadata["finish_reason"] == "stop"
            # Usage is converted to dict in metadata
            assert result.metadata["usage"]["prompt_tokens"] == 25
            assert result.metadata["usage"]["completion_tokens"] == 25
            assert result.error is None

    @pytest.mark.anyio
    async def test_process_with_llm_structured_output(self):
        """Test LLM processing with structured output - real template, boundary-only mocking."""
        core = LLMCore(
            model="gpt-4",
            template="test/structured_output",
            output_model=OutputModelForTesting,
        )

        # Mock ONLY the external LLM boundary
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
            result = await core.process_with_llm(template_vars={"text": "analyze this"})

            # Verify structured output was parsed correctly
            assert isinstance(result.content, OutputModelForTesting)
            assert result.content.summary == "Test summary"
            assert result.content.sentiment == "positive"

    @pytest.mark.anyio
    async def test_process_with_llm_processing_error(self):
        """Test LLM processing handles ProcessingError from real invalid template."""
        # Use strict mode with template that has unfilled vars to trigger ProcessingError
        core = LLMCore(
            model="gpt-4",
            template="test/with_unfilled_vars",
            fail_on_unfilled_parameters=True,
        )

        # Don't provide required variables - should trigger ProcessingError in _fill_template
        with pytest.raises(ProcessingError, match="unfilled parameters"):
            await core.process_with_llm(template_vars={"context": [], "records": []})

    @pytest.mark.anyio
    async def test_process_with_llm_unexpected_error(self):
        """Test LLM processing wraps unexpected errors from real scenarios."""
        # Use a non-existent template to trigger RuntimeError
        core = LLMCore(model="gpt-4", template="nonexistent/template/path")

        # Should wrap the error in ProcessingError
        with pytest.raises(ProcessingError, match="LLMCore processing failed"):
            await core.process_with_llm(template_vars={"text": "test"})

    @pytest.mark.anyio
    async def test_process_with_records(self):
        """Test LLM processing with BaseRecord objects - real template, boundary-only mocking."""
        core = LLMCore(model="gpt-4", template="test/with_records")

        # Create test records
        record1 = BaseRecord(record_id="rec1", dataset_name="test", split_type="train")
        record2 = BaseRecord(record_id="rec2", dataset_name="test", split_type="train")

        # Mock ONLY the external LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="Processed records",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            # Pass records as template variables since there's no separate records param
            result = await core.process_with_llm(
                template_vars={"records": [record1, record2]}
            )

            # Verify records were processed
            assert result.content == "Processed records"
            # Verify the template actually received the records
            # (The with_records template includes record IDs in the prompt)
            assert mock_client.call_chat.called
            call_args = mock_client.call_chat.call_args
            # Check kwargs for messages
            if call_args.kwargs:
                messages_sent = call_args.kwargs.get("messages", [])
            else:
                messages_sent = call_args[0] if call_args[0] else []
            # Convert messages to string to check if record IDs were included
            messages_text = str(messages_sent)
            assert "rec1" in messages_text or "rec2" in messages_text

    @pytest.mark.anyio
    async def test_template_metadata_preserved_in_result(self):
        """Test that template metadata (including template_hash) is preserved in LLMResult.

        Regression test for bug where template_hash was calculated but then
        overwritten when metadata dict was replaced instead of updated.

        This test uses REAL template loading to verify the actual bug is fixed.
        """
        core = LLMCore(model="gpt-4", template="test/simple")

        # Mock ONLY the external LLM boundary
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = CreateResult(
            content="LLM response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=25, completion_tokens=15),
            cached=False,
        )
        mock_bm.llms.get_autogen_chat_client.return_value = mock_client

        with patch("buttermilk._core.llm_core.bm", mock_bm):
            result = await core.process_with_llm(template_vars={"var": "test input"})

            # The bug: template metadata should be in result.metadata
            # If the bug exists, template metadata would be overwritten
            assert "template" in result.metadata, (
                "Template metadata should be present in result"
            )
            assert result.metadata["template"]["template_name"] == "test/simple"
            assert "template_hash" in result.metadata["template"]
            assert (
                result.metadata["template"]["template_hash"] != ""
            )  # Should have a hash
            assert result.metadata["template"]["unfilled_vars"] == []

            # Also verify other metadata is still there (wasn't overwritten)
            assert result.metadata["model"] == "gpt-4"
            assert result.metadata["finish_reason"] == "stop"

    @pytest.mark.anyio
    async def test_undefined_string_literal_vs_truly_undefined(self):
        """Test that passing literal 'undefined' string is different from truly undefined var.

        This tests the bug where a template expects a variable but receives the
        literal string 'undefined' - which should NOT satisfy the requirement.

        Scenario: If a variable is required by the template and receives the
        literal string 'undefined', the system should either:
        1. Fail because the variable is still considered unfilled (preferred)
        2. Process but produce nonsensical output (current buggy behavior)

        The test verifies both scenarios:
        - Truly undefined variable (missing from inputs) MUST fail with strict mode
        - Literal 'undefined' string is passed through (current behavior)
        """
        # Create core with strict parameter checking (default)
        core = LLMCore(
            model="gpt-4",
            template="test/with_unfilled_vars",
            fail_on_unfilled_parameters=True,
        )

        # Test 1: Truly undefined variable MUST fail
        with pytest.raises(ProcessingError, match="unfilled parameters"):
            await core._fill_template(
                template_vars={"required_var": "value", "context": [], "records": []}
                # missing_var is NOT provided - truly undefined
            )

        # Test 2: Literal 'undefined' string is treated as a value (BUG!)
        # This currently DOES NOT fail, but arguably should
        messages = await core._fill_template(
            template_vars={
                "required_var": "value",
                "missing_var": "undefined",  # Literal string "undefined"
                "context": [],
                "records": [],
            }
        )

        # The template was filled, but with nonsensical "undefined" string
        assert len(messages) >= 1
        # Check that the literal "undefined" made it into the message
        message_text = " ".join(
            msg.content for msg in messages if hasattr(msg, "content")
        )
        assert "undefined" in message_text.lower()

        # TODO: Consider if we should detect and fail on special values like:
        # - "undefined"
        # - "null"
        # - "None"
        # - "" (empty string)
        # These might indicate configuration errors rather than valid inputs.

    @pytest.mark.anyio
    async def test_fail_on_unfilled_parameters_default_is_true(self):
        """Test that fail_on_unfilled_parameters defaults to True (strict mode).

        This ensures the system fails fast on configuration errors by default,
        rather than silently proceeding with incomplete templates.
        """
        # Create core WITHOUT specifying fail_on_unfilled_parameters
        core = LLMCore(
            model="gpt-4",
            template="test/with_unfilled_vars",
            # fail_on_unfilled_parameters NOT specified - should default to True
        )

        # Should fail because default is strict mode
        with pytest.raises(ProcessingError, match="unfilled parameters"):
            await core._fill_template(
                template_vars={"required_var": "value", "context": [], "records": []}
                # missing_var is NOT provided
            )

        # Verify the internal flag is set to True
        assert core._fail_on_unfilled_parameters is True

    def test_parameters_includes_model_and_template(self):
        """Test that self.parameters captures model and template for trace writing.

        Regression test for issue #280: LLMCore rescore traces missing parameters.
        The trace writing in process() uses self.parameters, so model and template
        MUST be included there for observability.

        Without this, traces to BigQuery have incomplete parameters making it
        impossible to analyze which models/prompts were used.
        """
        core = LLMCore(
            model="gpt-4",
            template="judge",
            temperature=0.7,
            max_tokens=1000,
        )

        # CRITICAL: parameters must include model and template for trace writing
        assert "model" in core.parameters, "model must be in self.parameters for traces"
        assert "template" in core.parameters, (
            "template must be in self.parameters for traces"
        )

        # Verify the values are correct
        assert core.parameters["model"] == "gpt-4"
        assert core.parameters["template"] == "judge"

        # Additional kwargs should also be present
        assert core.parameters["temperature"] == 0.7
        assert core.parameters["max_tokens"] == 1000
