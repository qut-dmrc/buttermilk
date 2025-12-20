"""Unit tests for LiteLLMWrapper."""

from unittest.mock import MagicMock, patch

import pytest
from autogen_core.models import SystemMessage, UserMessage
from pydantic import BaseModel, ConfigDict

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import (
    LITELLM_AVAILABLE,
    LiteLLMWrapper,
    ModelInfo,
    ModelParameters,
    autogen_to_litellm_messages,
    litellm_to_autogen_result,
)


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not installed")
class TestLiteLLMWrapper:
    """Test suite for LiteLLMWrapper functionality."""

    def test_init_requires_litellm(self):
        """Test that LiteLLMWrapper requires LiteLLM to be installed."""
        if not LITELLM_AVAILABLE:
            model_info = ModelInfo(
                vision=False, function_calling=True, json_output=False, family="gpt-4"
            )

            with pytest.raises(ImportError, match="LiteLLM is not installed"):
                LiteLLMWrapper(
                    model="gpt-4", model_info=model_info, litellm_model_name="gpt-4"
                )

    def test_init_valid_params(self):
        """Test LiteLLMWrapper initialization with valid parameters."""
        model_info = ModelInfo(
            vision=False, function_calling=True, json_output=False, family="gpt-4"
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com",
            default_parameters=ModelParameters(temperature=0.7),
        )

        assert wrapper.model == "gpt-4"
        assert wrapper.litellm_model_name == "gpt-4"
        assert wrapper.api_key == "test-key"
        assert wrapper.base_url == "https://api.openai.com"
        assert wrapper.default_parameters.temperature == 0.7


class TestMessageFormatConversion:
    """Test message format conversion utilities."""

    def test_autogen_to_litellm_system_message(self):
        """Test conversion of SystemMessage."""
        messages = [SystemMessage(content="You are a helpful assistant.")]
        litellm_messages = autogen_to_litellm_messages(messages)

        assert len(litellm_messages) == 1
        assert litellm_messages[0]["role"] == "system"
        assert litellm_messages[0]["content"] == "You are a helpful assistant."

    def test_autogen_to_litellm_user_message(self):
        """Test conversion of UserMessage."""
        messages = [UserMessage(content="Hello!", source="user")]
        litellm_messages = autogen_to_litellm_messages(messages)

        assert len(litellm_messages) == 1
        assert litellm_messages[0]["role"] == "user"
        assert litellm_messages[0]["content"] == "Hello!"

    def test_autogen_to_litellm_conversation(self):
        """Test conversion of multi-turn conversation."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!", source="user"),
            UserMessage(content="How are you?", source="user"),
        ]
        litellm_messages = autogen_to_litellm_messages(messages)

        assert len(litellm_messages) == 3
        assert litellm_messages[0]["role"] == "system"
        assert litellm_messages[1]["role"] == "user"
        assert litellm_messages[2]["role"] == "user"

    def test_litellm_to_autogen_result_basic(self):
        """Test conversion of basic LiteLLM response."""
        # Mock LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! I'm doing well."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.cached = False

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        result = litellm_to_autogen_result(mock_response, mock_usage, "gpt-4")

        assert result.content == "Hello! I'm doing well."
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.cached is False


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not installed")
@pytest.mark.anyio
class TestLiteLLMWrapperCreate:
    """Test LiteLLMWrapper.create() method."""

    async def test_create_basic_completion(self):
        """Test basic completion call."""
        model_info = ModelInfo(
            vision=False, function_calling=True, json_output=False, family="gpt-4"
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
        )

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Say hello!", source="user"),
        ]

        # Mock the acompletion call
        with patch("buttermilk._core.llms.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Hello!"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_response.cached = False

            mock_acompletion.return_value = mock_response

            result = await wrapper.create(messages=messages)

            assert result.content == "Hello!"
            assert result.finish_reason == "stop"
            assert mock_acompletion.called

            # Check that correct parameters were passed
            call_args = mock_acompletion.call_args
            assert call_args[1]["model"] == "gpt-4"
            assert call_args[1]["api_key"] == "test-key"
            assert len(call_args[1]["messages"]) == 2

    async def test_create_with_retry_on_rate_limit(self):
        """Test retry logic on rate limit errors."""
        model_info = ModelInfo(
            vision=False, function_calling=True, json_output=False, family="gpt-4"
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
            max_retries=2,
            min_wait_seconds=0.1,  # Fast for testing
            jitter_seconds=0,
        )

        messages = [UserMessage(content="Hello!", source="user")]

        with patch("buttermilk._core.llms.acompletion") as mock_acompletion:
            # First call fails with rate limit, second succeeds
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Hello!"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_response.cached = False

            mock_acompletion.side_effect = [
                Exception("Rate limit exceeded"),
                mock_response,
            ]

            result = await wrapper.create(messages=messages)

            assert result.content == "Hello!"
            assert mock_acompletion.call_count == 2  # One failure + one success

    async def test_create_failure_after_max_retries(self):
        """Test that error is raised after max retries."""
        model_info = ModelInfo(
            vision=False, function_calling=True, json_output=False, family="gpt-4"
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
            max_retries=1,
            min_wait_seconds=0.01,
            jitter_seconds=0,
        )

        messages = [UserMessage(content="Hello!", source="user")]

        with patch("buttermilk._core.llms.acompletion") as mock_acompletion:
            mock_acompletion.side_effect = Exception("Rate limit exceeded")

            with pytest.raises(ProcessingError, match="LiteLLM call failed"):
                await wrapper.create(messages=messages)

            assert mock_acompletion.call_count == 2  # Initial + 1 retry


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not installed")
@pytest.mark.anyio
class TestLiteLLMWrapperStructuredOutput:
    """Test structured output with LiteLLMWrapper."""

    async def test_create_with_schema(self):
        """Test structured output with Pydantic schema."""

        class TestSchema(BaseModel):
            model_config = ConfigDict(extra='forbid')

            summary: str
            sentiment: str

        model_info = ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="gpt-4",
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
        )

        messages = [UserMessage(content="Analyze this text", source="user")]

        with patch("buttermilk._core.llms.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[
                0
            ].message.content = '{"summary": "Test summary", "sentiment": "positive"}'
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
            mock_response.cached = False

            mock_acompletion.return_value = mock_response

            result = await wrapper.create(messages=messages, schema=TestSchema)

            assert isinstance(result.parsed_object, TestSchema)
            assert result.parsed_object.summary == "Test summary"
            assert result.parsed_object.sentiment == "positive"


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not installed")
class TestLiteLLMWrapperPricing:
    """Test pricing calculation in LiteLLMWrapper."""

    def test_calculate_pricing_with_usage(self):
        """Test pricing calculation with valid usage data."""
        model_info = ModelInfo(
            vision=False, function_calling=True, json_output=False, family="gpt-4"
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
        )

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        pricing = wrapper._calculate_pricing(mock_usage)

        assert "prompt_tokens" in pricing
        assert "completion_tokens" in pricing
        assert "total_cost" in pricing
        assert pricing["prompt_tokens"] == 100
        assert pricing["completion_tokens"] == 50

    def test_calculate_pricing_without_usage(self):
        """Test pricing calculation when usage data is missing."""
        model_info = ModelInfo(
            vision=False, function_calling=True, json_output=False, family="gpt-4"
        )

        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="test-key",
        )

        pricing = wrapper._calculate_pricing(None)

        assert pricing["prompt_tokens"] == 0
        assert pricing["completion_tokens"] == 0
        assert pricing["total_cost"] == 0.0
