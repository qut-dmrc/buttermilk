"""Unit tests for LiteLLMWrapper."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import (
    LiteLLMWrapper,
    ModelInfo,
    ModelParameters,
    _add_anthropic_cache_control,
    litellm_to_model_output,
    to_litellm_messages,
)
from buttermilk._core.messages import SystemMessage, UserMessage


class TestLiteLLMWrapper:
    """Test suite for LiteLLMWrapper functionality."""

    def test_init_valid_params(self):
        """Test LiteLLMWrapper initialization with valid parameters."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4")

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

    def test_to_litellm_system_message(self):
        """Test conversion of SystemMessage."""
        messages = [SystemMessage(content="You are a helpful assistant.")]
        litellm_messages = to_litellm_messages(messages)

        assert len(litellm_messages) == 1
        assert litellm_messages[0]["role"] == "system"
        assert litellm_messages[0]["content"] == "You are a helpful assistant."

    def test_to_litellm_user_message(self):
        """Test conversion of UserMessage."""
        messages = [UserMessage(content="Hello!", source="user")]
        litellm_messages = to_litellm_messages(messages)

        assert len(litellm_messages) == 1
        assert litellm_messages[0]["role"] == "user"
        assert litellm_messages[0]["content"] == "Hello!"

    def test_to_litellm_conversation(self):
        """Test conversion of multi-turn conversation."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!", source="user"),
            UserMessage(content="How are you?", source="user"),
        ]
        litellm_messages = to_litellm_messages(messages)

        assert len(litellm_messages) == 3
        assert litellm_messages[0]["role"] == "system"
        assert litellm_messages[1]["role"] == "user"
        assert litellm_messages[2]["role"] == "user"

    def test_litellm_to_model_output_basic(self):
        """Test conversion of basic LiteLLM response."""
        # Mock LiteLLM response — use spec=[] on the message to prevent
        # MagicMock auto-creating attributes like reasoning_content, which
        # confuses getattr(..., None) calls in litellm_to_autogen_result.
        mock_message = MagicMock()
        mock_message.content = "Hello! I'm doing well."
        mock_message.tool_calls = None
        mock_message.reasoning_content = None
        mock_message.thought = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.cached = False

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        result = litellm_to_model_output(mock_response, mock_usage, "gpt-4")

        assert result.content == "Hello! I'm doing well."
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.cached is False


class TestNullContentDefensiveParse:
    """Reasoning models (Gemini-3.x on native vertex_ai/) can exhaust the output
    budget on hidden thinking tokens, returning either a null `message` (the legacy
    compat-shim shape) OR a non-null `message` whose `.content` is None (the native
    vertex_ai/ shape). Both must coerce to "" so ModelOutput doesn't raise a pydantic
    ValidationError out of create().
    """

    def _usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 32
        return usage

    def test_null_message_coerced_to_empty_string(self):
        """choice.message is None (legacy shape) -> content == ''."""
        mock_choice = MagicMock()
        mock_choice.message = None
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.cached = False

        result = litellm_to_model_output(mock_response, self._usage(), "gemini-3.5-flash")
        assert result.content == ""
        assert result.finish_reason == "length"

    def test_non_null_message_with_null_content_coerced_to_empty_string(self):
        """choice.message is present but message.content is None (native vertex_ai/
        reasoning shape) -> content == '' (previously raised a pydantic ValidationError).
        """
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = None
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.cached = False

        result = litellm_to_model_output(mock_response, self._usage(), "gemini-3.5-flash")
        assert result.content == ""
        assert result.finish_reason == "length"


@pytest.mark.slow
@pytest.mark.anyio
class TestLiteLLMWrapperCreate:
    """Test LiteLLMWrapper.create() method."""

    async def test_create_basic_completion(self):
        """Test basic completion call."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4")

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
        with patch("litellm.acompletion") as mock_acompletion:
            mock_msg = MagicMock()
            mock_msg.content = "Hello!"
            mock_msg.tool_calls = None
            mock_msg.reasoning_content = None
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
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
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4")

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

        with patch("litellm.acompletion") as mock_acompletion:
            # First call fails with rate limit, second succeeds
            mock_msg = MagicMock()
            mock_msg.content = "Hello!"
            mock_msg.tool_calls = None
            mock_msg.reasoning_content = None
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_response.cached = False

            import litellm

            # litellm classifies retryable failures via its typed exception hierarchy;
            # the wrapper retries those (not arbitrary string-matched Exceptions).
            mock_acompletion.side_effect = [
                litellm.RateLimitError("Rate limit exceeded", llm_provider="openai", model="gpt-4"),
                mock_response,
            ]

            result = await wrapper.create(messages=messages)

            assert result.content == "Hello!"
            assert mock_acompletion.call_count == 2  # One failure + one success

    async def test_create_failure_after_max_retries(self):
        """Test that error is raised after max retries."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4")

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

        with patch("litellm.acompletion") as mock_acompletion:
            import litellm

            # Typed litellm RateLimitError is retryable; exhausting retries -> ProcessingError.
            mock_acompletion.side_effect = litellm.RateLimitError("Rate limit exceeded", llm_provider="openai", model="gpt-4")

            with pytest.raises(ProcessingError, match="LiteLLM call failed"):
                await wrapper.create(messages=messages)

            assert mock_acompletion.call_count == 2  # Initial + 1 retry


@pytest.mark.slow
@pytest.mark.anyio
class TestLiteLLMWrapperStructuredOutput:
    """Test structured output with LiteLLMWrapper."""

    async def test_create_with_schema(self):
        """Test structured output with Pydantic schema."""

        class TestSchema(BaseModel):
            model_config = ConfigDict(extra="forbid")

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

        with patch("litellm.acompletion") as mock_acompletion:
            mock_msg = MagicMock()
            mock_msg.content = '{"summary": "Test summary", "sentiment": "positive"}'
            mock_msg.tool_calls = None
            mock_msg.reasoning_content = None
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
            mock_response.cached = False

            mock_acompletion.return_value = mock_response

            result = await wrapper.create(messages=messages, schema=TestSchema)

            assert isinstance(result.parsed_object, TestSchema)
            assert result.parsed_object.summary == "Test summary"
            assert result.parsed_object.sentiment == "positive"


class TestLiteLLMWrapperPricing:
    """Test pricing calculation in LiteLLMWrapper."""

    def test_calculate_pricing_with_usage(self):
        """Test pricing calculation with valid usage data."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4")

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
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4")

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


class TestAddAnthropicCacheControl:
    """Tests for _add_anthropic_cache_control helper."""

    def test_adds_cache_control_to_string_system_content(self):
        """System message with string content is converted to content block with cache_control."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = _add_anthropic_cache_control(messages)

        sys_msg = result[0]
        assert sys_msg["role"] == "system"
        assert isinstance(sys_msg["content"], list)
        assert len(sys_msg["content"]) == 1
        block = sys_msg["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "You are a helpful assistant."
        assert block["cache_control"] == {"type": "ephemeral"}

        # User message unchanged
        assert result[1] == {"role": "user", "content": "Hello!"}

    def test_no_system_message_returns_unchanged(self):
        """Messages without a system role are returned unchanged."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _add_anthropic_cache_control(messages)
        assert result == messages

    def test_only_last_system_message_gets_cache_control(self):
        """When multiple system messages exist, only the last one gets cache_control."""
        messages = [
            {"role": "system", "content": "First system block."},
            {"role": "system", "content": "Second system block."},
            {"role": "user", "content": "Question"},
        ]
        result = _add_anthropic_cache_control(messages)

        # First system message: unchanged (still a string)
        assert result[0]["content"] == "First system block."

        # Last system message: converted to content block with cache_control
        last_sys = result[1]
        assert isinstance(last_sys["content"], list)
        assert last_sys["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_appends_cache_control_to_last_block(self):
        """System message with list content gets cache_control on the last block."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Block one."},
                    {"type": "text", "text": "Block two."},
                ],
            },
            {"role": "user", "content": "Hi"},
        ]
        result = _add_anthropic_cache_control(messages)

        content = result[0]["content"]
        assert len(content) == 2
        assert "cache_control" not in content[0]  # first block unchanged
        assert content[1]["cache_control"] == {"type": "ephemeral"}

    def test_existing_cache_control_not_overwritten(self):
        """A block that already has cache_control is left alone."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Cached.", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ]
        result = _add_anthropic_cache_control(messages)
        # Only one block, already has cache_control — should not be duplicated/changed
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Ensure no extra keys added
        assert list(result[0]["content"][0].keys()) == ["type", "text", "cache_control"]

    def test_system_block_ordering_preserved(self):
        """System messages remain before user messages after transformation."""
        messages = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Record content."},
        ]
        result = _add_anthropic_cache_control(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


@pytest.mark.slow
@pytest.mark.anyio
class TestAnthropicCacheControlInCreate:
    """Tests that LiteLLMWrapper.create() injects cache_control on Anthropic paths."""

    def _make_mock_response(self, text: str = "OK") -> MagicMock:
        mock_msg = MagicMock()
        mock_msg.content = text
        mock_msg.tool_calls = None
        mock_msg.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.cached = False
        return mock_response

    async def test_anthropic_path_adds_cache_control(self):
        """create() adds cache_control to the system message on the anthropic path."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="claude")
        wrapper = LiteLLMWrapper(
            model="claude-sonnet-4-6",
            model_info=model_info,
            litellm_model_name="claude-sonnet-4-6",
            api_key="test-key",
            client_type="anthropic",
        )
        messages = [
            SystemMessage(content="You are a judge. Evaluate the following."),
            UserMessage(content="The article content here.", source="user"),
        ]

        with patch("litellm.acompletion", return_value=self._make_mock_response()):
            await wrapper.create(messages=messages)
            import litellm

            call_messages = litellm.acompletion.call_args[1]["messages"]

        sys_msg = call_messages[0]
        assert sys_msg["role"] == "system"
        assert isinstance(sys_msg["content"], list), "System content must be a list of blocks for Anthropic caching"
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        # User message must not have cache_control
        assert "cache_control" not in call_messages[1]

    async def test_anthropic_vertex_path_adds_cache_control(self):
        """create() adds cache_control on the anthropic_vertex path."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="claude")
        wrapper = LiteLLMWrapper(
            model="claude-sonnet-4-6",
            model_info=model_info,
            litellm_model_name="vertex_ai/claude-sonnet-4-6",
            client_type="anthropic_vertex",
            vertex_project="my-project",
            vertex_location="us-east5",
        )
        messages = [
            SystemMessage(content="System instructions."),
            UserMessage(content="User content.", source="user"),
        ]

        with patch("litellm.acompletion", return_value=self._make_mock_response()):
            await wrapper.create(messages=messages)
            import litellm

            call_messages = litellm.acompletion.call_args[1]["messages"]

        sys_msg = call_messages[0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    async def test_gemini_path_no_cache_control(self):
        """create() does NOT add cache_control on the gemini path (implicit caching)."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gemini")
        wrapper = LiteLLMWrapper(
            model="gemini-2.5-flash",
            model_info=model_info,
            litellm_model_name="gemini/gemini-2.5-flash",
            client_type="gemini",
        )
        messages = [
            SystemMessage(content="You are a judge."),
            UserMessage(content="Content.", source="user"),
        ]

        with patch("litellm.acompletion", return_value=self._make_mock_response()):
            await wrapper.create(messages=messages)
            import litellm

            call_messages = litellm.acompletion.call_args[1]["messages"]

        sys_msg = call_messages[0]
        # System content stays as a plain string — no cache_control injection
        assert isinstance(sys_msg["content"], str)
        assert "cache_control" not in sys_msg

    async def test_openai_path_no_cache_control(self):
        """create() does NOT add cache_control on the openai path."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="gpt-4", structured_output=True)
        wrapper = LiteLLMWrapper(
            model="gpt-4",
            model_info=model_info,
            litellm_model_name="gpt-4",
            api_key="sk-test",
            client_type="openai",
        )
        messages = [
            SystemMessage(content="Instructions."),
            UserMessage(content="Content.", source="user"),
        ]

        with patch("litellm.acompletion", return_value=self._make_mock_response()):
            await wrapper.create(messages=messages)
            import litellm

            call_messages = litellm.acompletion.call_args[1]["messages"]

        sys_msg = call_messages[0]
        assert isinstance(sys_msg["content"], str)
        assert "cache_control" not in sys_msg

    async def test_system_before_user_on_anthropic_path(self):
        """System message remains first (before user/record content) on Anthropic path."""
        model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="claude")
        wrapper = LiteLLMWrapper(
            model="claude-sonnet-4-6",
            model_info=model_info,
            litellm_model_name="claude-sonnet-4-6",
            api_key="test-key",
            client_type="anthropic",
        )
        messages = [
            SystemMessage(content="System prompt with criteria."),
            UserMessage(content="Variable record content.", source="user"),
        ]

        with patch("litellm.acompletion", return_value=self._make_mock_response()):
            await wrapper.create(messages=messages)
            import litellm

            call_messages = litellm.acompletion.call_args[1]["messages"]

        assert call_messages[0]["role"] == "system"
        assert call_messages[1]["role"] == "user"
