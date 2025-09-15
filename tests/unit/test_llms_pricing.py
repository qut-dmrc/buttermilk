"""Tests for pricing functionality in llms.py module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core.models import ModelInfo, RequestUsage

from buttermilk._core.llms import AutoGenWrapper, LLMs, ModelOutput
from buttermilk._core.types import UserMessage


class TestModelOutputPricing:
    """Test that ModelOutput properly includes pricing information."""
    
    def test_model_output_with_pricing_metadata(self):
        """Test ModelOutput can store pricing in metadata."""
        usage = RequestUsage(prompt_tokens=100, completion_tokens=50)
        
        model_output = ModelOutput(
            content="Test response",
            finish_reason="stop",
            usage=usage,
            cached=False
        )
        
        # Add pricing to metadata
        model_output.metadata = {
            "pricing": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_cost": 0.003
            }
        }
        
        assert model_output.metadata["pricing"]["total_cost"] == 0.003
        assert model_output.usage.prompt_tokens == 100
        assert model_output.usage.completion_tokens == 50


class TestAutoGenWrapperPricing:
    """Test pricing calculation in AutoGenWrapper."""

    @pytest.mark.anyio
    @patch("buttermilk._core.llms.calculate_token_cost")
    async def test_create_adds_pricing_to_metadata(self, mock_calculate_cost):
        """Test that create() adds pricing info to the result."""
        mock_calculate_cost.return_value = (100, 50, 0.003)

        # Create mock client that inherits from ChatCompletionClient
        from autogen_core.models import ChatCompletionClient
        mock_client = MagicMock(spec=ChatCompletionClient)
        mock_result = MagicMock()
        mock_result.content = "Test response"
        mock_result.finish_reason = "stop"
        mock_result.usage = RequestUsage(prompt_tokens=100, completion_tokens=50)
        mock_result.cached = False
        mock_result.thought = None
        mock_client.create.return_value = mock_result

        # Create proper model info
        model_info = ModelInfo(
            family="gpt-4",
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=False
        )

        # Create wrapper with proper model info
        wrapper = AutoGenWrapper(
            client=mock_client,
            model_info=model_info,
            litellm_model_name="openai/gpt-4"
        )
        
        # Call create
        messages = [UserMessage(content="Hello", source="user")]
        result = await wrapper.create(messages)
        
        # Verify result is ModelOutput with pricing
        assert isinstance(result, ModelOutput)
        assert hasattr(result, "metadata")
        assert "pricing" in result.metadata
        assert result.metadata["pricing"]["prompt_tokens"] == 100
        assert result.metadata["pricing"]["completion_tokens"] == 50
        assert result.metadata["pricing"]["total_cost"] == 0.003

    @pytest.mark.anyio
    @patch("buttermilk._core.llms.calculate_token_cost")
    async def test_call_chat_aggregates_tokens(self, mock_calculate_cost):
        """Test that call_chat() aggregates tokens from multiple calls."""
        # First call (with tools): 100 prompt, 50 completion
        # Second call (synthesis): 150 prompt, 75 completion
        mock_calculate_cost.side_effect = [
            (100, 50, 0.003),
            (150, 75, 0.004)
        ]
        
        # Create mock client that inherits from ChatCompletionClient
        from autogen_core.models import ChatCompletionClient
        mock_client = MagicMock(spec=ChatCompletionClient)

        # First result (tool call)
        from autogen_core import FunctionCall
        tool_call = FunctionCall(id="1", name="test_tool", arguments="{}")
        mock_result1 = MagicMock()
        mock_result1.content = [tool_call]
        mock_result1.finish_reason = "function_calls"
        mock_result1.usage = RequestUsage(prompt_tokens=100, completion_tokens=50)
        mock_result1.cached = False
        mock_result1.thought = None

        # Second result (synthesis)
        mock_result2 = MagicMock()
        mock_result2.content = "Final response"
        mock_result2.finish_reason = "stop"
        mock_result2.usage = RequestUsage(prompt_tokens=150, completion_tokens=75)
        mock_result2.cached = False
        mock_result2.thought = None

        mock_client.create.side_effect = [mock_result1, mock_result2]

        # Create proper model info
        model_info = ModelInfo(
            family="gpt-4",
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=False
        )

        # Create wrapper
        wrapper = AutoGenWrapper(
            client=mock_client,
            model_info=model_info,
            litellm_model_name="openai/gpt-4"
        )
        
        # Mock tool execution
        from autogen_core.models import FunctionExecutionResult
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.run_json = AsyncMock(return_value="tool result")
        mock_tool.return_value_as_string = MagicMock(return_value="tool result")

        with patch.object(wrapper, "_execute_tools", return_value=[FunctionExecutionResult(call_id="1", name="test_tool", content="tool result")]):
            # Call call_chat with tools
            messages = [UserMessage(content="Hello", source="user")]
            result = await wrapper.call_chat(
                messages=messages,
                cancellation_token=None,
                tools_list=[mock_tool]
            )
        
        # Verify aggregated pricing
        assert isinstance(result, ModelOutput)
        assert result.metadata["pricing"]["prompt_tokens"] == 250  # 100 + 150
        assert result.metadata["pricing"]["completion_tokens"] == 125  # 50 + 75
        assert result.metadata["pricing"]["total_cost"] == 0.007  # 0.003 + 0.004

    @pytest.mark.anyio
    async def test_create_handles_missing_usage(self):
        """Test create() handles responses without usage data gracefully."""
        # Create mock client that inherits from ChatCompletionClient
        from autogen_core.models import ChatCompletionClient
        mock_client = MagicMock(spec=ChatCompletionClient)
        mock_result = MagicMock()
        mock_result.content = "Test response"
        mock_result.finish_reason = "stop"
        mock_result.usage = RequestUsage(prompt_tokens=0, completion_tokens=0)  # Empty usage data
        mock_result.cached = False
        mock_result.thought = None
        mock_client.create.return_value = mock_result

        # Create proper model info
        model_info = ModelInfo(
            family="gpt-4",
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=False
        )

        # Create wrapper
        wrapper = AutoGenWrapper(
            client=mock_client,
            model_info=model_info,
            litellm_model_name="openai/gpt-4"
        )
        
        # Call create
        messages = [UserMessage(content="Hello", source="user")]
        result = await wrapper.create(messages)
        
        # Verify result has empty pricing
        assert isinstance(result, ModelOutput)
        assert result.metadata["pricing"]["prompt_tokens"] == 0
        assert result.metadata["pricing"]["completion_tokens"] == 0
        assert result.metadata["pricing"]["total_cost"] == 0.0


class TestLiteLLMModelNameResolution:
    """Test robust model name resolution for LiteLLM cost calculation."""

    def test_gemini_vertex_openai_resolution(self):
        """Test that Gemini models with vertex_openai client resolve correctly."""
        # This is the key test case that was failing before the fix
        result = LLMs.lookup_litellm_model_name("google/gemini-2.5-flash", "vertex_openai")
        assert result == "vertex_ai/google/gemini-2.5-flash"

        result = LLMs.lookup_litellm_model_name("google/gemini-2.5-pro", "vertex_openai")
        assert result == "vertex_ai/google/gemini-2.5-pro"

    def test_gemini_direct_api_resolution(self):
        """Test that Gemini models with direct API client stay as-is."""
        result = LLMs.lookup_litellm_model_name("gemini-2.5-flash", "gemini")
        assert result == "gemini-2.5-flash"

        result = LLMs.lookup_litellm_model_name("gemini-2.5-pro", "gemini")
        assert result == "gemini-2.5-pro"

    def test_anthropic_vertex_resolution(self):
        """Test Anthropic models on Vertex resolve correctly."""
        result = LLMs.lookup_litellm_model_name("claude-sonnet-4@20250514", "anthropic_vertex")
        assert result == "vertex_ai/claude-sonnet-4@20250514"

        result = LLMs.lookup_litellm_model_name("claude-opus-4-1", "anthropic_vertex")
        assert result == "vertex_ai/claude-opus-4-1"

    def test_anthropic_direct_api_resolution(self):
        """Test Anthropic models with direct API stay as-is."""
        result = LLMs.lookup_litellm_model_name("claude-3-5-sonnet-20241022", "anthropic")
        assert result == "claude-3-5-sonnet-20241022"

    def test_openai_azure_resolution(self):
        """Test OpenAI models on Azure resolve correctly."""
        result = LLMs.lookup_litellm_model_name("gpt-5-chat", "azure")
        assert result == "azure/gpt-5-chat"

        result = LLMs.lookup_litellm_model_name("gpt-5-nano", "azure")
        assert result == "azure/gpt-5-nano"

    def test_openai_direct_api_resolution(self):
        """Test OpenAI models with direct API stay as-is."""
        result = LLMs.lookup_litellm_model_name("gpt-4o", "openai")
        assert result == "gpt-4o"

    def test_llama_vertex_openai_resolution(self):
        """Test Llama models on Vertex OpenAI endpoint resolve correctly."""
        result = LLMs.lookup_litellm_model_name("meta/llama-4-maverick-17b-128e-instruct-maas", "vertex_openai")
        assert result == "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"

    def test_existing_prefix_handling(self):
        """Test models that already have provider prefixes are handled correctly."""
        # If a model already has the expected prefix, it should be returned as-is
        result = LLMs.lookup_litellm_model_name("vertex_ai/some-model", "vertex_openai")
        assert result == "vertex_ai/some-model"

        # If it has a different prefix, it should be preserved for cross-provider compatibility
        result = LLMs.lookup_litellm_model_name("openai/gpt-4", "azure")
        assert result == "azure/openai/gpt-4"

        # Test azure prefix with azure client type stays as-is
        result = LLMs.lookup_litellm_model_name("azure/gpt-4", "azure")
        assert result == "azure/gpt-4"

    def test_empty_or_none_model_names(self):
        """Test edge cases with empty or None model names."""
        result = LLMs.lookup_litellm_model_name("", "vertex_openai")
        assert result == ""

        result = LLMs.lookup_litellm_model_name(None, "vertex_openai")
        assert result is None

    def test_unknown_client_types_fallback(self):
        """Test that unknown client types get treated as fallback."""
        result = LLMs.lookup_litellm_model_name("some-model", "unknown_provider")
        assert result == "unknown_provider/some-model"

    def test_provider_prefix_mapping(self):
        """Test that _provider_prefix_for_client_type maps correctly."""
        assert LLMs._provider_prefix_for_client_type("azure") == "azure"
        assert LLMs._provider_prefix_for_client_type("openai") == "openai"
        assert LLMs._provider_prefix_for_client_type("gemini") == "gemini"
        assert LLMs._provider_prefix_for_client_type("gemini_vertex") == "gemini"
        assert LLMs._provider_prefix_for_client_type("vertex_openai") == "vertex_ai"
        assert LLMs._provider_prefix_for_client_type("anthropic_vertex") == "vertex_ai"
        assert LLMs._provider_prefix_for_client_type("anthropic") == "anthropic"

    def test_base_model_name_extraction(self):
        """Test that _extract_base_model_name handles various patterns."""
        # For vertex_openai with google/ models, preserve the google/ prefix
        result = LLMs._extract_base_model_name("google/gemini-2.5-flash", "vertex_openai")
        assert result == "google/gemini-2.5-flash"

        # For other cases, strip mismatched prefixes
        result = LLMs._extract_base_model_name("azure/gpt-4", "openai")
        assert result == "azure/gpt-4"  # Keep full name for cross-provider compatibility

        # No prefix found, return as-is
        result = LLMs._extract_base_model_name("gpt-4", "openai")
        assert result == "gpt-4"

    def test_real_world_model_registry_examples(self):
        """Test with real model names from the model registry."""
        # Test current gemini models that were causing issues
        result = LLMs.lookup_litellm_model_name("google/gemini-2.5-flash", "vertex_openai")
        assert result == "vertex_ai/google/gemini-2.5-flash"

        result = LLMs.lookup_litellm_model_name("google/gemini-2.5-pro", "vertex_openai")
        assert result == "vertex_ai/google/gemini-2.5-pro"

        # Test Llama model
        result = LLMs.lookup_litellm_model_name("meta/llama-4-maverick-17b-128e-instruct-maas", "vertex_openai")
        assert result == "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"

        # Test Azure models
        result = LLMs.lookup_litellm_model_name("gpt-5-chat", "azure")
        assert result == "azure/gpt-5-chat"

        # Test Anthropic on Vertex
        result = LLMs.lookup_litellm_model_name("claude-sonnet-4@20250514", "anthropic_vertex")
        assert result == "vertex_ai/claude-sonnet-4@20250514"
