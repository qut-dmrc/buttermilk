"""Tests for pricing functionality in llms.py module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core.models import RequestUsage

from buttermilk._core.llms import AutoGenWrapper, ModelOutput
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
        
        # Create mock client
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "Test response"
        mock_result.finish_reason = "stop"
        mock_result.usage = RequestUsage(prompt_tokens=100, completion_tokens=50)
        mock_result.cached = False
        mock_result.thought = None
        mock_client.create.return_value = mock_result
        
        # Create wrapper with model info
        wrapper = AutoGenWrapper(
            client=mock_client,
            model_info={"model": "gpt-4", "structured_output": False}
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
        
        # Create mock client
        mock_client = AsyncMock()
        
        # First result (tool call)
        from autogen_core import FunctionCall
        tool_call = FunctionCall(id="1", name="test_tool", arguments="{}")
        mock_result1 = MagicMock()
        mock_result1.content = [tool_call]
        mock_result1.finish_reason = "tool_calls"
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
        
        # Create wrapper
        wrapper = AutoGenWrapper(
            client=mock_client,
            model_info={"model": "gpt-4", "structured_output": False}
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
        # Create mock client without usage data
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "Test response"
        mock_result.finish_reason = "stop"
        mock_result.usage = None  # No usage data
        mock_result.cached = False
        mock_result.thought = None
        mock_client.create.return_value = mock_result
        
        # Create wrapper
        wrapper = AutoGenWrapper(
            client=mock_client,
            model_info={"model": "gpt-4", "structured_output": False}
        )
        
        # Call create
        messages = [UserMessage(content="Hello", source="user")]
        result = await wrapper.create(messages)
        
        # Verify result has empty pricing
        assert isinstance(result, ModelOutput)
        assert result.metadata["pricing"]["prompt_tokens"] == 0
        assert result.metadata["pricing"]["completion_tokens"] == 0
        assert result.metadata["pricing"]["total_cost"] == 0.0
