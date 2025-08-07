"""Tests for token/cost tracking in message service."""

import datetime
from unittest.mock import patch, MagicMock

import pytest

from buttermilk._core import AgentConfig
from buttermilk._core.contract import AgentOutput, AgentTrace, AgentInput, ErrorEvent
from buttermilk.api.services.message_service import MessageService, ChatMessage


class TestMessageServiceTokenExtraction:
    """Test token and cost extraction in MessageService."""

    @patch('buttermilk.api.services.message_service.calculate_token_cost')
    @patch('buttermilk.api.services.message_service.extract_usage_from_metadata')
    def test_format_message_extracts_tokens_from_agent_output(self, mock_extract, mock_calculate):
        """Test that tokens are extracted from AgentOutput metadata."""
        # Setup mocks
        mock_extract.return_value = {
            "prompt_tokens": 100,
            "completion_tokens": 50
        }
        mock_calculate.return_value = (100, 50, 0.003)
        
        # Create test AgentOutput with usage metadata
        agent_config = AgentConfig(
            name="test_agent",
            parameters={"model": "gpt41"}
        )
        
        # Create AgentOutput with AssistantMessage as output
        from buttermilk._core.types import AssistantMessage
        
        agent_output = AgentOutput(
            agent_id="test_agent",
            outputs=AssistantMessage(content="Test response", source="test_agent"),
            metadata={
                "agent_model": "gpt41",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50
                }
            }
        )
        
        # Mock getattr to return agent_config when 'agent_info' is requested
        original_getattr = getattr
        def mock_getattr(obj, name, default=None):
            if obj is agent_output and name == 'agent_info':
                return agent_config
            return original_getattr(obj, name, default)
        
        with patch('builtins.getattr', mock_getattr):
            # Format the message
            result = MessageService.format_message_for_client(agent_output)
            
            # Verify token extraction
            assert isinstance(result, ChatMessage)
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert result.cost_usd == 0.003
            
            # Verify mocks were called correctly
            mock_extract.assert_called_once()
            mock_calculate.assert_called_once_with(
                model="gpt41",
                usage_dict={"prompt_tokens": 100, "completion_tokens": 50}
            )

    @patch('buttermilk.api.services.message_service.calculate_token_cost')
    @patch('buttermilk.api.services.message_service.extract_usage_from_metadata')
    def test_format_message_extracts_tokens_from_agent_trace(self, mock_extract, mock_calculate):
        """Test that tokens are extracted from AgentTrace metadata."""
        mock_extract.return_value = {
            "input_tokens": 200,
            "output_tokens": 75
        }
        mock_calculate.return_value = (200, 75, 0.005)
        
        agent_config = AgentConfig(
            name="test_agent",
            parameters={"model": "sonnet"}
        )
        
        agent_input = AgentInput(
            inputs={},
            parameters={},
            context=[],
            records=[]
        )
        
        from buttermilk._core.types import AssistantMessage
        
        agent_trace = AgentTrace(
            agent_id="test_agent",
            agent_info=agent_config,
            inputs=agent_input,
            outputs=AssistantMessage(content="Test response", source="test_agent"),
            metadata={
                "agent_model": "sonnet",
                "outputs": {
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 75
                    }
                }
            }
        )
        
        result = MessageService.format_message_for_client(agent_trace)
        
        assert isinstance(result, ChatMessage)
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 75
        assert result.cost_usd == 0.005

    def test_format_message_no_usage_data(self):
        """Test handling when no usage data is available."""
        from buttermilk._core.types import AssistantMessage
        
        agent_output = AgentOutput(
            agent_id="test_agent",
            outputs=AssistantMessage(content="Test response", source="test_agent"),
            metadata={"some_field": "value"}  # No usage data
        )
        
        result = MessageService.format_message_for_client(agent_output)
        
        assert isinstance(result, ChatMessage)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.cost_usd == 0.0

    def test_format_message_with_error(self):
        """Test handling of error messages with token data."""
        agent_config = AgentConfig(
            name="test_agent",
            parameters={}
        )
        
        error_event = ErrorEvent(
            source="test_agent",
            content="Test error"
        )
        
        agent_output = AgentOutput(
            agent_id="test_agent",
            outputs=None,
            error=[error_event]
        )
        
        # Mock getattr to return agent_config when 'agent_info' is requested
        original_getattr = getattr
        def mock_getattr(obj, name, default=None):
            if obj is agent_output and name == 'agent_info':
                return agent_config
            return original_getattr(obj, name, default)
        
        with patch('builtins.getattr', mock_getattr):
            result = MessageService.format_message_for_client(agent_output)
            
            assert isinstance(result, ChatMessage)
            assert result.type == "system_error"
            assert result.prompt_tokens == 0
            assert result.completion_tokens == 0
            assert result.cost_usd == 0.0

    @patch('buttermilk.api.services.message_service.calculate_token_cost')
    def test_format_message_model_from_metadata(self, mock_calculate):
        """Test extracting model name from metadata when not in agent_info."""
        from buttermilk._core.types import AssistantMessage
        
        mock_calculate.return_value = (50, 25, 0.001)
        
        agent_output = AgentOutput(
            agent_id="test_agent",
            outputs=AssistantMessage(content="Test response", source="test_agent"),
            metadata={
                "agent_model": "gpt41mini",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25
                }
            }
        )
        
        result = MessageService.format_message_for_client(agent_output)
        
        assert isinstance(result, ChatMessage)
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 25
        assert result.cost_usd == 0.001
        
        mock_calculate.assert_called_with(
            model="gpt41mini",
            usage_dict={"prompt_tokens": 50, "completion_tokens": 25}
        )