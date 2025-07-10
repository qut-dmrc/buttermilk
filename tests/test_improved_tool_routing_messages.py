"""Test improved tool routing messages in HostAgent and StructuredLLMHostAgent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from autogen_core import FunctionCall

from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.agents.flowcontrol.host import HostAgent


class TestImprovedToolRoutingMessages:
    """Test that tool routing produces better status messages."""

    def test_host_describe_tool_call(self):
        """Test the _describe_tool_call method in HostAgent."""
        host = HostAgent(agent_name="test_host", role="HOST")

        # Test with query argument
        desc = host._describe_tool_call("search", {"query": "What is the weather in Paris today?"})
        assert desc == "search('What is the weather in Paris today?')"

        # Test with long query (should truncate)
        desc = host._describe_tool_call("search", {"query": "This is a very long query that should be truncated after 40 characters"})
        assert desc == "search('This is a very long query that should be...')"

        # Test with message argument
        desc = host._describe_tool_call("send_message", {"message": "Hello, world!"})
        assert desc == "send_message('Hello, world!')"

        # Test with content argument
        desc = host._describe_tool_call("process", {"content": "Some content here"})
        assert desc == "process('Some content here')"

        # Test with target argument
        desc = host._describe_tool_call("route", {"target": "agent.method"})
        assert desc == "route(target='agent.method')"

        # Test with nested inputs
        desc = host._describe_tool_call("complex_tool", {"inputs": {"query": "nested query"}})
        assert desc == "complex_tool(query='nested query')"

        # Test with other arguments
        desc = host._describe_tool_call("compute", {"value": 42, "operation": "multiply"})
        assert desc == "compute(value='42')"

        # Test fallback
        desc = host._describe_tool_call("empty_tool", {})
        assert desc == "empty_tool(0 args)"

    def test_structured_llm_host_tool_call_summary(self):
        """Test the _create_tool_call_summary method in StructuredLLMHostAgent."""
        agent = StructuredLLMHostAgent(
            agent_name="test_structured",
            role="STRUCTURED_HOST",
            model_name="test-model",
            parameters={"model": "test-model"}
        )

        # Test empty list
        summary = agent._create_tool_call_summary([])
        assert summary == "No tool calls requested"

        # Test single tool call with query
        tool_call = FunctionCall(
            id="1",
            name="search_database",
            arguments='{"query": "Find all users with active subscriptions"}'
        )
        summary = agent._create_tool_call_summary([tool_call])
        assert summary == "Searching for: Find all users with active subscriptions"

        # Test single tool call with long query (should truncate)
        tool_call = FunctionCall(
            id="2",
            name="search",
            arguments='{"query": "This is a very long search query that should be truncated after 50 characters to keep the message concise"}'
        )
        summary = agent._create_tool_call_summary([tool_call])
        assert summary == "Searching for: This is a very long search query that should be tr..."

        # Test single tool call with message
        tool_call = FunctionCall(
            id="3",
            name="send_notification",
            arguments='{"message": "System maintenance scheduled"}'
        )
        summary = agent._create_tool_call_summary([tool_call])
        assert summary == "Processing: System maintenance scheduled"

        # Test single tool call with target
        tool_call = FunctionCall(
            id="4",
            name="invoke",
            arguments='{"target": "data_processor.analyze"}'
        )
        summary = agent._create_tool_call_summary([tool_call])
        assert summary == "Targeting data_processor.analyze with invoke"

        # Test single generic tool call
        tool_call = FunctionCall(
            id="5",
            name="calculate_metrics",
            arguments='{"start_date": "2024-01-01", "end_date": "2024-01-31"}'
        )
        summary = agent._create_tool_call_summary([tool_call])
        assert summary == "Calling calculate_metrics"

        # Test multiple calls to same tool
        tool_calls = [
            FunctionCall(id="6", name="fetch_data", arguments='{"id": 1}'),
            FunctionCall(id="7", name="fetch_data", arguments='{"id": 2}'),
            FunctionCall(id="8", name="fetch_data", arguments='{"id": 3}')
        ]
        summary = agent._create_tool_call_summary(tool_calls)
        assert summary == "Making 3 fetch_data calls"

        # Test multiple different tools (few)
        tool_calls = [
            FunctionCall(id="9", name="search", arguments='{}'),
            FunctionCall(id="10", name="analyze", arguments='{}'),
            FunctionCall(id="11", name="report", arguments='{}')
        ]
        summary = agent._create_tool_call_summary(tool_calls)
        assert summary == "Calling: search, analyze, report"

        # Test many different tools
        tool_calls = [
            FunctionCall(id=str(i), name=f"tool_{i}", arguments='{}')
            for i in range(12, 20)
        ]
        summary = agent._create_tool_call_summary(tool_calls)
        assert summary == "Orchestrating 8 tool calls across 8 tools"

    @pytest.mark.anyio
    async def test_structured_llm_host_integration(self):
        """Test that StructuredLLMHostAgent produces better output messages."""
        agent = StructuredLLMHostAgent(
            agent_name="test_structured",
            role="STRUCTURED_HOST",
            model_name="test-model",
            parameters={"model": "test-model"}
        )

        # Mock the tool schemas method
        with patch.object(agent, '_get_tool_schemas', return_value=[]):

            # Mock the LLM to return tool calls
            mock_create_result = Mock()
            mock_create_result.content = [
                FunctionCall(
                    id="1",
                    name="search_knowledge_base",
                    arguments='{"query": "What are the latest AI developments?"}'
                ),
                FunctionCall(
                    id="2",
                    name="analyze_sentiment",
                    arguments='{"text": "The results look promising"}'
                )
            ]

            # Mock the necessary methods
            with patch.object(agent, '_model_client') as mock_client, \
                 patch.object(agent, '_route_tool_calls_to_agents', new_callable=AsyncMock):

                mock_client.create = AsyncMock(return_value=mock_create_result)

                # Process a message
                from buttermilk._core.contract import AgentInput
                message = AgentInput(inputs={"content": "Test input"})

                result = await agent._process(message=message)

                # Check that the output message is more descriptive
                assert result.outputs in [
                    "Calling: search_knowledge_base, analyze_sentiment",
                    "Searching for: What are the latest AI developments?"  # If it processes them one by one
                ]
                assert result.metadata["tool_calls"] == 2
