"""Test structured tool handling improvements for issues #88 and #85."""

import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pytest

from buttermilk._core.contract import (
    AgentInput, 
    AgentOutput, 
    ConductorRequest,
    StepRequest,
    AgentTrace
)
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.agents.rag import RagAgent


class TestStructuredToolHandling:
    """Test the structured tool handling improvements."""

    @pytest.fixture
    def mock_participants(self):
        """Create mock participants dict."""
        return {
            "RESEARCHER": "Agent that searches and analyzes research data",
            "ANALYST": "Agent that performs data analysis",
        }

    @pytest.fixture
    def mock_participant_tools(self):
        """Create mock participant tools."""
        return {
            "RESEARCHER": [
                {
                    "name": "case_search_tool",
                    "description": "Search for case studies",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results to return"
                            }
                        },
                        "required": ["query"]
                    },
                    "output_schema": {"type": "object"}
                }
            ],
            "ANALYST": [
                {
                    "name": "analyze_data",
                    "description": "Analyze dataset",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "array"},
                            "method": {"type": "string"}
                        }
                    },
                    "output_schema": {"type": "object"}
                }
            ]
        }

    @pytest.mark.anyio
    async def test_tool_registration_from_conductor_request(self, mock_participants, mock_participant_tools):
        """Test that tools are properly registered when ConductorRequest is received."""
        # Create host agent
        host = StructuredLLMHostAgent(
            agent_id="test_host",
            agent_name="TestHost",
            role="HOST"
        )

        # Initialize with mock callback
        mock_callback = AsyncMock()
        await host.initialize(callback_to_groupchat=mock_callback)

        # Create ConductorRequest
        conductor_request = ConductorRequest(
            inputs={},
            participants=mock_participants,
            participant_tools=mock_participant_tools
        )

        # Handle the ConductorRequest
        await host._handle_events(
            message=conductor_request,
            cancellation_token=Mock(),
            public_callback=Mock()
        )

        # Verify tools were built
        assert hasattr(host, '_tools_list')
        assert len(host._tools_list) > 0

        # Check that tools have the expected names
        tool_names = [tool.name for tool in host._tools_list]
        assert "researcher.case_search_tool" in tool_names
        assert "analyst.analyze_data" in tool_names

    @pytest.mark.anyio
    async def test_tool_invocation_with_role_prefix(self, mock_participants, mock_participant_tools):
        """Test that tools with role prefix are correctly parsed and invoked."""
        # Create host agent
        host = StructuredLLMHostAgent(
            agent_id="test_host",
            agent_name="TestHost",
            role="HOST"
        )

        # Initialize
        mock_callback = AsyncMock()
        await host.initialize(callback_to_groupchat=mock_callback)

        # Set up participants and tools
        host._participants = mock_participants
        host._participant_tools = mock_participant_tools
        await host._build_agent_tools()

        # Simulate LLM response with tool call
        mock_llm_response = AgentTrace(
            agent_id="test_host",
            agent_type="host",
            agent_name="TestHost",
            content="I'll search for that information.",
            outputs={
                "tool_code": "researcher.case_search_tool",
                "parameters": {
                    "query": "test search query",
                    "max_results": 5
                }
            }
        )

        # Process the response
        await host._listen(
            message=mock_llm_response,
            cancellation_token=Mock(),
            source="",
            public_callback=Mock()
        )

        # Verify StepRequest was created correctly
        assert not host._proposed_step.empty()
        step_request = await host._proposed_step.get()

        assert step_request.role == "RESEARCHER"
        assert step_request.inputs["tool"] == "case_search_tool"
        assert step_request.inputs["query"] == "test search query"
        assert step_request.inputs["max_results"] == 5

    @pytest.mark.anyio
    async def test_tool_parameters_passed_correctly(self):
        """Test that tool parameters are passed correctly to agents (issue #85)."""
        # Create a mock RAG agent
        rag_agent = RagAgent(
            agent_id="test_rag",
            agent_name="TestRAG",
            role="RESEARCHER",
            parameters={}
        )

        # Create mock AgentOutput with structured result
        mock_output = AgentOutput(
            agent_id="test_rag",
            outputs="Here are the search results...",
            metadata={
                "query": "test search query",
                "total_results": 2
            }
        )

        # Mock the _process method
        with patch.object(rag_agent, '_process', new_callable=AsyncMock, return_value=mock_output) as mock_process:
            # Create StepRequest with query parameter
            step_request = StepRequest(
                role="RESEARCHER",
                inputs={
                    "query": "test search query",
                    "max_results": 10
                }
            )

            # Process the request
            result = await rag_agent._process(message=step_request)

            # Verify the result
            assert result.outputs == "Here are the search results..."
            assert result.metadata["query"] == "test search query"
            assert result.metadata["total_results"] == 2

            # Verify _process was called with correct message
            mock_process.assert_called_once()
            call_args = mock_process.call_args[1]
            assert call_args['message'] == step_request

    @pytest.mark.anyio
    async def test_error_handling_for_missing_participants(self):
        """Test error handling when participants list is empty."""
        # Create host agent
        host = StructuredLLMHostAgent(
            agent_id="test_host",
            agent_name="TestHost",
            role="HOST"
        )

        # Initialize without participants
        mock_callback = AsyncMock()
        await host.initialize(callback_to_groupchat=mock_callback)

        # Simulate LLM response with tool call
        mock_llm_response = AgentTrace(
            agent_id="test_host",
            agent_type="host",
            agent_name="TestHost",
            content="I'll search for that.",
            outputs={
                "tool_code": "researcher.case_search_tool",
                "parameters": {"query": "test"}
            }
        )

        # Process the response
        await host._listen(
            message=mock_llm_response,
            cancellation_token=Mock(),
            source="",
            public_callback=Mock()
        )

        # Verify error message was sent
        mock_callback.assert_called_once()
        error_trace = mock_callback.call_args[0][0]
        assert isinstance(error_trace, AgentTrace)
        assert "not currently available" in error_trace.content
        assert "Available agents are: none" in error_trace.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
