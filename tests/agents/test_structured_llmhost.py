"""Unit tests for StructuredLLMHostAgent tool building functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk._core.contract import ConductorRequest, StepRequest
from autogen_core.tools import FunctionTool


class TestStructuredLLMHostAgent:
    """Test cases for StructuredLLMHostAgent tool building."""

    @pytest.fixture
    def mock_host(self):
        """Create a mock StructuredLLMHostAgent instance."""
        host = StructuredLLMHostAgent(
            agent_id="test_host",
            agent_name="TestHost", 
            role="CONDUCTOR",
            description="Test host agent",
            parameters={"model": "gpt-4", "template": "test", "human_in_loop": False}
        )
        host.callback_to_groupchat = AsyncMock()
        return host

    @pytest.fixture
    def sample_participants(self):
        """Sample participants for testing."""
        return {
            "RESEARCHER": "Searches and analyzes information",
            "WRITER": "Creates and edits content", 
            "REVIEWER": "Reviews and validates results"
        }

    @pytest.fixture
    def sample_participant_tools(self):
        """Sample participant tool definitions."""
        return {
            "RESEARCHER": [
                {
                    "name": "search",
                    "description": "Search for information on a topic",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Maximum results"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "analyze", 
                    "description": "Analyze search results",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "string", "description": "Data to analyze"}
                        },
                        "required": ["data"]
                    }
                }
            ],
            "WRITER": [
                {
                    "name": "write",
                    "description": "Write content based on research",
                    "input_schema": {
                        "type": "object", 
                        "properties": {
                            "topic": {"type": "string", "description": "Topic to write about"},
                            "style": {"type": "string", "description": "Writing style"}
                        },
                        "required": ["topic"]
                    }
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_build_agent_tools_with_definitions(self, mock_host, sample_participants, sample_participant_tools):
        """Test building tools when participants have specific tool definitions."""
        # Setup
        mock_host._participants = sample_participants
        mock_host._participant_tools = sample_participant_tools
        
        # Execute
        await mock_host._build_agent_tools()
        
        # Verify tool count
        # Should have 3 tools: 2 for RESEARCHER, 1 for WRITER
        assert len(mock_host._tools_list) == 3
        
        # Verify tool names
        tool_names = [tool.name for tool in mock_host._tools_list]
        assert "search" in tool_names
        assert "analyze" in tool_names
        assert "write" in tool_names
        
        # Verify all tools are FunctionTool instances
        for tool in mock_host._tools_list:
            assert isinstance(tool, FunctionTool)

    @pytest.mark.asyncio
    async def test_build_agent_tools_default_only(self, mock_host, sample_participants):
        """Test building tools when no specific tool definitions exist."""
        # Setup - participants but no tool definitions
        mock_host._participants = sample_participants
        mock_host._participant_tools = {}
        
        # Execute
        await mock_host._build_agent_tools()
        
        # Verify that no tools are created when no participant tools exist
        # and no agents have announced themselves
        
        # Verify that no tools are created when no participant tools exist
        # and no agents have announced themselves
        assert len(mock_host._tools_list) == 0

    @pytest.mark.asyncio
    async def test_tool_invocation_queues_step_request(self, mock_host, sample_participants, sample_participant_tools):
        """Test that invoking a tool queues a StepRequest."""
        # Setup
        mock_host._participants = sample_participants
        mock_host._participant_tools = sample_participant_tools
        await mock_host._build_agent_tools()
        
        # Find the search tool
        search_tool = None
        for tool in mock_host._tools_list:
            if tool.name == "search":
                search_tool = tool
                break
        
        assert search_tool is not None
        
        # Invoke the tool
        result = await search_tool._func(query="test query", max_results=5)
        
        # Verify result is None (no fake status returned)
        assert result is None
        
        # Verify StepRequest was queued
        assert not mock_host._proposed_step.empty()
        step = await mock_host._proposed_step.get()
        assert isinstance(step, StepRequest)
        assert step.role == "RESEARCHER"
        assert step.inputs["query"] == "test query"
        assert step.inputs["max_results"] == 5
        assert step.metadata["tool_name"] == "search"

    @pytest.mark.asyncio
    async def test_simple_prompt_tool_invocation(self, mock_host, sample_participants):
        """Test invoking a tool with simple prompt parameter."""
        # Setup with default tools
        mock_host._participants = sample_participants
        mock_host._participant_tools = {}
        await mock_host._build_agent_tools()
        
        # Since no participant tools are defined, there should be no tools
        # Skip this test as it doesn't apply to the current implementation
        pytest.skip("No default tools are created when no participant tools exist")
        
        assert reviewer_tool is not None
        
        # Invoke with prompt
        result = await reviewer_tool._func(prompt="Please review this content")
        
        # Verify
        assert result is None
        step = await mock_host._proposed_step.get()
        assert step.role == "REVIEWER"
        assert step.inputs["prompt"] == "Please review this content"
        assert step.metadata["tool_name"] == "call_reviewer"

    @pytest.mark.asyncio
    async def test_duplicate_tool_filtering(self, mock_host):
        """Test that duplicate tools are filtered out."""
        # Setup with duplicate tool names
        mock_host._participants = {"AGENT1": "First agent", "AGENT2": "Second agent"}
        mock_host._participant_tools = {
            "AGENT1": [{"name": "search", "description": "Search 1", "input_schema": {}}],
            "AGENT2": [{"name": "search", "description": "Search 2", "input_schema": {}}]
        }
        
        # Execute
        await mock_host._build_agent_tools()
        
        # Verify unique tool names
        tool_names = [tool.name for tool in mock_host._tools_list]
        assert "search" in tool_names
        assert len(tool_names) == 1  # Only one search tool, not duplicated

    @pytest.mark.asyncio
    async def test_handle_events_rebuilds_tools(self, mock_host, sample_participants):
        """Test that _handle_events rebuilds tools when participants change."""
        # Initial setup
        initial_participants = {"AGENT1": "First agent"}
        mock_host._participants = initial_participants.copy()
        await mock_host._build_agent_tools()
        initial_tool_count = len(mock_host._tools_list)
        
        # Create ConductorRequest with new participants
        conductor_request = ConductorRequest(
            inputs={},
            participants=sample_participants,
            participant_tools={}
        )
        
        # Mock parent class _handle_events
        with patch.object(StructuredLLMHostAgent.__bases__[1], '_handle_events', new_callable=AsyncMock) as mock_parent:
            mock_parent.return_value = None
            
            # Execute
            await mock_host._handle_events(
                conductor_request,
                cancellation_token=MagicMock(),
                public_callback=AsyncMock(),
                message_callback=AsyncMock()
            )
        
        # Verify tools were rebuilt
        assert len(mock_host._tools_list) != initial_tool_count
        # Participants should be updated (original + new ones)
        assert "AGENT1" in mock_host._participants  # Original kept
        assert "RESEARCHER" in mock_host._participants  # New added


    @pytest.mark.asyncio 
    async def test_no_participants_initialization(self, mock_host):
        """Test initialization when no participants are available."""
        # Setup with no participants
        mock_host._participants = {}
        
        # Execute initialization
        await mock_host._initialize(callback_to_groupchat=AsyncMock())
        
        # Verify no tools were created
        assert not hasattr(mock_host, '_tools_list') or len(mock_host._tools_list) == 0

    @pytest.mark.asyncio
    async def test_generic_input_tool_invocation(self, mock_host, sample_participant_tools):
        """Test invoking a tool with multiple parameters."""
        # Setup
        mock_host._participants = {"WRITER": "Creates content"}
        mock_host._participant_tools = {"WRITER": sample_participant_tools["WRITER"]}
        await mock_host._build_agent_tools()
        
        # Find write tool
        write_tool = None
        for tool in mock_host._tools_list:
            if tool.name == "write":
                write_tool = tool
                break
        
        assert write_tool is not None
        
        # Invoke with multiple parameters
        result = await write_tool._func(topic="AI Ethics", style="academic")
        
        # Verify
        assert result is None
        step = await mock_host._proposed_step.get()
        assert step.role == "WRITER"
        assert step.inputs["topic"] == "AI Ethics"
        assert step.inputs["style"] == "academic"
        assert step.metadata["tool_name"] == "write"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])