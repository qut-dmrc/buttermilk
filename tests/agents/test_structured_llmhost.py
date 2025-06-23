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
        host.set_testing_mode(True)  # Enable testing mode for unit tests
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
        # Should have 4 tools: 2 for RESEARCHER, 1 for WRITER, 1 default for REVIEWER
        assert len(mock_host._tools_list) == 4
        
        # Verify tool names
        tool_names = [tool.name for tool in mock_host._tools_list]
        assert "researcher.search" in tool_names
        assert "researcher.analyze" in tool_names
        assert "writer.write" in tool_names
        assert "reviewer.call_reviewer" in tool_names  # Default tool
        
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
        
        # Verify tool count - should have 3 default tools
        assert len(mock_host._tools_list) == 3
        
        # Verify default tool names
        tool_names = [tool.name for tool in mock_host._tools_list]
        assert "researcher.call_researcher" in tool_names
        assert "writer.call_writer" in tool_names
        assert "reviewer.call_reviewer" in tool_names

    @pytest.mark.asyncio
    async def test_tool_invocation_queues_step_request(self, mock_host, sample_participants, sample_participant_tools):
        """Test that invoking a tool queues a StepRequest."""
        # Setup
        mock_host._participants = sample_participants
        mock_host._participant_tools = sample_participant_tools
        await mock_host._build_agent_tools()
        
        # Find the researcher.search tool
        search_tool = None
        for tool in mock_host._tools_list:
            if tool.name == "researcher.search":
                search_tool = tool
                break
        
        assert search_tool is not None
        
        # Invoke the tool
        result = await search_tool._func(query="test query", max_results=5)
        
        # Verify result
        assert result == {"status": "queued", "step": "RESEARCHER"}
        
        # Verify StepRequest was queued
        assert not mock_host._proposed_step.empty()
        step = await mock_host._proposed_step.get()
        assert isinstance(step, StepRequest)
        assert step.role == "RESEARCHER"
        assert step.inputs["tool"] == "search"
        assert step.inputs["tool_inputs"]["query"] == "test query"
        assert step.inputs["tool_inputs"]["max_results"] == 5

    @pytest.mark.asyncio
    async def test_simple_prompt_tool_invocation(self, mock_host, sample_participants):
        """Test invoking a tool with simple prompt parameter."""
        # Setup with default tools
        mock_host._participants = sample_participants
        mock_host._participant_tools = {}
        await mock_host._build_agent_tools()
        
        # Find a default tool
        reviewer_tool = None
        for tool in mock_host._tools_list:
            if tool.name == "reviewer.call_reviewer":
                reviewer_tool = tool
                break
        
        assert reviewer_tool is not None
        
        # Invoke with prompt
        result = await reviewer_tool._func(prompt="Please review this content")
        
        # Verify
        assert result == {"status": "queued", "step": "REVIEWER"}
        step = await mock_host._proposed_step.get()
        assert step.role == "REVIEWER"
        assert step.inputs["tool"] == "call_reviewer"
        assert step.inputs["tool_inputs"]["prompt"] == "Please review this content"

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
        assert "agent1.search" in tool_names
        assert "agent2.search" in tool_names
        assert len(tool_names) == len(set(tool_names))  # No duplicates

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
        
        # Find writer.write tool
        write_tool = None
        for tool in mock_host._tools_list:
            if tool.name == "writer.write":
                write_tool = tool
                break
        
        assert write_tool is not None
        
        # Invoke with multiple parameters
        result = await write_tool._func(topic="AI Ethics", style="academic")
        
        # Verify
        assert result == {"status": "queued", "step": "WRITER"}
        step = await mock_host._proposed_step.get()
        assert step.role == "WRITER"
        assert step.inputs["tool"] == "write"
        assert step.inputs["tool_inputs"]["topic"] == "AI Ethics"
        assert step.inputs["tool_inputs"]["style"] == "academic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])