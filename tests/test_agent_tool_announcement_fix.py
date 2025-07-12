"""Test the fix for agent tool announcement behavior.

This test validates that agents with @tool decorators properly announce their
actual tool definitions instead of generic agent tools, resolving the
"kwargs Field required" validation error.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement, AgentInput, AgentOutput
from buttermilk._core.mcp_decorators import tool
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.agents.rag import RagAgent


class MockToolAgent(Agent):
    """Mock agent with @tool decorated methods for testing."""
    
    def __init__(self, **data):
        super().__init__(**data)
        
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        """Simple implementation for testing."""
        return AgentOutput(
            outputs="Test output",
            role=self.role,
            agent_id=self.agent_id
        )
    
    @tool(name="analyze", description="Analyze the given data")
    async def analyze(self, data: str, format: str = "json") -> dict[str, Any]:
        """Mock analysis tool."""
        return {"analysis": f"analyzed {data} in {format} format"}
    
    @tool(name="summarize", description="Summarize the given text")
    async def summarize(self, text: str, max_length: int = 100) -> str:
        """Mock summarization tool."""
        return f"Summary of {text[:max_length]}"


class TestAgentToolAnnouncementFix:
    """Test suite for the agent tool announcement fix."""

    @pytest.fixture
    def mock_tool_agent(self):
        """Create a mock agent with @tool decorators."""
        config = {
            "role": "ANALYST",
            "description": "Test analyst agent",
            "parameters": {"model": "test-model"},
            "unique_identifier": "test123"
        }
        return MockToolAgent(**config)

    @pytest.fixture
    def enhanced_rag_agent(self):
        """Create an EnhancedRagAgent for testing."""
        config = AgentConfig(
            role="POLICY_ANALYST",
            agent_obj="EnhancedRagAgent",
            description="Policy analysis agent",
            parameters={"model": "gpt-4o-mini"}
        )
        return EnhancedRagAgent(**config.model_dump())

    def test_agent_extracts_tool_definitions_from_decorators(self, mock_tool_agent):
        """Test that agents extract tool definitions from @tool decorators."""
        tool_definitions = mock_tool_agent.get_tool_definitions()

        assert len(tool_definitions) == 2

        # Check first tool (analyze)
        analyze_tool = tool_definitions[0]
        assert isinstance(analyze_tool, AgentToolDefinition)
        assert analyze_tool.name == "analyze"
        assert analyze_tool.description == "Analyze the given data"

        # Check tool schema format
        schema = analyze_tool.schema["function"]
        assert schema["name"] == "analyze"
        assert schema["description"] == "Analyze the given data"
        assert "parameters" in schema
        assert "data" in schema["parameters"]["properties"]
        assert "format" in schema["parameters"]["properties"]

        # Check second tool (summarize)
        summarize_tool = tool_definitions[1]
        assert summarize_tool.name == "summarize"
        assert summarize_tool.description == "Summarize the given text"

    def test_agent_announcement_uses_actual_tool_definitions(self, mock_tool_agent):
        """Test that agent announcements use actual @tool definitions instead of generic ones."""
        # Create announcement
        announcement = mock_tool_agent.create_announcement("initial", "joining")

        # Should use the first @tool method instead of generic agent tool
        assert announcement.tool_definition["name"] == "analyze"
        assert announcement.tool_definition["description"] == "Analyze the given data"
        assert "data" in announcement.tool_definition["parameters"]["properties"]
        assert "format" in announcement.tool_definition["parameters"]["properties"]

        # Should not be the generic "call_analyst" tool
        assert announcement.tool_definition["name"] != "call_analyst"

    def test_agent_falls_back_to_generic_tool_when_no_decorators(self):
        """Test that agents fall back to generic tool when no @tool decorators exist."""

        class NoToolAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(outputs="test", role=self.role, agent_id=self.agent_id)

        config = {
            "role": "BASIC",
            "description": "Basic agent without tools",
            "parameters": {"model": "test-model"}
        }
        agent = NoToolAgent(**config)

        # Should fall back to generic tool definition
        announcement = agent.create_announcement("initial", "joining")
        assert announcement.tool_definition["name"] == "call_basic"
        assert "prompt" in announcement.tool_definition["input_schema"]["properties"]

    def test_enhanced_rag_agent_announces_search_tool(self, enhanced_rag_agent):
        """Test that EnhancedRagAgent announces its @tool(name='search') method."""
        # Get tool definitions
        tool_definitions = enhanced_rag_agent.get_tool_definitions()
        assert len(tool_definitions) == 1

        search_tool = tool_definitions[0]
        assert search_tool.name == "search"
        assert search_tool.description == "Search the knowledge base with an intelligent RAG system"

        # Check tool schema has correct parameters
        schema = search_tool.schema["function"]
        assert "query" in schema["parameters"]["properties"]
        assert "max_results" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]

        # Check announcement uses the search tool
        announcement = enhanced_rag_agent.create_announcement("initial", "joining")
        assert announcement.tool_definition["name"] == "search"
        assert announcement.tool_definition["description"] == "Search the knowledge base with an intelligent RAG system"

    @pytest.mark.anyio
    async def test_structured_llmhost_builds_tools_from_announcements(self, enhanced_rag_agent):
        """Test that StructuredLLMHostAgent builds tools from agent announcements."""
        # Create host agent
        host_config = AgentConfig(
            role="HOST",
            agent_obj="StructuredLLMHostAgent", 
            description="Test host",
            parameters={"model": "gpt-4o-mini"}
        )
        host_agent = StructuredLLMHostAgent(**host_config.model_dump())

        # Get RAG agent announcement
        rag_announcement = enhanced_rag_agent.create_announcement("initial", "joining")

        # Update host registry with announcement
        await host_agent.update_agent_registry(rag_announcement)

        # Verify host built tools from announcement
        assert len(host_agent._tools_list) == 1

        tool = host_agent._tools_list[0]
        assert tool.name == "search"
        assert tool.description == "Search the knowledge base with an intelligent RAG system"

    @pytest.mark.anyio
    async def test_tool_wrapper_creates_step_request(self, enhanced_rag_agent):
        """Test that host agent tool wrappers create StepRequest messages."""
        # Create host agent
        host_config = AgentConfig(
            role="HOST",
            agent_obj="StructuredLLMHostAgent",
            description="Test host", 
            parameters={"model": "gpt-4o-mini"}
        )
        host_agent = StructuredLLMHostAgent(**host_config.model_dump())

        # Get RAG agent announcement and register
        rag_announcement = enhanced_rag_agent.create_announcement("initial", "joining")
        await host_agent.update_agent_registry(rag_announcement)

        # Test that tool function can be called without validation errors
        tool = host_agent._tools_list[0]

        # Mock the _proposed_step queue to capture StepRequest
        step_requests = []
        async def mock_put(step_request):
            step_requests.append(step_request)

        host_agent._proposed_step.put = mock_put

        # Call the tool wrapper - this should NOT raise validation errors
        await tool._func(query="test query", max_results=5)

        # Verify StepRequest was created correctly
        assert len(step_requests) == 1
        step_request = step_requests[0]

        assert step_request.role == "POLICY_ANALYST"
        assert step_request.inputs == {"query": "test query", "max_results": 5}
        assert step_request.metadata["tool_name"] == "search"

    def test_fix_resolves_validation_error_scenario(self, enhanced_rag_agent):
        """Test that the fix resolves the original validation error scenario."""
        # Before fix: would have been generic "call_policy_analyst" tool
        generic_tool = enhanced_rag_agent.get_autogen_tool_definition()

        # After fix: uses actual @tool decorated method
        announcement = enhanced_rag_agent.create_announcement("initial", "joining")
        actual_tool = announcement.tool_definition

        # Demonstrate the fix
        assert generic_tool["name"] == "call_policy_analyst"
        assert actual_tool["name"] == "search"

        # The actual tool has specific parameters that match the @tool signature
        assert "query" in actual_tool["parameters"]["properties"]
        assert "max_results" in actual_tool["parameters"]["properties"]

        # The generic tool would have caused validation errors
        assert "prompt" in generic_tool["input_schema"]["properties"]
        assert "query" not in generic_tool["input_schema"]["properties"]

        # This confirms the fix: agent now announces actual tool signature
        # instead of generic agent interface, preventing validation errors
