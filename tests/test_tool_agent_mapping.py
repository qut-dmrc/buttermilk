"""Tests for tool-to-agent mapping functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_core.tools import FunctionTool

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.flowcontrol.host import HostAgent


class TestToolAgentMapping:
    """Test suite for tool-to-agent mapping functionality."""

    @pytest.fixture
    def host_agent(self):
        """Create a host agent instance for testing."""
        return HostAgent(
            role="HOST",
            description="Test host agent",
            parameters={"human_in_loop": False},
            unique_identifier="test_host",
        )

    @pytest.fixture
    def fetch_agent_config(self):
        """Create a fetch agent config for testing."""
        return AgentConfig(
            role="FETCH",
            description="Test fetch agent",
            unique_identifier="fetch123",
        )

    @pytest.fixture
    def fetch_tools(self):
        """Create example tools for fetch agent."""
        async def mock_fetch_uri(uri: str) -> dict:
            return {"mock": "result"}
            
        async def mock_fetch_record(record_id: str) -> dict:
            return {"mock": "result"}
            
        return [
            FunctionTool(
                name="fetch_uri",
                description="Get a record from a given URI",
                func=mock_fetch_uri,
                strict=True,
            ),
            FunctionTool(
                name="fetch_record",
                description="Get a record from a given record ID",
                func=mock_fetch_record,
                strict=True,
            ),
        ]

    @pytest.fixture
    def rag_agent_config(self):
        """Create a RAG agent config for testing."""
        return AgentConfig(
            role="RAG",
            description="Test RAG agent",
            unique_identifier="rag456",
        )

    @pytest.fixture
    def rag_tool(self):
        """Create example tool for RAG agent."""
        return AgentToolDefinition(
            name="search_documents",
            description="Search documents for relevant information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array", "description": "Search results"},
                },
            },
        )

    @pytest.mark.anyio
    async def test_host_initializes_tool_mapping(self, host_agent):
        """Test that host agent initializes tool-to-agent mapping."""
        assert hasattr(host_agent, "_tool_to_agent_map")
        assert isinstance(host_agent._tool_to_agent_map, dict)
        assert len(host_agent._tool_to_agent_map) == 0

    @pytest.mark.anyio
    async def test_tool_mapping_with_single_tool_agent(
        self, host_agent, rag_agent_config, rag_tool
    ):
        """Test tool mapping with an agent that has a single tool."""
        # Create announcement - host will build the mapping from tool_definitions
        announcement = AgentAnnouncement(
            content="RAG agent joining",
            agent_config=rag_agent_config,
            tool_definitions=[rag_tool],
            status="joining",
            announcement_type="initial",
        )

        # Mock the message context
        ctx = MagicMock()

        # Update registry
        await host_agent.update_agent_registry(announcement, ctx)

        # Verify tool mapping was updated
        agent_id = rag_agent_config.agent_id
        assert host_agent._tool_to_agent_map["search_documents"] == agent_id
        assert len(host_agent._tool_to_agent_map) == 1

        # Verify agent is in registry
        assert agent_id in host_agent._agent_registry
        assert host_agent._agent_registry[agent_id] == announcement

    @pytest.mark.anyio
    async def test_tool_mapping_with_multi_tool_agent(
        self, host_agent, fetch_agent_config, fetch_tools
    ):
        """Test tool mapping with an agent that has multiple tools."""
        # Create announcement - host will build mapping from tool_definitions
        announcement = AgentAnnouncement(
            content="Fetch agent joining",
            agent_config=fetch_agent_config,
            tool_definitions=fetch_tools,
            status="joining",
            announcement_type="initial",
        )

        # Mock the message context
        ctx = MagicMock()

        # Update registry
        await host_agent.update_agent_registry(announcement, ctx)

        # Verify both tools map to the same agent
        agent_id = fetch_agent_config.agent_id
        assert host_agent._tool_to_agent_map["fetch_uri"] == agent_id
        assert host_agent._tool_to_agent_map["fetch_record"] == agent_id
        assert len(host_agent._tool_to_agent_map) == 2

        # Verify agent is in registry
        assert agent_id in host_agent._agent_registry

    @pytest.mark.anyio
    async def test_tool_mapping_with_multiple_agents(
        self, host_agent, fetch_agent_config, fetch_tools, rag_agent_config, rag_tool
    ):
        """Test tool mapping with multiple agents each having different tools."""
        # Create announcements for both agents - host builds mappings from tool_definitions
        fetch_announcement = AgentAnnouncement(
            content="Fetch agent joining",
            agent_config=fetch_agent_config,
            tool_definitions=fetch_tools,
            status="joining",
            announcement_type="initial",
        )

        rag_announcement = AgentAnnouncement(
            content="RAG agent joining",
            agent_config=rag_agent_config,
            tool_definitions=[rag_tool],
            status="joining",
            announcement_type="initial",
        )

        # Mock the message context
        ctx = MagicMock()

        # Update registry with both agents
        await host_agent.update_agent_registry(fetch_announcement, ctx)
        await host_agent.update_agent_registry(rag_announcement, ctx)

        # Verify all tools are mapped correctly
        fetch_agent_id = fetch_agent_config.agent_id
        rag_agent_id = rag_agent_config.agent_id
        assert host_agent._tool_to_agent_map["fetch_uri"] == fetch_agent_id
        assert host_agent._tool_to_agent_map["fetch_record"] == fetch_agent_id
        assert host_agent._tool_to_agent_map["search_documents"] == rag_agent_id
        assert len(host_agent._tool_to_agent_map) == 3

        # Verify both agents are in registry
        assert fetch_agent_id in host_agent._agent_registry
        assert rag_agent_id in host_agent._agent_registry

    @pytest.mark.anyio
    async def test_tool_routing_uses_mapping(self, host_agent, fetch_agent_config, fetch_tools):
        """Test that tool call routing uses the tool-to-agent mapping."""
        # Set up the agent in the registry
        announcement = AgentAnnouncement(
            content="Fetch agent joining",
            agent_config=fetch_agent_config,
            tool_definitions=fetch_tools,
            status="joining",
            announcement_type="initial",
        )

        ctx = MagicMock()
        await host_agent.update_agent_registry(announcement, ctx)

        # Mock a tool call
        mock_tool_call = MagicMock()
        mock_tool_call.name = "fetch_uri"
        mock_tool_call.arguments = '{"url": "https://example.com"}'
        mock_tool_call.id = "call_123"

        # Mock the publish method to capture what gets sent
        host_agent._publish = AsyncMock()

        # Route the tool call
        await host_agent._route_tool_calls_to_agents([mock_tool_call])

        # Verify that _publish was called with the correct StepRequest
        host_agent._publish.assert_called_once()
        call_args = host_agent._publish.call_args
        step_request = call_args[0][0]

        # Verify the step request has the correct role and metadata
        assert step_request.role == "FETCH"
        assert step_request.metadata["tool_name"] == "fetch_uri"
        assert step_request.metadata["tool_call_id"] == "call_123"
        assert step_request.inputs == {"url": "https://example.com"}

    @pytest.mark.anyio
    async def test_tool_routing_unknown_tool(self, host_agent):
        """Test that routing an unknown tool logs an error."""
        # Mock a tool call for a tool that doesn't exist
        mock_tool_call = MagicMock()
        mock_tool_call.name = "unknown_tool"
        mock_tool_call.arguments = '{"param": "value"}'
        mock_tool_call.id = "call_456"

        # Mock the publish method - it should not be called
        host_agent._publish = AsyncMock()

        # Route the tool call - should handle gracefully
        await host_agent._route_tool_calls_to_agents([mock_tool_call])

        # Verify that _publish was not called since the tool is unknown
        host_agent._publish.assert_not_called()