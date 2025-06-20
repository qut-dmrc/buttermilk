"""Tests for host agent registry functionality."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement, AgentInput, AgentOutput
from buttermilk.agents.flowcontrol.host import HostAgent


class TestHostAgentRegistry:
    """Test suite for host agent registry functionality."""
    
    @pytest.fixture
    def host_agent(self):
        """Create a host agent instance for testing."""
        return HostAgent(
            role="HOST",
            description="Test host agent",
            parameters={"human_in_loop": False},
            unique_identifier="test_host"
        )
    
    @pytest.mark.asyncio
    async def test_host_agent_initializes_registry(self, host_agent):
        """Test that host agent initializes agent and tool registries."""
        # Verify registries are initialized
        assert hasattr(host_agent, "agent_registry")
        assert hasattr(host_agent, "tool_registry")
        assert isinstance(host_agent.agent_registry, dict)
        assert isinstance(host_agent.tool_registry, dict)
        assert len(host_agent.agent_registry) == 0
        assert len(host_agent.tool_registry) == 0
    
    @pytest.mark.asyncio
    async def test_update_agent_registry_joining(self, host_agent):
        """Test updating registry when agent joins."""
        # Create announcement for joining agent
        agent_config = AgentConfig(
            role="JUDGE",
            description="Test judge agent",
            unique_identifier="judge123"
        )
        announcement = AgentAnnouncement(
            content="Judge joining",
            agent_config=agent_config,
            available_tools=["score", "evaluate"],
            supported_message_types=["UIMessage"],
            status="joining",
            announcement_type="initial"
        )
        
        # Update registry
        host_agent.update_agent_registry(announcement)
        
        # Verify agent was added
        assert "JUDGE-judge123" in host_agent.agent_registry
        assert host_agent.agent_registry["JUDGE-judge123"] == announcement
        
        # Verify tools were registered
        assert "score" in host_agent.tool_registry
        assert "evaluate" in host_agent.tool_registry
        assert "JUDGE-judge123" in host_agent.tool_registry["score"]
        assert "JUDGE-judge123" in host_agent.tool_registry["evaluate"]
    
    @pytest.mark.asyncio
    async def test_update_agent_registry_leaving(self, host_agent):
        """Test updating registry when agent leaves."""
        # First add an agent
        agent_config = AgentConfig(
            role="FETCH",
            description="Test fetch agent",
            unique_identifier="fetch456"
        )
        joining_announcement = AgentAnnouncement(
            content="Fetch joining",
            agent_config=agent_config,
            available_tools=["fetch", "search"],
            supported_message_types=["ConductorRequest"],
            status="joining",
            announcement_type="initial"
        )
        host_agent.update_agent_registry(joining_announcement)
        
        # Verify agent is in registry
        assert "FETCH-fetch456" in host_agent.agent_registry
        assert "fetch" in host_agent.tool_registry
        
        # Create leaving announcement
        leaving_announcement = AgentAnnouncement(
            content="Fetch leaving",
            agent_config=agent_config,
            available_tools=["fetch", "search"],
            supported_message_types=["ConductorRequest"],
            status="leaving",
            announcement_type="update"
        )
        
        # Update registry
        host_agent.update_agent_registry(leaving_announcement)
        
        # Verify agent was removed
        assert "FETCH-fetch456" not in host_agent.agent_registry
        
        # Verify tools were unregistered
        assert "FETCH-fetch456" not in host_agent.tool_registry.get("fetch", [])
        assert "FETCH-fetch456" not in host_agent.tool_registry.get("search", [])
    
    @pytest.mark.asyncio
    async def test_update_agent_registry_multiple_agents_same_tool(self, host_agent):
        """Test registry handles multiple agents with same tool."""
        # Add first agent with "analyze" tool
        agent1_config = AgentConfig(
            role="ANALYST",
            description="First analyst",
            unique_identifier="analyst1"
        )
        announcement1 = AgentAnnouncement(
            content="Analyst 1 joining",
            agent_config=agent1_config,
            available_tools=["analyze", "summarize"],
            status="joining",
            announcement_type="initial"
        )
        host_agent.update_agent_registry(announcement1)
        
        # Add second agent with "analyze" tool
        agent2_config = AgentConfig(
            role="ANALYST",
            description="Second analyst",
            unique_identifier="analyst2"
        )
        announcement2 = AgentAnnouncement(
            content="Analyst 2 joining",
            agent_config=agent2_config,
            available_tools=["analyze", "report"],
            status="joining",
            announcement_type="initial"
        )
        host_agent.update_agent_registry(announcement2)
        
        # Verify both agents are in registry
        assert "ANALYST-analyst1" in host_agent.agent_registry
        assert "ANALYST-analyst2" in host_agent.agent_registry
        
        # Verify analyze tool has both agents
        assert len(host_agent.tool_registry["analyze"]) == 2
        assert "ANALYST-analyst1" in host_agent.tool_registry["analyze"]
        assert "ANALYST-analyst2" in host_agent.tool_registry["analyze"]
        
        # Verify unique tools
        assert len(host_agent.tool_registry["summarize"]) == 1
        assert len(host_agent.tool_registry["report"]) == 1
    
    @pytest.mark.asyncio
    async def test_create_registry_summary(self, host_agent):
        """Test creating a summary of the agent registry."""
        # Add multiple agents
        agents_data = [
            ("JUDGE", "judge1", ["score", "evaluate"], "gpt-4"),
            ("FETCH", "fetch1", ["fetch", "search"], None),
            ("ANALYST", "analyst1", ["analyze", "summarize"], "claude-3"),
        ]
        
        for role, unique_id, tools, model in agents_data:
            config = AgentConfig(
                role=role,
                description=f"Test {role.lower()} agent",
                unique_identifier=unique_id,
                parameters={"model": model} if model else {}
            )
            announcement = AgentAnnouncement(
                content=f"{role} joining",
                agent_config=config,
                available_tools=tools,
                status="active",
                announcement_type="initial"
            )
            host_agent.update_agent_registry(announcement)
        
        # Create summary
        summary = host_agent.create_registry_summary()
        
        # Verify summary structure
        assert "active_agents" in summary
        assert "available_tools" in summary
        assert "total_agents" in summary
        
        # Verify agent count
        assert summary["total_agents"] == 3
        assert len(summary["active_agents"]) == 3
        
        # Verify agent details
        judge_agent = next(a for a in summary["active_agents"] if a["role"] == "JUDGE")
        assert judge_agent["agent_id"] == "JUDGE-judge1"
        assert judge_agent["tools"] == ["score", "evaluate"]
        assert judge_agent["model"] == "gpt-4"
        
        fetch_agent = next(a for a in summary["active_agents"] if a["role"] == "FETCH")
        assert fetch_agent["model"] is None
        
        # Verify tool mapping
        assert len(summary["available_tools"]["score"]) == 1
        assert "JUDGE-judge1" in summary["available_tools"]["score"]
        assert len(summary["available_tools"]["fetch"]) == 1
        assert "FETCH-fetch1" in summary["available_tools"]["fetch"]
    
    @pytest.mark.asyncio
    async def test_host_listens_for_announcements(self, host_agent):
        """Test that host agent processes announcement messages in _listen."""
        # Mock callbacks
        public_callback = AsyncMock()
        message_callback = AsyncMock()
        
        # Initialize host
        await host_agent.initialize(callback_to_groupchat=AsyncMock())
        
        # Create announcement
        agent_config = AgentConfig(
            role="WORKER",
            description="Test worker",
            unique_identifier="worker1"
        )
        announcement = AgentAnnouncement(
            content="Worker joining",
            agent_config=agent_config,
            available_tools=["process", "transform"],
            status="joining",
            announcement_type="initial"
        )
        
        # Process announcement
        await host_agent._listen(
            message=announcement,
            cancellation_token=MagicMock(),
            source="WORKER-worker1",
            public_callback=public_callback,
            message_callback=message_callback
        )
        
        # Verify agent was registered
        assert "WORKER-worker1" in host_agent.agent_registry
        assert "process" in host_agent.tool_registry
    
    @pytest.mark.asyncio
    async def test_host_announces_itself_and_receives_responses(self, host_agent):
        """Test host announces itself and processes response announcements."""
        # Mock callbacks
        public_callback = AsyncMock()
        
        # Initialize and announce
        await host_agent.initialize_with_announcement(
            callback_to_groupchat=AsyncMock(),
            public_callback=public_callback
        )
        
        # Verify host announcement was sent
        public_callback.assert_called_once()
        host_announcement = public_callback.call_args[0][0]
        assert isinstance(host_announcement, AgentAnnouncement)
        assert host_announcement.agent_config.role == "HOST"
        assert host_announcement.announcement_type == "initial"
        
        # Simulate response from another agent
        responder_config = AgentConfig(
            role="RESPONDER",
            description="Responding agent",
            unique_identifier="resp1"
        )
        response_announcement = AgentAnnouncement(
            content="Responding to host",
            agent_config=responder_config,
            available_tools=["respond"],
            status="active",
            announcement_type="response",
            responding_to=host_agent.agent_id
        )
        
        # Process response
        await host_agent._listen(
            message=response_announcement,
            cancellation_token=MagicMock(),
            source="RESPONDER-resp1",
            public_callback=public_callback,
            message_callback=AsyncMock()
        )
        
        # Verify responder was registered
        assert "RESPONDER-resp1" in host_agent.agent_registry
        assert host_agent.agent_registry["RESPONDER-resp1"].responding_to == host_agent.agent_id
    
    @pytest.mark.asyncio
    async def test_registry_summary_in_ui_messages(self, host_agent):
        """Test that registry summary is included in UI messages."""
        # Add an agent to registry
        agent_config = AgentConfig(
            role="WORKER",
            description="Test worker",
            unique_identifier="worker1"
        )
        announcement = AgentAnnouncement(
            content="Worker active",
            agent_config=agent_config,
            available_tools=["work"],
            status="active",
            announcement_type="initial"
        )
        host_agent.update_agent_registry(announcement)
        
        # Create UI message with registry summary
        ui_message = host_agent.create_ui_message_with_registry(
            content="Processing request",
            options=True
        )
        
        # Verify UI message includes registry summary
        assert ui_message.content == "Processing request"
        assert ui_message.options == True
        assert ui_message.agent_registry_summary is not None
        assert ui_message.agent_registry_summary["total_agents"] == 1
        assert len(ui_message.agent_registry_summary["active_agents"]) == 1