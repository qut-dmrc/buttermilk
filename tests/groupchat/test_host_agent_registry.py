"""Tests for host agent registry functionality.

These tests verify the HostAgent's ability to create registry summaries
for UI display. Most registry tests are covered by integration tests.
"""

import asyncio

import pytest

from buttermilk.agents.flowcontrol.host import HostAgent


class TestHostAgentRegistry:
    """Test suite for host agent registry functionality."""

    @pytest.fixture
    def host_agent(self):
        """Create a host agent instance for testing."""
        return HostAgent(
            role="HOST",
            description="Test host agent",
            parameters={
                "human_in_loop": False,
                "max_wait_time": 120,
                "max_user_confirmation_time": 60,
                "error_threshold": 0.5,
            },
            unique_identifier="test_host",
        )

    @pytest.mark.anyio
    async def test_host_agent_initializes_registry(self, host_agent):
        """Test that host agent initializes with empty registries."""
        # Verify registries are initialized
        assert hasattr(host_agent, "_agent_registry")
        assert hasattr(host_agent, "_tool_to_agent_map")
        assert hasattr(host_agent, "_registry_lock")
        assert isinstance(host_agent._agent_registry, dict)
        assert isinstance(host_agent._tool_to_agent_map, dict)
        assert isinstance(host_agent._registry_lock, asyncio.Lock)
        assert len(host_agent._agent_registry) == 0
        assert len(host_agent._tool_to_agent_map) == 0

    @pytest.mark.anyio
    async def test_create_empty_registry_summary(self, host_agent):
        """Test creating a summary when no agents are registered."""
        summary = host_agent.create_registry_summary()

        # Verify summary structure
        assert "active_agents" in summary
        assert "total_agents" in summary

        # Verify empty state
        assert summary["total_agents"] == 0
        assert len(summary["active_agents"]) == 0

    @pytest.mark.anyio
    async def test_registry_summary_caching(self, host_agent):
        """Test that registry summaries are cached."""
        summary1 = host_agent.create_registry_summary()
        summary2 = host_agent.create_registry_summary()

        # Should be the exact same object (cached)
        assert summary1 is summary2

    @pytest.mark.anyio
    async def test_create_ui_message_with_registry(self, host_agent):
        """Test creating UI message includes registry summary."""
        ui_message = host_agent.create_ui_message_with_registry(
            content="Test message",
            options=["option1", "option2"],
        )

        # Verify message structure
        assert ui_message.content == "Test message"
        assert ui_message.options == ["option1", "option2"]
        assert ui_message.agent_registry_summary is not None
        assert ui_message.agent_registry_summary["total_agents"] == 0
