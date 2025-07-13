"""Tests for agent announcement behavior in the base Agent class."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import (
    AgentAnnouncement,
    AgentInput,
    AgentOutput,
    AgentTrace,
    ErrorEvent,
    HeartBeat,
)
from buttermilk._core.types import Record
from buttermilk._core.exceptions import ProcessingError


class MockAgent(Agent):
    """A concrete implementation of Agent for testing."""
    
    async def _process(self, message: AgentInput) -> AgentOutput:
        """Simple implementation that returns a basic output."""
        return AgentOutput(
            outputs="Test output",
            role=self.role,
            agent_id=self.agent_id
        )
    
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return empty list by default, can be overridden in tests."""
        return getattr(self, '_mock_tools', [])


class TestAgentAnnouncementBehavior:
    """Test suite for agent announcement behavior."""

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return {
            "role": "TEST",
            "description": "Test agent for announcements",
            "parameters": {"model": "test-model"},
            "unique_identifier": "test123"
        }

    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create a mock agent instance."""
        return MockAgent(**agent_config)

    @pytest.mark.anyio
    async def test_agent_announces_on_initialize(self, mock_agent):
        """Test that agent sends announcement when initialized."""
        # Create mock callbacks
        public_callback = AsyncMock()

        # Initialize the agent with announcement support
        await mock_agent.initialize_with_announcement(public_callback=public_callback)

        # Verify announcement was sent
        public_callback.assert_called_once()
        announcement = public_callback.call_args[0][0]

        assert isinstance(announcement, AgentAnnouncement)
        assert announcement.agent_config.agent_id == mock_agent.agent_id
        assert announcement.announcement_type == "initial"
        assert announcement.status == "joining"
        assert announcement.content == f"Agent {mock_agent.agent_name} joining"

    @pytest.mark.anyio
    async def test_agent_detects_available_tools(self, mock_agent):
        """Test that agent correctly detects its available tools."""
        # Set mock tools
        mock_agent._mock_tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"}
        ]

        # Get available tools
        tools = mock_agent.get_available_tools()

        assert tools == ["tool1", "tool2"]

    @pytest.mark.anyio
    async def test_agent_responds_to_host_announcement(self, mock_agent):
        """Test that agent responds when receiving a host announcement."""
        # Create mock callbacks
        public_callback = AsyncMock()

        # Create a host announcement
        host_config = AgentConfig(
            role="HOST",
            description="Test host agent",
            unique_identifier="host123"
        )
        host_announcement = AgentAnnouncement(
            content="Host agent joining",
            agent_config=host_config,
            announcement_type="initial",
            status="joining",
            source="HOST-host123"
        )

        # Process the announcement
        await mock_agent._listen(
            message=host_announcement,
            cancellation_token=MagicMock(),
            source="HOST-host123",
            public_callback=public_callback
        )

        # Verify response announcement was sent
        public_callback.assert_called_once()
        response = public_callback.call_args[0][0]

        assert isinstance(response, AgentAnnouncement)
        assert response.agent_config.agent_id == mock_agent.agent_id
        assert response.announcement_type == "response"
        assert response.responding_to == "HOST-host123"
        assert response.status == "active"

    @pytest.mark.anyio
    async def test_agent_does_not_respond_to_non_host_announcement(self, mock_agent):
        """Test that agent ignores announcements from non-host agents."""
        # Create mock callbacks
        public_callback = AsyncMock()

        # Create a non-host announcement
        other_config = AgentConfig(
            role="JUDGE",
            description="Test judge agent",
            unique_identifier="judge123"
        )
        other_announcement = AgentAnnouncement(
            content="Judge agent joining",
            agent_config=other_config,
            announcement_type="initial",
            status="joining",
            source="JUDGE-judge123"
        )

        # Process the announcement
        await mock_agent._listen(
            message=other_announcement,
            cancellation_token=MagicMock(),
            source="JUDGE-judge123",
            public_callback=public_callback
        )

        # Verify no response was sent
        public_callback.assert_not_called()

    @pytest.mark.anyio
    async def test_agent_announces_on_cleanup(self, mock_agent):
        """Test that agent sends leaving announcement on cleanup."""
        # Create mock callback
        public_callback = AsyncMock()

        # Store callback in agent for cleanup to use
        mock_agent._announcement_callback = public_callback

        # Cleanup the agent
        await mock_agent.cleanup_with_announcement()

        # Verify leaving announcement was sent
        public_callback.assert_called_once()
        announcement = public_callback.call_args[0][0]

        assert isinstance(announcement, AgentAnnouncement)
        assert announcement.agent_config.agent_id == mock_agent.agent_id
        assert announcement.announcement_type == "update"
        assert announcement.status == "leaving"
        assert announcement.content == f"Agent {mock_agent.agent_name} leaving"

    @pytest.mark.anyio
    async def test_agent_announcement_includes_tools_and_message_types(self, mock_agent):
        """Test that announcements include available tools and supported message types."""
        # Set mock tools
        mock_agent._mock_tools = [
            {"name": "analyze", "description": "Analyze data"},
            {"name": "summarize", "description": "Summarize text"}
        ]

        # Create announcement
        announcement = mock_agent.create_announcement("initial", "joining")

        assert announcement.available_tools == ["analyze", "summarize"]
        assert announcement.supported_message_types == []  # Always empty since method is removed

    @pytest.mark.anyio
    async def test_agent_handles_announcement_errors_gracefully(self, mock_agent):
        """Test that agent handles errors during announcement gracefully."""
        # Create mock callback that raises an error
        public_callback = AsyncMock(side_effect=Exception("Network error"))

        # This should not raise an exception
        try:
            await mock_agent.send_announcement(
                public_callback=public_callback,
                announcement_type="initial",
                status="joining"
            )
        except Exception:
            pytest.fail("Agent should handle announcement errors gracefully")

    @pytest.mark.anyio
    async def test_agent_announcement_in_invoke_lifecycle(self, mock_agent):
        """Test that agent can announce during invoke when callback is stored."""
        # Create mock callbacks
        public_callback = AsyncMock()

        # First initialize with announcement to store the callback
        await mock_agent.initialize_with_announcement(public_callback=public_callback)

        # Create test input
        test_input = AgentInput(
            inputs={"message": "Test message"},
            parameters={}
        )

        # Invoke the agent
        result = await mock_agent.invoke(
            message=test_input,
            public_callback=public_callback
        )

        # Find announcement calls (should have initial announcement)
        announcement_calls = [
            call for call in public_callback.call_args_list
            if len(call[0]) > 0 and isinstance(call[0][0], AgentAnnouncement)
        ]

        # Should have the initial announcement
        assert len(announcement_calls) >= 1

        # Check the initial announcement
        announcement = announcement_calls[0][0][0]
        assert announcement.announcement_type == "initial"
        assert announcement.status == "joining"
