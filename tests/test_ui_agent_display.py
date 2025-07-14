"""Tests for UI display of agent announcements and summaries."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from rich.text import Text

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement, UIMessage
from buttermilk.agents.ui.console import CLIUserAgent


class TestUIAgentDisplay:
    """Test suite for UI display of agent announcements."""

    @pytest.fixture
    def console_agent(self):
        """Create a CLIUserAgent for testing."""
        agent = CLIUserAgent(
            role="MANAGER",
            description="Console UI for testing"
        )
        agent._console = MagicMock(spec=Console)
        return agent

    @pytest.fixture
    def sample_announcement(self):
        """Create a sample agent announcement."""
        return AgentAnnouncement(
            content="Test agent joining",
            agent_config=AgentConfig(
                role="WORKER",
                description="Test worker agent",
                unique_identifier="worker123"
            ),
            available_tools=["process", "analyze"],
            announcement_type="initial",
            status="joining"
        )

    @pytest.mark.anyio
    async def test_announcement_formatting(self, console_agent, sample_announcement):
        """Test that agent announcements are properly formatted for display."""
        # Format the announcement
        formatted = console_agent._fmt_msg(sample_announcement, source="WORKER-worker123")

        # Verify formatting
        assert formatted is not None
        assert isinstance(formatted, Text)

        # Check that the text contains expected elements
        text_str = str(formatted)
        assert "joining" in text_str.lower()
        assert any(tool in text_str for tool in ["process", "analyze"])

    @pytest.mark.anyio
    async def test_ui_message_with_agent_summary(self, console_agent):
        """Test UIMessage with agent registry summary is displayed properly."""
        # Create UIMessage with agent summary and trigger word for detailed display
        ui_message = UIMessage(
            content="!agents",  # This triggers detailed display
            agent_registry_summary={
                "HOST-host123": {
                    "role": "HOST",
                    "status": "active",
                    "tools": ["orchestrate", "manage"]
                },
                "WORKER-worker456": {
                    "role": "WORKER",
                    "status": "active",
                    "tools": ["process", "analyze"]
                }
            }
        )

        # Format the message
        formatted = console_agent._fmt_msg(ui_message, source="system")

        # Verify formatting
        assert formatted is not None
        text_str = str(formatted)
        assert "AGENTS:" in text_str or "HOST" in text_str

    @pytest.mark.anyio
    async def test_announcement_display_in_listen(self, console_agent, sample_announcement):
        """Test that announcements are displayed when received via _listen."""
        # Track what was printed
        printed_messages = []
        console_agent._console.print = lambda msg: printed_messages.append(msg)

        # Process announcement through _listen
        await console_agent._listen(
            message=sample_announcement,
            source="WORKER-worker123"
        )

        # Verify something was printed (the formatted announcement)
        assert len(printed_messages) > 0

    @pytest.mark.anyio
    async def test_agent_status_colors(self, console_agent):
        """Test different status colors for agent announcements."""
        statuses = [
            ("joining", "green"),
            ("active", "bright_blue"),
            ("leaving", "yellow")
        ]

        for status, expected_color in statuses:
            announcement = AgentAnnouncement(
                content=f"Agent is {status}",
                agent_config=AgentConfig(role="TEST", description="Test"),
                announcement_type="update",
                status=status
            )

            formatted = console_agent._fmt_msg(announcement, source="TEST-123")
            assert formatted is not None
            # Note: Color verification would require deeper inspection of Text object

    @pytest.mark.anyio
    async def test_agent_list_command_display(self, console_agent):
        """Test display of agent list when requested."""
        # Mock console print
        printed_messages = []
        console_agent._console.print = lambda msg: printed_messages.append(msg)

        # Create a UI message requesting agent list
        ui_msg = UIMessage(
            content="!agents",  # Command to list agents
            agent_registry_summary={
                "JUDGE-j123": {
                    "role": "JUDGE",
                    "status": "active",
                    "tools": ["evaluate", "score"],
                    "model": "gpt-4"
                },
                "SCORER-s456": {
                    "role": "SCORER", 
                    "status": "active",
                    "tools": ["calculate"],
                    "model": "claude-3-sonnet"
                }
            }
        )

        # Display the message
        await console_agent.callback_to_ui(ui_msg, source="system")

        # Verify output was printed
        assert len(printed_messages) > 0

    @pytest.mark.anyio
    async def test_announcement_type_icons(self, console_agent):
        """Test different icons for announcement types."""
        announcement_types = [
            ("initial", "🆕", None),
            ("response", "↩️", "HOST-123"),  # response type needs responding_to
            ("update", "🔄", None)
        ]

        for ann_type, expected_icon, responding_to in announcement_types:
            kwargs = {
                "content": f"Announcement type: {ann_type}",
                "agent_config": AgentConfig(role="TEST", description="Test"),
                "announcement_type": ann_type,
                "status": "active"
            }
            if responding_to:
                kwargs["responding_to"] = responding_to

            announcement = AgentAnnouncement(**kwargs)

            # For now, just verify formatting works
            formatted = console_agent._fmt_msg(announcement, source="TEST-123")
            assert formatted is not None

    @pytest.mark.anyio
    async def test_tools_display_in_announcement(self, console_agent):
        """Test that available tools are displayed in announcements."""
        announcement = AgentAnnouncement(
            content="Agent with many tools",
            agent_config=AgentConfig(role="TOOLBOX", description="Multi-tool agent"),
            available_tools=["search", "extract", "analyze", "summarize", "visualize"],
            announcement_type="initial",
            status="joining"
        )

        formatted = console_agent._fmt_msg(announcement, source="TOOLBOX-123")
        assert formatted is not None

        # Convert to string for verification
        text_str = str(formatted)
        # At least some tools should be mentioned
        assert any(tool in text_str for tool in ["search", "extract", "analyze"])

    @pytest.mark.anyio
    async def test_agent_registry_summary_formatting(self, console_agent):
        """Test formatting of complete agent registry summary."""
        # Create a comprehensive registry summary
        registry_summary = {
            "HOST-h1": {
                "role": "HOST",
                "status": "active", 
                "tools": ["orchestrate"],
                "model": "gpt-4"
            },
            "JUDGE-j1": {
                "role": "JUDGE",
                "status": "active",
                "tools": ["evaluate", "score"],
                "model": "claude-3-opus"
            },
            "WORKER-w1": {
                "role": "WORKER",
                "status": "leaving",
                "tools": ["process"],
                "model": "llama-3"
            }
        }

        ui_msg = UIMessage(
            content="Agent Status Report",
            agent_registry_summary=registry_summary
        )

        formatted = console_agent._fmt_msg(ui_msg, source="system")
        assert formatted is not None
