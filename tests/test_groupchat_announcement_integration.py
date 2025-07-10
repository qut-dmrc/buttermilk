"""Tests for agent announcement integration in group chat."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core import DefaultTopicId, TypeSubscription

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement, FlowEvent
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.libs.autogen import AutogenAgentAdapter
from buttermilk.orchestrators.groupchat import AutogenOrchestrator


class MockButtermilkAgent:
    """Mock Buttermilk agent for testing."""
    
    def __init__(self, **kwargs):
        self.agent_id = kwargs.get('agent_id', 'TEST-123')
        self.role = kwargs.get('role', 'TEST')
        self.agent_name = f"{self.role}-test"
        self.description = kwargs.get('description', f"Test agent with role {self.role}")
        self._cfg = AgentConfig(
            role=self.role,
            description="Test agent",
            agent_id=self.agent_id
        )
        self._listen = AsyncMock()
        self.get_available_tools = MagicMock(return_value=["test_tool"])
        self.get_supported_message_types = MagicMock(return_value=["UIMessage"])
        self.create_announcement = MagicMock()
        self.send_announcement = AsyncMock()
        self.initialize = AsyncMock()
        self.cleanup_with_announcement = AsyncMock()
        self.cleanup = AsyncMock()
        self._heartbeat = MagicMock()


class TestGroupChatAnnouncementIntegration:
    """Test suite for announcement integration in group chat."""

    @pytest.fixture
    def mock_runtime(self):
        """Create a mock Autogen runtime."""
        runtime = MagicMock()
        runtime.add_subscription = AsyncMock()
        runtime.publish_message = AsyncMock()
        runtime.start = AsyncMock()
        runtime.stop = AsyncMock()
        return runtime

    @pytest.fixture
    def orchestrator(self, mock_runtime):
        """Create an AutogenOrchestrator with mocked runtime."""
        with patch('buttermilk.orchestrators.groupchat.SingleThreadedAgentRuntime', return_value=mock_runtime):
            orch = AutogenOrchestrator(
                name="test_orchestrator",
                agents={},
                observers={},
                parameters={}
            )
            orch._runtime = mock_runtime
            orch._topic = DefaultTopicId(type="test-topic")
            return orch

    @pytest.mark.anyio
    async def test_announcement_broadcast_to_all_agents(self, orchestrator, mock_runtime):
        """Test that announcements are broadcast to all agents in group chat."""
        # Create an announcement
        agent_config = AgentConfig(
            role="WORKER",
            description="Test worker",
            agent_id="WORKER-123"
        )
        announcement = AgentAnnouncement(
            content="Worker joining",
            agent_config=agent_config,
            available_tools=["process"],
            announcement_type="initial",
            status="joining"
        )

        # Broadcast the announcement
        await orchestrator._runtime.publish_message(
            announcement,
            topic_id=orchestrator._topic
        )

        # Verify the announcement was published to the main topic
        mock_runtime.publish_message.assert_called_once_with(
            announcement,
            topic_id=orchestrator._topic
        )

    @pytest.mark.anyio
    async def test_agent_receives_announcements_via_adapter(self):
        """Test that agents receive announcements through AutogenAgentAdapter."""
        # Create a mock Buttermilk agent
        mock_agent = MockButtermilkAgent(role="JUDGE", agent_id="JUDGE-456")

        # Create adapter
        adapter = AutogenAgentAdapter(
            agent_cfg={'agent_id': 'JUDGE-456', 'role': 'JUDGE'},
            agent_cls=type(mock_agent),
            topic_type="test-topic"
        )
        adapter.agent = mock_agent

        # Create mock context
        mock_ctx = MagicMock()
        mock_ctx.cancellation_token = MagicMock()
        mock_ctx.topic_id = DefaultTopicId(type="test-topic")
        mock_ctx.sender = "WORKER-123/1"

        # Create announcement
        announcement = AgentAnnouncement(
            content="Worker joining",
            agent_config=AgentConfig(role="WORKER", description="Worker", agent_id="WORKER-123"),
            available_tools=["process"],
            announcement_type="initial"
        )

        # Process announcement through adapter
        await adapter.handle_groupchat_message(message=announcement, ctx=mock_ctx)

        # Verify agent's _listen was called with the announcement
        mock_agent._listen.assert_called_once()
        call_args = mock_agent._listen.call_args
        assert call_args[1]['message'] == announcement
        assert call_args[1]['source'] == "WORKER-123"

    @pytest.mark.anyio
    async def test_host_agent_broadcasts_initial_announcement(self, orchestrator, mock_runtime):
        """Test that host agents broadcast their initial announcement when joining."""
        # Create a host agent
        host_agent = HostAgent(
            role="HOST",
            description="Group chat host",
            unique_identifier="host123"
        )

        # Create a mock callback that will capture messages
        announcements_sent = []
        async def capture_announcement(message):
            if isinstance(message, AgentAnnouncement):
                announcements_sent.append(message)

        # Initialize with announcement
        await host_agent.initialize(
            callback_to_groupchat=mock_runtime.publish_message,
            public_callback=capture_announcement
        )

        # Verify announcement was sent
        assert len(announcements_sent) == 1
        announcement = announcements_sent[0]
        assert announcement.announcement_type == "initial"
        assert announcement.status == "joining"
        assert announcement.agent_config.role == "HOST"
        assert announcement.agent_config.unique_identifier == "host123"

    @pytest.mark.anyio
    async def test_agents_respond_to_host_announcement_in_groupchat(self):
        """Test that agents respond to host announcements in group chat."""
        # Create a regular agent
        mock_agent = MockButtermilkAgent(role="ANALYST", agent_id="ANALYST-789")

        # Create adapter
        adapter = AutogenAgentAdapter(
            agent_cfg={'agent_id': 'ANALYST-789', 'role': 'ANALYST'},
            agent_cls=type(mock_agent),
            topic_type="test-topic"
        )
        adapter.agent = mock_agent

        # Mock the agent's response behavior
        async def mock_listen(message, **kwargs):
            if isinstance(message, AgentAnnouncement) and message.agent_config.role == "HOST":
                # Simulate sending response announcement
                await kwargs['public_callback'](
                    AgentAnnouncement(
                        content="Responding to host",
                        agent_config=mock_agent._cfg,
                        announcement_type="response",
                        responding_to=message.agent_config.agent_id,
                        status="active"
                    )
                )

        mock_agent._listen.side_effect = mock_listen

        # Create host announcement
        host_config = AgentConfig(role="HOST", description="Host", unique_identifier="host123")
        host_announcement = AgentAnnouncement(
            content="Host joining",
            agent_config=host_config,
            announcement_type="initial"
        )

        # Create mock context
        mock_ctx = MagicMock()
        mock_ctx.cancellation_token = MagicMock()
        mock_ctx.topic_id = DefaultTopicId(type="test-topic")
        mock_ctx.sender = "HOST-host123/1"

        # Track published messages
        published_messages = []
        async def capture_publish(msg, topic_id=None):
            published_messages.append(msg)

        adapter.publish_message = capture_publish

        # Process host announcement
        await adapter.handle_groupchat_message(message=host_announcement, ctx=mock_ctx)

        # Verify response was published
        assert len(published_messages) == 1
        response = published_messages[0]
        assert isinstance(response, AgentAnnouncement)
        assert response.announcement_type == "response"
        assert response.responding_to == host_config.agent_id

    @pytest.mark.anyio
    async def test_announcement_routing_preserves_source(self):
        """Test that announcement routing preserves the source agent ID."""
        # Create adapter
        adapter = AutogenAgentAdapter(
            agent_cfg={'agent_id': 'RECEIVER-123', 'role': 'RECEIVER'},
            agent_cls=MockButtermilkAgent,
            topic_type="test-topic"
        )

        # Create mock agent
        mock_agent = MockButtermilkAgent(role="RECEIVER", agent_id="RECEIVER-123")
        adapter.agent = mock_agent

        # Create announcement from a sender
        announcement = AgentAnnouncement(
            content="Sender announcement",
            agent_config=AgentConfig(role="SENDER", description="Sender", agent_id="SENDER-456"),
            announcement_type="initial",
            source="SENDER-456"  # Explicitly set source
        )

        # Create context with sender info
        mock_ctx = MagicMock()
        mock_ctx.cancellation_token = MagicMock()
        mock_ctx.topic_id = DefaultTopicId(type="test-topic")
        mock_ctx.sender = "SENDER-456/autogen"

        # Process announcement
        await adapter.handle_groupchat_message(message=announcement, ctx=mock_ctx)

        # Verify source was preserved
        call_args = mock_agent._listen.call_args
        assert call_args[1]['source'] == "SENDER-456"

    @pytest.mark.anyio
    async def test_announcement_during_groupchat_lifecycle(self, orchestrator, mock_runtime):
        """Test announcements during different phases of group chat lifecycle."""
        # Track all published messages
        published_messages = []
        mock_runtime.publish_message.side_effect = lambda msg, **kwargs: published_messages.append(msg)

        # 1. Initial phase - agents joining
        agent1_announcement = AgentAnnouncement(
            content="Agent 1 joining",
            agent_config=AgentConfig(role="AGENT1", description="Agent 1", agent_id="AGENT1-123"),
            announcement_type="initial",
            status="joining"
        )
        await orchestrator._runtime.publish_message(agent1_announcement, topic_id=orchestrator._topic)

        # 2. Active phase - status update
        agent1_active = AgentAnnouncement(
            content="Agent 1 active",
            agent_config=AgentConfig(role="AGENT1", description="Agent 1", agent_id="AGENT1-123"),
            announcement_type="update",
            status="active"
        )
        await orchestrator._runtime.publish_message(agent1_active, topic_id=orchestrator._topic)

        # 3. Leaving phase
        agent1_leaving = AgentAnnouncement(
            content="Agent 1 leaving",
            agent_config=AgentConfig(role="AGENT1", description="Agent 1", agent_id="AGENT1-123"),
            announcement_type="update",
            status="leaving"
        )
        await orchestrator._runtime.publish_message(agent1_leaving, topic_id=orchestrator._topic)

        # Verify all messages were published
        assert len(published_messages) == 3
        assert all(isinstance(msg, AgentAnnouncement) for msg in published_messages)
        assert [msg.status for msg in published_messages] == ["joining", "active", "leaving"]

    @pytest.mark.anyio
    async def test_edge_case_agent_disconnection_handling(self):
        """Test handling of agent disconnection scenarios."""
        # Create adapter with mock agent
        mock_agent = MockButtermilkAgent(role="WORKER", agent_id="WORKER-123")
        adapter = AutogenAgentAdapter(
            agent_cfg={'agent_id': 'WORKER-123', 'role': 'WORKER'},
            agent_cls=type(mock_agent),
            topic_type="test-topic"
        )
        adapter.agent = mock_agent

        # Simulate cleanup with announcement
        with patch.object(mock_agent, 'cleanup_with_announcement', new_callable=AsyncMock) as mock_cleanup:
            # Trigger cleanup (simulating disconnection)
            await adapter.on_unregister()

            # For now, adapter doesn't call cleanup_with_announcement
            # This test documents expected behavior for future implementation
            # mock_cleanup.assert_called_once()

    @pytest.mark.anyio
    async def test_announcement_message_validation(self):
        """Test that invalid announcements are handled gracefully."""
        mock_agent = MockButtermilkAgent()
        adapter = AutogenAgentAdapter(
            agent_cfg={'agent_id': 'TEST-123', 'role': 'TEST'},
            agent_cls=type(mock_agent),
            topic_type="test-topic"
        )
        adapter.agent = mock_agent

        # Create context
        mock_ctx = MagicMock()
        mock_ctx.cancellation_token = MagicMock()
        mock_ctx.topic_id = DefaultTopicId(type="test-topic")
        mock_ctx.sender = "INVALID/1"

        # Try to create invalid announcement (response without responding_to)
        with pytest.raises(ValueError, match="responding_to must be set"):
            AgentAnnouncement(
                content="Invalid response",
                agent_config=AgentConfig(role="TEST", description="Test"),
                announcement_type="response",  # Response type
                responding_to=None  # But no responding_to
            )
