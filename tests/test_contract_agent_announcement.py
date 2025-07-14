"""Tests for the AgentAnnouncement OOBMessage type in contract.py."""

import pytest
from pydantic import ValidationError

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement, FlowEvent


class TestAgentAnnouncement:
    """Test suite for AgentAnnouncement message type."""

    def test_agent_announcement_creation_minimal(self):
        """Test creating an AgentAnnouncement with minimal required fields."""
        # Create a minimal agent config
        agent_config = AgentConfig(
            role="JUDGE",
            description="Test judge agent"
        )
        
        # Create announcement
        announcement = AgentAnnouncement(
            content="Agent joining group chat",
            agent_config=agent_config,
            announcement_type="initial"
        )
        
        # Verify fields
        assert announcement.agent_config == agent_config
        assert announcement.content == "Agent joining group chat"
        assert announcement.announcement_type == "initial"
        assert announcement.status == "joining"  # default
        assert announcement.available_tools == []  # default
        assert announcement.responding_to is None  # default
        
    def test_agent_announcement_creation_full(self):
        """Test creating an AgentAnnouncement with all fields."""
        # Create agent config with tools
        agent_config = AgentConfig(
            role="FETCH",
            description="Data fetching agent",
            parameters={"model": "gpt-4"}
        )
        
        # Create announcement with all fields
        announcement = AgentAnnouncement(
            content="Fetch agent ready",
            agent_config=agent_config,
            available_tools=["fetch", "search", "scrape"],
            status="active",
            announcement_type="response",
            responding_to="HOST_123",
            source="FETCH_456"
        )
        
        # Verify all fields
        assert announcement.agent_config.role == "FETCH"
        assert announcement.available_tools == ["fetch", "search", "scrape"]
        assert announcement.status == "active"
        assert announcement.announcement_type == "response"
        assert announcement.responding_to == "HOST_123"
        assert announcement.source == "FETCH_456"
        
    def test_agent_announcement_inheritance(self):
        """Test that AgentAnnouncement properly inherits from FlowEvent."""
        agent_config = AgentConfig(role="TEST", description="Test agent")
        announcement = AgentAnnouncement(
            content="Test announcement",
            agent_config=agent_config,
            announcement_type="initial"
        )
        
        # Should inherit FlowEvent fields
        assert hasattr(announcement, "call_id")
        assert hasattr(announcement, "source")
        assert hasattr(announcement, "content")
        assert hasattr(announcement, "agent_info")
        
        # Should be instance of FlowEvent
        assert isinstance(announcement, FlowEvent)
        
    def test_agent_announcement_str_representation(self):
        """Test string representation of AgentAnnouncement."""
        # Provide unique_identifier to control the generated agent_id
        agent_config = AgentConfig(
            role="JUDGE",
            description="Test judge",
            unique_identifier="abc123"  # This will generate agent_id as "JUDGE-abc123"
        )
        
        announcement = AgentAnnouncement(
            content="Judge agent joining",
            agent_config=agent_config,
            announcement_type="initial",
            status="joining"
        )
        
        str_repr = str(announcement)
        assert str_repr == "AgentAnnouncement[JUDGE-abc123]: initial - joining"
        
    def test_agent_announcement_status_validation(self):
        """Test that status field only accepts valid values."""
        agent_config = AgentConfig(role="TEST", description="Test")
        
        # Valid statuses
        for status in ["joining", "active", "leaving"]:
            announcement = AgentAnnouncement(
                content="Test",
                agent_config=agent_config,
                announcement_type="initial",
                status=status
            )
            assert announcement.status == status
            
        # Invalid status should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentAnnouncement(
                content="Test",
                agent_config=agent_config,
                announcement_type="initial",
                status="invalid_status"
            )
        assert "status" in str(exc_info.value)
        
    def test_agent_announcement_type_validation(self):
        """Test that announcement_type field only accepts valid values."""
        agent_config = AgentConfig(role="TEST", description="Test")
        
        # Valid announcement types
        for ann_type in ["initial", "response", "update"]:
            announcement = AgentAnnouncement(
                content="Test",
                agent_config=agent_config,
                announcement_type=ann_type
            )
            assert announcement.announcement_type == ann_type
            
        # Invalid type should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentAnnouncement(
                content="Test",
                agent_config=agent_config,
                announcement_type="invalid_type"
            )
        assert "announcement_type" in str(exc_info.value)
        
    def test_agent_announcement_response_type_requires_responding_to(self):
        """Test that response type announcements can include responding_to field."""
        agent_config = AgentConfig(role="JUDGE", description="Judge agent")
        
        # Response type with responding_to
        announcement = AgentAnnouncement(
            content="Responding to host",
            agent_config=agent_config,
            announcement_type="response",
            responding_to="HOST_xyz789"
        )
        
        assert announcement.announcement_type == "response"
        assert announcement.responding_to == "HOST_xyz789"
        
    def test_agent_announcement_leaving_status(self):
        """Test announcement for agent leaving."""
        agent_config = AgentConfig(
            role="FETCH",
            description="Fetch agent",
            unique_identifier="123"  # This will generate agent_id as "FETCH-123"
        )
        
        announcement = AgentAnnouncement(
            content="Agent disconnecting",
            agent_config=agent_config,
            announcement_type="update",
            status="leaving"
        )
        
        assert announcement.status == "leaving"
        assert str(announcement) == "AgentAnnouncement[FETCH-123]: update - leaving"
        
    def test_agent_announcement_serialization(self):
        """Test that AgentAnnouncement can be serialized/deserialized."""
        agent_config = AgentConfig(
            role="ANALYST",
            description="Analysis agent",
            parameters={"model": "claude-3"}
        )
        
        announcement = AgentAnnouncement(
            content="Analyst ready",
            agent_config=agent_config,
            available_tools=["analyze", "summarize"],
            announcement_type="initial"
        )
        
        # Serialize to dict
        announcement_dict = announcement.model_dump()
        
        # Deserialize back
        announcement_restored = AgentAnnouncement(**announcement_dict)
        
        # Verify fields match
        assert announcement_restored.agent_config.role == "ANALYST"
        assert announcement_restored.available_tools == ["analyze", "summarize"]
        assert announcement_restored.announcement_type == "initial"