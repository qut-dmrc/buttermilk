"""Test AgentConfig ID generation fix for session loading issue."""

import pytest
from buttermilk._core.config import AgentConfig


def test_agentconfig_conditional_id_generation():
    """Test that agent_id is only generated when not provided."""
    
    # Test 1: ID generation when none provided
    config1 = AgentConfig(role="TEST", description="Test agent")
    assert config1.agent_id != ""
    assert len(config1.agent_id) > 0
    # Should be a shortuuid (22 characters, alphanumeric)
    assert len(config1.agent_id) == 22
    assert config1.agent_id.isalnum()
    
    # Test 2: Preserve existing ID (session loading scenario)
    existing_id = "test-session-id-123"
    config2 = AgentConfig(
        role="TEST", 
        description="Test agent",
        agent_id=existing_id
    )
    assert config2.agent_id == existing_id
    
    # Test 3: Agent name uses actual agent_id
    assert config2.agent_name == f"TEST {existing_id}"


def test_agentconfig_no_unique_identifier():
    """Test that unique_identifier field no longer exists."""
    
    config = AgentConfig(role="TEST", description="Test agent")
    
    # Should not have unique_identifier attribute
    assert not hasattr(config, 'unique_identifier')
    
    # Agent name should contain agent_id, not unique_identifier
    assert config.agent_id in config.agent_name


def test_agentconfig_shortuuid_format():
    """Test that generated agent_id is a proper shortuuid."""
    
    config = AgentConfig(role="TEST", description="Test agent")
    
    # Shortuuid should be alphanumeric with no hyphens
    assert config.agent_id.isalnum()
    
    # Should be 22 characters long (standard shortuuid length)
    assert len(config.agent_id) == 22


def test_multiple_configs_unique_ids():
    """Test that multiple configs get unique IDs."""
    
    config1 = AgentConfig(role="TEST1", description="Test agent 1")
    config2 = AgentConfig(role="TEST2", description="Test agent 2")
    
    # Should have different IDs
    assert config1.agent_id != config2.agent_id
    
    # Both should be valid shortuuids
    assert len(config1.agent_id) == 22
    assert len(config2.agent_id) == 22