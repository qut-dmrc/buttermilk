"""Test AgentConfig ID generation fix for session loading issue."""

from buttermilk._core.config import AgentConfig


def test_agentconfig_conditional_id_generation():
    """Test that agent_id is only generated when not provided."""
    
    # Test 1: ID generation when none provided
    config1 = AgentConfig(role="TEST", description="Test agent")
    assert config1.agent_id != ""
    assert len(config1.agent_id) > 0
    # Should be in format {ROLE}-{6-char-uuid}
    assert config1.agent_id.startswith("TEST-")
    # The UUID part should be 6 characters
    uuid_part = config1.agent_id.split("-")[1]
    assert len(uuid_part) == 6
    assert uuid_part.isalnum()
    
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
    assert not hasattr(config, "unique_identifier")
    
    # Agent name should contain agent_id, not unique_identifier
    assert config.agent_id in config.agent_name


def test_agentconfig_shortuuid_format():
    """Test that generated agent_id has the expected format."""
    
    config = AgentConfig(role="TEST", description="Test agent")
    
    # Should be in format {ROLE}-{6-char-uuid}
    assert "-" in config.agent_id
    parts = config.agent_id.split("-")
    assert len(parts) == 2
    assert parts[0] == "TEST"
    # UUID part should be alphanumeric
    assert parts[1].isalnum()
    assert len(parts[1]) == 6


def test_multiple_configs_unique_ids():
    """Test that multiple configs get unique IDs."""
    
    config1 = AgentConfig(role="TEST1", description="Test agent 1")
    config2 = AgentConfig(role="TEST2", description="Test agent 2")
    
    # Should have different IDs
    assert config1.agent_id != config2.agent_id
    
    # Both should follow the {ROLE}-{6-char-uuid} format
    assert config1.agent_id.startswith("TEST1-")
    assert config2.agent_id.startswith("TEST2-")
    
    # UUID parts should be different
    uuid1 = config1.agent_id.split("-")[1]
    uuid2 = config2.agent_id.split("-")[1]
    assert uuid1 != uuid2
    assert len(uuid1) == 6
    assert len(uuid2) == 6
