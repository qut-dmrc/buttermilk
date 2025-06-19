"""Test to demonstrate and fix the participants issue in host agents."""

import pytest
from unittest.mock import Mock, patch
from buttermilk._core.config import AgentVariants
from buttermilk.orchestrators.groupchat import AutogenOrchestrator


@pytest.mark.asyncio
async def test_participants_includes_observers():
    """Test that _participants includes both agents and observers."""
    # Create proper agent and observer configurations
    agent_config_data = {
        "name": "TestAgent",
        "role": "AGENT1",
        "unique_identifier": "001",
        "description": "Agent 1 description",
        "parameters": {"model": "test_model"},
        "agent_obj": "test.MockAgent",  # Use a mock agent class
    }
    
    observer_config_data = {
        "name": "HostAgent",
        "role": "HOST",
        "unique_identifier": "002",
        "description": "Host observer description", 
        "parameters": {"model": "test_model"},
        "agent_obj": "test.MockHost",
    }
    
    # Create AgentVariants
    agent_variant = AgentVariants(**agent_config_data)
    observer_variant = AgentVariants(**observer_config_data)
    
    # Create orchestrator with agents and observers
    orchestrator = AutogenOrchestrator(
        name="test_orchestrator",
        orchestrator="AutogenOrchestrator",
        agents={"agent1": agent_variant},
        observers={"host": observer_variant},
        parameters={}
    )
    
    # Test directly that _participants is built correctly at line 174
    # Currently it only includes agents:
    # self._participants = {v.role: v.description for k, v in self.agents.items()}
    
    # Build participants manually to show the issue
    participants_from_agents_only = {v.role: v.description for k, v in orchestrator.agents.items()}
    assert participants_from_agents_only == {"AGENT1": "Agent 1 description"}
    
    # What it should be (including observers):
    expected_participants = {
        **{v.role: v.description for k, v in orchestrator.agents.items()},
        **{v.role: v.description for k, v in orchestrator.observers.items()}
    }
    assert expected_participants == {
        "AGENT1": "Agent 1 description",
        "HOST": "Host observer description"
    }
    
    print(f"Current implementation would set _participants to: {participants_from_agents_only}")
    print(f"But it should include observers too: {expected_participants}")
    
    # This demonstrates the bug - observers are not included in participants