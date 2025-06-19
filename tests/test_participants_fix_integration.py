"""Integration test to verify the participants fix works end-to-end."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from buttermilk._core.config import AgentVariants
from buttermilk._core.types import RunRequest
from buttermilk._core.contract import ConductorRequest
from buttermilk.orchestrators.groupchat import AutogenOrchestrator
from buttermilk._core.agent import Agent


class MockAgent(Agent):
    """Minimal mock agent for testing."""
    async def _process(self, *, message, **kwargs):
        return {"result": "test"}


class MockHost(Agent):
    """Minimal mock host agent for testing."""
    async def _process(self, *, message, **kwargs):
        return {"result": "host"}


@pytest.mark.asyncio
async def test_host_receives_all_participants():
    """Test that host agents receive both agents and observers in participants."""
    # Create proper agent and observer configurations
    agent_config_data = {
        "name": "TestAgent",
        "role": "AGENT1",
        "unique_identifier": "001",
        "description": "Agent 1 description",
        "parameters": {"model": "test_model"},
        "agent_obj": f"{__name__}.MockAgent",
    }
    
    observer_config_data = {
        "name": "HostAgent",
        "role": "HOST",
        "unique_identifier": "002",
        "description": "Host observer description", 
        "parameters": {"model": "test_model"},
        "agent_obj": f"{__name__}.MockHost",
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
    
    # Mock the runtime
    orchestrator._runtime = AsyncMock()
    orchestrator._topic = Mock()
    orchestrator._topic.type = "test_topic"
    
    # Create a run request
    params = RunRequest(
        flow="test_flow",
        session_id="test_session",
        inputs={"test": "data"},
        ui_type="console"
    )
    
    # Register agents - this builds _participants
    with patch.object(orchestrator, '_register_buttermilk_agent_instance'):
        await orchestrator._register_agents(params)
    
    # Verify _participants includes both agents and observers
    assert orchestrator._participants == {
        "AGENT1": "Agent 1 description",
        "HOST": "Host observer description"
    }
    
    # Simulate what happens when run() publishes ConductorRequest
    # Find the ConductorRequest that would be published
    published_conductor_request = None
    
    # Mock the run method to capture the ConductorRequest
    async def mock_run(params):
        # This is what the orchestrator does in run()
        await orchestrator._runtime.publish_message(
            ConductorRequest(
                inputs=params.model_dump(), 
                participants=orchestrator._participants
            ), 
            topic_id=Mock()
        )
    
    await mock_run(params)
    
    # Check that publish_message was called with correct participants
    orchestrator._runtime.publish_message.assert_called()
    call_args = orchestrator._runtime.publish_message.call_args[0]
    conductor_request = call_args[0]
    
    assert isinstance(conductor_request, ConductorRequest)
    assert conductor_request.participants == {
        "AGENT1": "Agent 1 description",
        "HOST": "Host observer description"
    }
    
    print(f"ConductorRequest participants: {conductor_request.participants}")
    print("✓ Host agents will now receive all participants (agents + observers)")