"""Tests for the SelectorOrchestrator class."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from autogen_core import ClosureAgent, ClosureContext, MessageContext, SingleThreadedAgentRuntime, TopicId
from buttermilk._core.contract import AgentInput, AgentOutput, ManagerRequest, ManagerResponse, StepRequest
from buttermilk._core.types import Record, RunRequest
from buttermilk.runner.selector import SelectorOrchestrator


@pytest.fixture
def mock_runtime():
    """Mock Autogen runtime for testing."""
    runtime = MagicMock(spec=SingleThreadedAgentRuntime)
    runtime.publish_message = AsyncMock()
    runtime.send_message = AsyncMock()
    runtime.get = AsyncMock(return_value="test-agent-id")
    runtime.add_subscription = AsyncMock()
    return runtime


@pytest.fixture
def test_orchestrator():
    """Create a test orchestrator with minimal configuration."""
    return SelectorOrchestrator(
        name="test",
        description="Test orchestrator",
        agents={
            "host": {"variants": [{"id": "host", "role": "host", "type": "test"}]},
            "user": {"variants": [{"id": "user", "role": "user", "type": "test"}]},
            "judge": {"variants": [{"id": "judge1", "role": "judge", "type": "test"},
                                  {"id": "judge2", "role": "judge", "type": "test"}]},
        },
        params={"task": "Test task instructions"}
    )


@pytest.mark.asyncio
async def test_setup(test_orchestrator, mock_runtime):
    """Test the orchestrator setup properly initializes agents and queues."""
    with patch.object(test_orchestrator, '_runtime', mock_runtime):
        await test_orchestrator._setup()
        
        # Check that _register_agents was called
        mock_runtime.start.assert_called_once()
        
        # Verify host manager agent was properly set up
        assert test_orchestrator._host_agent is not None
        
        # Verify UI integration was properly set up
        assert test_orchestrator._user_confirmation is not None


@pytest.mark.asyncio
async def test_interactive_host_integration(test_orchestrator, mock_runtime):
    """Test that the host agent properly manages the conversation flow."""
    with patch.object(test_orchestrator, '_runtime', mock_runtime):
        await test_orchestrator._setup()
        
        # Mock host agent response
        step_request = StepRequest(role="judge", prompt="Test prompt", description="Test description")
        mock_runtime.send_message.return_value = AgentOutput(
            role="host",
            outputs=step_request
        )
        
        # Test get_next_step
        next_step = await test_orchestrator._get_next_step()
        assert next_step is not None
        assert next_step.role == "judge"
        assert next_step.prompt == "Test prompt"
        
        # Verify the host was asked for the next step
        mock_runtime.send_message.assert_called()


@pytest.mark.asyncio
async def test_ui_agent_interaction(test_orchestrator, mock_runtime):
    """Test that the UI agent properly interacts with the user."""
    with patch.object(test_orchestrator, '_runtime', mock_runtime):
        await test_orchestrator._setup()
        
        # Setup a step request
        step = StepRequest(role="judge", prompt="Test analysis", description="Analyze content")
        
        # Mock user confirmation
        with patch.object(test_orchestrator, '_wait_for_human', AsyncMock(return_value=True)):
            result = await test_orchestrator._in_the_loop(step)
            assert result is True
            
            # Verify UI agent was sent a message
            mock_runtime.publish_message.assert_called()
            
            # Check the message format (should be a ManagerRequest)
            args, kwargs = mock_runtime.publish_message.call_args
            assert len(args) > 0
            assert isinstance(args[0], ManagerRequest)


@pytest.mark.asyncio
async def test_variant_exploration(test_orchestrator, mock_runtime):
    """Test the ability to explore different agent variants sequentially."""
    with patch.object(test_orchestrator, '_runtime', mock_runtime):
        await test_orchestrator._setup()
        
        # Mock execution of a step
        step_input = AgentInput(
            prompt="Test prompt",
            inputs={"key": "value"},
            context=[],
            records=[]
        )
        
        # Mock response from agent
        mock_runtime.send_message.return_value = AgentOutput(
            role="judge1",
            outputs={"analysis": "Test analysis"}
        )
        
        # Execute the step using a specific variant
        output = await test_orchestrator._execute_step(
            step="judge",
            input=step_input,
            variant_index=0  # Use the first variant
        )
        
        # Verify the output
        assert output is not None
        assert output.role == "judge1"
        
        # Now try a different variant
        mock_runtime.send_message.return_value = AgentOutput(
            role="judge2",
            outputs={"analysis": "Different analysis"}
        )
        
        output = await test_orchestrator._execute_step(
            step="judge",
            input=step_input,
            variant_index=1  # Use the second variant
        )
        
        # Verify we got a different output
        assert output is not None
        assert output.role == "judge2"


@pytest.mark.asyncio
async def test_full_exploration_flow(test_orchestrator, mock_runtime):
    """Test a complete flow with user guidance affecting the path."""
    with patch.object(test_orchestrator, '_runtime', mock_runtime):
        await test_orchestrator._setup()
        
        # Mock host agent suggesting a step
        host_response = AgentOutput(
            role="host",
            outputs=StepRequest(role="judge", prompt="Initial analysis", description="Analyze content")
        )
        mock_runtime.send_message.return_value = host_response
        
        # Mock user confirming the step
        with patch.object(test_orchestrator, '_wait_for_human', AsyncMock(return_value=True)):
            # Get the next step
            step = await test_orchestrator._get_next_step()
            assert step is not None
            assert step.role == "judge"
            
            # Confirm the step with the user
            confirmed = await test_orchestrator._in_the_loop(step)
            assert confirmed is True
            
            # Now mock the judge execution
            mock_runtime.send_message.return_value = AgentOutput(
                role="judge1",
                outputs={"analysis": "Test analysis result"}
            )
            
            # Prepare and execute the step
            step_input = await test_orchestrator._prepare_step(step)
            output = await test_orchestrator._execute_step(step=step.role, input=step_input)
            
            # Verify the step execution
            assert output is not None
            assert output.role == "judge1"
            assert "analysis" in output.outputs
            
            # The orchestrator should track the exploration path
            assert len(test_orchestrator._exploration_path) > 0
            assert test_orchestrator._exploration_path[-1].startswith("judge")
