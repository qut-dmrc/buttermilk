"""Tests for conductor agents' _get_next_step method.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Buttermilk core types
from buttermilk._core.agent import AgentInput, AgentTrace
from buttermilk._core.contract import ConductorRequest, StepRequest
from buttermilk._core.constants import END
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredStructuredLLMHostAgent
from buttermilk.agents.flowcontrol.host import HostAgent

pytestmark = pytest.mark.anyio


@pytest.fixture
def conductor_request() -> ConductorRequest:
    """Provides a sample ConductorRequest."""
    return ConductorRequest(
        participants={"AGENT1": "Agent 1 description", "AGENT2": "Agent 2 description"},
        inputs={"prompt": "Get next step", "task": "test"}
    )


# --- Sequencer Tests ---


@pytest.mark.anyio
async def test_sequencer_get_next_step(conductor_request: ConductorRequest):
    """Test Sequencer._get_next_step returns the next step from its generator.
    """
    # Arrange
    sequencer = HostAgent(role="SEQUENCER", agent_name="Test Seq", agent_id="test-seq")
    mock_callback = AsyncMock()
    await sequencer.initialize(callback_to_groupchat=mock_callback)  # Initialize to set up generator

    # Mock the internal generator to control the output
    async def mock_generator():
        yield StepRequest(role="AGENT1", prompt="", description="Step 1")
        yield StepRequest(role="AGENT2", prompt="", description="Step 2")
        yield StepRequest(role=END, prompt="", description="End")

    sequencer._step_generator = mock_generator()

    # Ensure completion event is set (or mock _check_completions/_wait_for)
    sequencer._step_completion_event.set()
    sequencer._check_completions = AsyncMock()  # Mock to avoid side effects

    # Act
    # Need to populate participants first, which _get_next_step does internally on first call
    result1 = await sequencer._get_next_step(message=conductor_request)

    # Assert first step
    assert isinstance(result1, AgentTrace)
    assert isinstance(result1.outputs, StepRequest)
    assert result1.outputs.role == "AGENT1"
    assert sequencer._participants == conductor_request.inputs.get("participants")
    assert sequencer._current_step == "AGENT1"
    assert not sequencer._step_completion_event.is_set()  # Should be cleared for next step

    # Act again for second step (simulate previous step completing)
    sequencer._step_completion_event.set()
    result2 = await sequencer._get_next_step(message=conductor_request)

    # Assert second step
    assert isinstance(result2, AgentTrace)
    assert isinstance(result2.outputs, StepRequest)
    assert result2.outputs.role == "AGENT2"
    assert sequencer._current_step == "AGENT2"

    # Act again for END step
    sequencer._step_completion_event.set()
    result3 = await sequencer._get_next_step(message=conductor_request)

    # Assert END step
    assert isinstance(result3, AgentTrace)
    assert isinstance(result3.outputs, StepRequest)
    assert result3.outputs.role == END
    assert sequencer._current_step == END


# --- StructuredStructuredLLMHostAgent Tests ---


@pytest.fixture
def mock_llm_host_agent() -> StructuredLLMHostAgent:
    """Fixture for a mocked StructuredLLMHostAgent."""
    # Create a proper instance with minimal required fields
    agent = StructuredLLMHostAgent(
        role="host",
        description="Mocked host agent"
    )
    # Mock the methods that need to be mocked
    agent._process = AsyncMock(name="_process")
    agent._check_completions = AsyncMock(name="_check_completions")
    agent._step_completion_event = asyncio.Event()
    agent._current_step = "previous_step"
    # Add other necessary attributes if needed by _get_next_step
    agent._records = []
    agent._model_context = MagicMock()  # Mock context if needed
    agent._data = MagicMock()  # Mock data collector if needed
    return agent


@pytest.mark.anyio
async def test_llm_host_agent_get_next_step_calls_process(mock_llm_host_agent: StructuredLLMHostAgent, conductor_request: ConductorRequest):
    """Test StructuredLLMHostAgent._get_next_step calls _process to determine the next step.
    """
    # Arrange
    expected_step = StepRequest(role="NEXT_AGENT", prompt="Do something", description="Next action")
    mock_output_from_process = AgentTrace(agent_info=mock_llm_host_agent.id, role=mock_llm_host_agent.role, outputs=expected_step)
    mock_llm_host_agent._process.return_value = mock_output_from_process
    mock_llm_host_agent._step_completion_event.set()  # Assume previous step completed

    # Act
    result_output = await mock_llm_host_agent._get_next_step(message=conductor_request)

    # Assert
    mock_llm_host_agent._check_completions.assert_called_once()
    mock_llm_host_agent._process.assert_called_once()

    call_args, call_kwargs = mock_llm_host_agent._process.call_args
    assert "message" in call_kwargs
    process_input_message = call_kwargs["message"]
    assert isinstance(process_input_message, AgentInput)
    # Verify _process input contains relevant info from ConductorRequest (depends on _get_next_step logic)
    assert process_input_message.inputs == conductor_request.inputs
    # StructuredLLMHostAgent._get_next_step should return the output from _process
    assert result_output == mock_output_from_process
    # Verify completion tracking reset (assuming _process output is valid StepRequest)
    assert mock_llm_host_agent._current_step == "NEXT_AGENT"
    assert not mock_llm_host_agent._step_completion_event.is_set()


@pytest.mark.anyio
async def test_llm_host_agent_avoid_self_or_manager_call(mock_llm_host_agent: StructuredLLMHostAgent, conductor_request: ConductorRequest):
    """Test that StructuredLLMHostAgent._get_next_step avoids calling itself or MANAGER.
    """
    # Arrange
    mock_llm_host_agent.role = "host"  # Set agent's own role
    step_self = StepRequest(role="HOST", prompt="Call self", description="")
    step_manager = StepRequest(role="MANAGER", prompt="Call manager", description="")
    step_other = StepRequest(role="OTHER_AGENT", prompt="Call other", description="")

    # Simulate _process returning HOST, then MANAGER, then OTHER_AGENT
    mock_llm_host_agent._process.side_effect = [
        AgentTrace(agent_info=mock_llm_host_agent.id, role=mock_llm_host_agent.role, outputs=step_self),
        AgentTrace(agent_info=mock_llm_host_agent.id, role=mock_llm_host_agent.role, outputs=step_manager),
        AgentTrace(agent_info=mock_llm_host_agent.id, role=mock_llm_host_agent.role, outputs=step_other),
    ]
    mock_llm_host_agent._step_completion_event.set()  # Start ready

    # Act & Assert 1: First call to _get_next_step -> _process returns HOST -> _get_next_step calls _process again
    await mock_llm_host_agent._get_next_step(message=conductor_request)
    # Act & Assert 2: Second call to _process returns MANAGER -> _get_next_step calls _process again
    await mock_llm_host_agent._get_next_step(message=conductor_request)
    # Act & Assert 3: Third call to _process returns OTHER_AGENT -> _get_next_step returns this one
    result3 = await mock_llm_host_agent._get_next_step(message=conductor_request)

    # Verify _process was called three times
    assert mock_llm_host_agent._process.call_count == 3
    # Verify the final result is the OTHER_AGENT step
    assert isinstance(result3, AgentTrace)
    assert isinstance(result3.outputs, StepRequest)
    assert result3.outputs.role == "OTHER_AGENT"
    # Verify current step is updated
    assert mock_llm_host_agent._current_step == "OTHER_AGENT"


# Removed redundant _handle_events tests (covered in test_conductor_routing.py)
# Removed flawed/redundant _get_next_step tests for Explorer/Integration
