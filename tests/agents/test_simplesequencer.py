
import pytest

from buttermilk._core.contract import END, AgentInput, AgentOutput, ConductorRequest, StepRequest
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.flowcontrol.llmhost import LLMHostAgent


@pytest.mark.anyio
async def test_sequencer_initialization():
    """Test that Sequencer initializes correctly."""
    agent = Sequencer(role="sequencer", name="Test Sequencer", description="A test sequencer")

    await agent.initialize()
    assert agent.role == "sequencer"
    assert "Test Sequencer" in agent.name  # Checks generated name

    # Process a basic input to test initialization and the _process path
    test_input = AgentInput(prompt="Hello", inputs={"task": "Test task"})  # Add task as _process checks it
    result = await agent._process(message=test_input)  # Pass as 'message' kwarg
    assert isinstance(result, AgentOutput)
    assert result.outputs is not None
    assert isinstance(result.outputs, dict)
    assert "status" in result.outputs
    assert "Sequencer received" in result.outputs["status"]  # Check status field in outputs dict


@pytest.mark.anyio
async def test_sequencer_round_robin():
    """Test that Sequencer produces a round-robin sequence of steps."""
    agent = Sequencer(role="sequencer", name="Test Sequencer", description="A test sequencer")

    await agent.initialize()

    # Create a conductor request with participants
    participants = {
        "AGENT1": {"role": "agent1", "description": "First agent"},
        "AGENT2": {"role": "agent2", "description": "Second agent"},
        "AGENT3": {"role": "agent3", "description": "Third agent"},
    }

    request = ConductorRequest()
    request = ConductorRequest(inputs={"participants": participants})

    # Get the first step
    step1_response = await agent._get_next_step(message=request)  # Pass as 'message' kwarg
    assert isinstance(step1_response, AgentOutput), "Response should be AgentOutput"
    assert isinstance(step1_response.outputs, StepRequest), "outputs should contain StepRequest"

    # Store the first agent selected - ensure we're working with StepRequest
    step1 = step1_response.outputs
    assert isinstance(step1, StepRequest)
    first_agent = step1.role
    # Exclude Sequencer and MANAGER roles from expected sequence
    expected_sequence = [role for role in participants if role not in [agent.role.upper(), "MANAGER"]]

    # Ensure the first step is one of the expected participants
    assert first_agent in expected_sequence, f"First agent '{first_agent}' should be in {expected_sequence}"

    # Get the second step
    step2_response = await agent._get_next_step(message=request)  # Pass as 'message' kwarg
    step2 = step2_response.outputs
    assert isinstance(step2, StepRequest)
    second_agent = step2.role
    assert second_agent in expected_sequence, f"Second agent '{second_agent}' should be in {expected_sequence}"

    # The second agent should be different from the first (round-robin)
    assert second_agent != first_agent, f"Round-robin failed: Second agent '{second_agent}' is same as first '{first_agent}'"

    # Get the third step
    step3_response = await agent._get_next_step(message=request)  # Pass as 'message' kwarg
    step3 = step3_response.outputs
    assert isinstance(step3, StepRequest)
    third_agent = step3.role
    assert third_agent in expected_sequence, f"Third agent '{third_agent}' should be in {expected_sequence}"

    # The third agent should be different from the first two
    assert third_agent != first_agent and third_agent != second_agent, f"Round-robin failed: Third agent '{third_agent}' matches previous"

    # Get the fourth step - should be END after one cycle
    step4_response = await agent._get_next_step(message=request)  # Pass as 'message' kwarg
    step4 = step4_response.outputs
    assert isinstance(step4, StepRequest)
    fourth_agent_role = step4.role

    # Sequencer implementation yields END after one full round.
    assert fourth_agent_role == END, f"Expected END after one round-robin cycle, got '{fourth_agent_role}'"
