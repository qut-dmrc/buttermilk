import pytest
import asyncio
from buttermilk._core.contract import AgentInput, StepRequest, ConductorRequest, AgentOutput, END, WAIT
from buttermilk.agents.flowcontrol.simple_sequencer import SimpleSequencerAgent


@pytest.mark.anyio
async def test_simple_sequencer_initialization():
    """Test that SimpleSequencerAgent initializes correctly."""
    agent = SimpleSequencerAgent(role="sequencer", name="Test Sequencer", description="A test sequencer")

    await agent.initialize()
    assert agent.role == "sequencer"
    assert "Test Sequencer" in agent.name

    # Process a basic input to test initialization output
    result = await agent._process(inputs=AgentInput(prompt="Hello"))
    assert isinstance(result, AgentOutput)
    assert result.content is not None
    assert "SimpleSequencerAgent received" in str(result.content)


@pytest.mark.anyio
async def test_simple_sequencer_round_robin():
    """Test that SimpleSequencerAgent produces a round-robin sequence of steps."""
    agent = SimpleSequencerAgent(role="sequencer", name="Test Sequencer", description="A test sequencer")

    await agent.initialize()

    # Create a conductor request with participants
    participants = {
        "AGENT1": {"role": "agent1", "description": "First agent"},
        "AGENT2": {"role": "agent2", "description": "Second agent"},
        "AGENT3": {"role": "agent3", "description": "Third agent"},
    }

    request = ConductorRequest()
    request.inputs = {"participants": participants}

    # Get the first step
    step1_response = await agent._get_next_step(inputs=request)
    assert isinstance(step1_response, AgentOutput)
    assert isinstance(step1_response.outputs, StepRequest)

    # Store the first agent selected - ensure we're working with StepRequest
    step1 = step1_response.outputs
    assert isinstance(step1, StepRequest)
    first_agent = step1.role
    assert first_agent in participants or first_agent == WAIT or first_agent == END, f"First agent '{first_agent}' should be in participants or END"

    # Get the second step
    step2_response = await agent._get_next_step(inputs=request)
    step2 = step2_response.outputs
    assert isinstance(step2, StepRequest)
    second_agent = step2.role
    assert (
        second_agent in participants or first_agent == WAIT or second_agent == END
    ), f"Second agent '{second_agent}' should be in participants or END"

    # The second agent should be different from the first (round-robin)
    assert second_agent != first_agent, "Round-robin should select a different agent"

    # Get the third step
    step3_response = await agent._get_next_step(inputs=request)
    step3 = step3_response.outputs
    assert isinstance(step3, StepRequest)
    third_agent = step3.role
    assert third_agent in participants or first_agent == WAIT or third_agent == END, f"Third agent '{third_agent}' should be in participants or END"

    # The third agent should be different from the first two
    assert third_agent != first_agent and third_agent != second_agent, "Round-robin should select a different agent"

    # Get the fourth step - should cycle back to the first agent or be END
    step4_response = await agent._get_next_step(inputs=request)
    step4 = step4_response.outputs
    assert isinstance(step4, StepRequest)
    fourth_agent = step4.role

    # After cycling through all three agents, we should either get END or start over
    assert fourth_agent == END or fourth_agent == first_agent, f"Expected END or first agent, got '{fourth_agent}'"
