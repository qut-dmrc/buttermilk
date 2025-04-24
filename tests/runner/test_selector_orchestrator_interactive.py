"""
Tests for the interactive features of SelectorOrchestrator.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from buttermilk._core.contract import (
    AgentOutput,
    ConductorRequest,
    ConductorResponse,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
)
from buttermilk._core.types import Record
from buttermilk._core.variants import AgentVariants
from buttermilk.runner.selector import Selector


@pytest.fixture
def selector_config():
    """Basic config for testing the Selector."""
    # Create mock AgentVariants objects
    test_variants = MagicMock(spec=AgentVariants)
    test_variants.get_configs.return_value = [
        (MagicMock(), MagicMock(id="test_agent1", role="test")),
        (MagicMock(), MagicMock(id="test_agent2", role="test")),
    ]

    conductor_variants = MagicMock(spec=AgentVariants)
    conductor_variants.get_configs.return_value = [(MagicMock(), MagicMock(id="conductor_agent", role="conductor"))]

    ui_variants = MagicMock(spec=AgentVariants)
    ui_variants.get_configs.return_value = [(MagicMock(), MagicMock(id="ui_agent", role="ui"))]

    return {
        "name": "test_selector",
        "description": "Test selector orchestrator",
        "data": [],
        "agents": {
            "test": test_variants,
            "conductor": conductor_variants,
            "ui": ui_variants,
        },
        "parameters": {"task": "Test task"},
    }


@pytest.mark.anyio
async def test_interactive_user_feedback(selector_config):
    """Test interactive user feedback influencing next steps."""
    orchestrator = Selector(**selector_config)

    # Mock runtime and confirmation queue
    orchestrator._runtime = AsyncMock()
    orchestrator._user_confirmation = asyncio.Queue()
    orchestrator._topic = MagicMock()
    orchestrator._send_ui_message = AsyncMock()

    # Mock _ask_agents to simulate conductor responses
    # First response: conductor suggests a step
    # Second response: conductor incorporates user feedback
    step1 = StepRequest(role="test", description="Initial test step", prompt="Test prompt 1")
    step2 = StepRequest(role="test", description="Adjusted test step", prompt="Test prompt 2")

    orchestrator._ask_agents = AsyncMock(
        side_effect=[
            [AgentOutput(agent_info="test", outputs=step1)],  # First call returns step1
            [AgentOutput(agent_info="test", outputs=step2)],  # Second call returns step2
        ]
    )

    # First iteration: get step, receive confirmation with feedback
    next_step = await orchestrator._get_host_suggestion()
    assert next_step == step1
    assert orchestrator._ask_agents.call_count == 1

    # Put confirmation with feedback in queue
    feedback = "Try a different approach with more depth"
    await orchestrator._user_confirmation.put(ManagerResponse(confirm=True, feedback=feedback, selection="test_agent2"))

    # Check the in_the_loop method picks up the feedback
    # Make sure next_step is not None before calling _in_the_loop
    assert next_step is not None
    result = await orchestrator._in_the_loop(next_step)
    assert result is True

    # Now when we execute the step, it should use the selected variant
    orchestrator._execute_step = AsyncMock()
    step_input = MagicMock()
    await orchestrator._execute_step(step=next_step)

    # Verify variant selection was used (index 1 for test_agent2)
    orchestrator._execute_step.assert_called_once_with(step=next_step, variant_index=1)

    # Next iteration: get step again, should incorporate user feedback
    next_step = await orchestrator._get_host_suggestion()
    assert next_step == step2
    assert orchestrator._ask_agents.call_count == 2

    # Verify feedback was passed to the conductor
    conductor_request = orchestrator._ask_agents.call_args_list[1][0][1]
    assert isinstance(conductor_request, ConductorRequest)
    assert "user_feedback" in conductor_request.inputs
    assert conductor_request.inputs["user_feedback"] == feedback


@pytest.mark.anyio
async def test_host_interaction_with_ui(selector_config):
    """Test the host agent (conductor) interacting with UI agent."""
    orchestrator = Selector(**selector_config)

    # Mock runtime and components
    orchestrator._runtime = AsyncMock()
    orchestrator._user_confirmation = asyncio.Queue()
    orchestrator._topic = MagicMock()

    # Set up a record
    test_record = Record(id="test123", content="Test content")
    orchestrator._records = [test_record]

    # Set up some exploration history
    orchestrator._exploration_path = ["test_0_abcd", "test_1_efgh"]
    orchestrator._exploration_results = {
        "test_0_abcd": {"agent": "test_agent1", "outputs": {"result": "First analysis"}},
        "test_1_efgh": {"agent": "test_agent2", "outputs": {"result": "Second analysis"}},
    }

    # Mock publish_message to capture conductor-ui interactions
    orchestrator._runtime.publish_message = AsyncMock()

    # Mock host agent asking a question (via conductor response)
    host_question = "Would you like to compare these two analyses or explore a different approach?"
    conductor_message = ConductorResponse(
        content=host_question,
        outputs={"type": "question", "options": ["Compare analyses", "Try different approach", "End exploration"]},
    )

    # Inject the host message directly
    await orchestrator._handle_host_message(conductor_message)

    # Verify question was relayed to UI
    orchestrator._runtime.publish_message.assert_called_once()
    args = orchestrator._runtime.publish_message.call_args[0]

    # First arg should be a ManagerRequest with the question
    assert isinstance(args[0], ManagerRequest)
    assert host_question in args[0].content
    assert "Compare analyses" in args[0].content

    # Second arg should be the topic ID
    assert args[1] == orchestrator._topic

    # User selects "Compare analyses"
    await orchestrator._user_confirmation.put(ManagerResponse(confirm=True, selection="Compare analyses"))

    # Wait a bit for async processing
    await asyncio.sleep(0.1)

    # Verify that we processed the response
    assert orchestrator._last_user_selection == "Compare analyses"


@pytest.mark.anyio
async def test_variant_comparison(selector_config):
    """Test comparing results from different variants."""
    orchestrator = Selector(**selector_config)

    # Create a comparison message that would be sent by the conductor
    comparison_request = ConductorResponse(
        role="conductor",
        content="Here's a comparison of the results from both agents:",
        outputs={
            "type": "comparison",
            "variants": ["test_agent1", "test_agent2"],
            "results": {
                "test_agent1": {"key_finding": "High sentiment score", "confidence": 0.85},
                "test_agent2": {"key_finding": "Mixed sentiment with caveats", "confidence": 0.72},
            },
        },
    )

    # Mock methods
    orchestrator._runtime = AsyncMock()
    orchestrator._send_ui_message = AsyncMock()

    # Process the comparison
    await orchestrator._handle_comparison(comparison_request)

    # Verify a formatted comparison was sent to UI
    orchestrator._send_ui_message.assert_called_once()
    ui_message = orchestrator._send_ui_message.call_args[0][0]

    # Message should contain both variant results
    assert isinstance(ui_message, ManagerMessage)
    assert "test_agent1" in ui_message.content
    assert "test_agent2" in ui_message.content
    assert "High sentiment score" in ui_message.content
    assert "Mixed sentiment with caveats" in ui_message.content
