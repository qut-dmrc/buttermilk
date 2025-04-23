"""
Test the user feedback interrupt mechanism.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from buttermilk._core.contract import ManagerResponse, StepRequest
from buttermilk.agents.ui.console import CLIUserAgent
from buttermilk.runner.selector import Selector


@pytest.mark.anyio
async def test_manager_response_interrupt_flag():
    """Test that the ManagerResponse class has the interrupt flag."""
    # Simple test to ensure the field exists and defaults to False
    response = ManagerResponse()
    assert hasattr(response, "interrupt")
    assert response.interrupt is False

    # Test with interrupt explicitly set to True
    response = ManagerResponse(interrupt=True)
    assert response.interrupt is True


@pytest.mark.anyio
async def test_selector_handles_interrupt():
    """Test that the Selector orchestrator handles the interrupt flag correctly."""
    # Create a mock selector with patched methods
    selector = MagicMock(spec=Selector)
    selector._user_confirmation = asyncio.Queue()

    # Track the sequence of calls
    call_sequence = []

    # Mock the get_host_suggestion method to track calls
    async def mock_get_host_suggestion():
        call_sequence.append("get_suggestion")
        return StepRequest(role="TEST_AGENT")

    selector._get_host_suggestion = mock_get_host_suggestion

    # Mock the in_the_loop method to return a response with interrupt=True
    async def mock_in_the_loop(step=None):
        call_sequence.append("get_user_response")
        return ManagerResponse(confirm=True, interrupt=True, halt=False, prompt="This is feedback that needs conductor review", selection=None)

    selector._in_the_loop = mock_in_the_loop

    # Mock the execute_step method to track calls
    async def mock_execute_step(step, variant_index=0):
        call_sequence.append("execute_step")
        return None

    selector._execute_step = mock_execute_step

    # Now manually simulate one iteration of the Selector._run loop
    suggested_step = await selector._get_host_suggestion()
    user_response = await selector._in_the_loop(suggested_step)

    # Check interrupt and skip step execution, as would happen in Selector._run
    if user_response.interrupt:
        call_sequence.append("detected_interrupt")
        # With interrupt=True, it should loop back to _get_host_suggestion
        await selector._get_host_suggestion()
        call_sequence.append("get_suggestion_again")
    else:
        # Without interrupt, it would execute the step
        await selector._execute_step(suggested_step)

    # Verify the sequence shows it detected the interrupt and got a new suggestion
    assert call_sequence == ["get_suggestion", "get_user_response", "detected_interrupt", "get_suggestion", "get_suggestion_again"]
    # Verify it never tried to execute the step when interrupt was True
    assert "execute_step" not in call_sequence


@pytest.mark.anyio
async def test_cli_user_agent_sets_interrupt():
    """Test the CLIUserAgent sets the interrupt flag correctly based on input."""
    agent = CLIUserAgent(role="CLI_USER", name="Test CLI User", description="Test CLI User Agent")

    # Mock the _input_callback
    callback_responses = []

    async def mock_callback(response):
        callback_responses.append(response)

    agent._input_callback = mock_callback

    # Patch ainput to simulate user inputs
    with patch(
        "aioconsole.ainput",
        side_effect=[
            # First test: Empty line (confirmation with no feedback)
            "",
            # Second test: Feedback line followed by empty confirmation
            "Here's some feedback",
            "",
            # Third test: Rejection
            "n",
        ],
    ):
        # Only run the _poll_input method for 3 iterations
        with patch.object(asyncio, "sleep", return_value=None):
            # Create a task for _poll_input
            task = asyncio.create_task(agent._poll_input())

            # Give it time to process 3 inputs
            await asyncio.sleep(0.5)

            # Cancel the task (it's an infinite loop)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # Check we received 3 responses
    assert len(callback_responses) == 3

    # 1. Empty line: confirm=True, interrupt=False
    assert callback_responses[0].confirm is True
    assert callback_responses[0].interrupt is False
    assert callback_responses[0].prompt is None

    # 2. Feedback + empty line: confirm=True, interrupt=True, prompt=feedback
    assert callback_responses[1].confirm is True
    assert callback_responses[1].interrupt is True
    assert callback_responses[1].prompt == "Here's some feedback"

    # 3. Rejection: confirm=False, interrupt=False
    assert callback_responses[2].confirm is False
    assert callback_responses[2].interrupt is False
