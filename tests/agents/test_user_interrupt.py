"""Tests for the user interrupt feature in the Selector orchestrator."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from buttermilk._core.contract import ManagerResponse, StepRequest
from buttermilk.runner.selector import Selector


class TestUserInterrupt:
    """Tests for the user interrupt feature in the Selector orchestrator."""

    @pytest.mark.asyncio
    async def test_user_interrupt_flow(self):
        """Test that the user interrupt flag causes a re-request of host suggestions."""
        # Create a selector instance with mocked components
        selector = Selector(name="test_selector", description="Test Selector")

        # Mock the internal methods
        selector._setup = AsyncMock()
        selector._cleanup = AsyncMock()
        selector._get_host_suggestion = AsyncMock()
        selector._in_the_loop = AsyncMock()
        selector._execute_step = AsyncMock()

        # Set up the run to exit after 3 iterations
        iteration_count = 0

        async def mock_get_host_suggestion():
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 3:
                return StepRequest(role="END", prompt="End of test")
            return StepRequest(role="TEST", prompt=f"Test prompt {iteration_count}")

        selector._get_host_suggestion.side_effect = mock_get_host_suggestion

        # First call: User confirms without interrupt
        # Second call: User confirms with interrupt
        selector._in_the_loop.side_effect = [
            ManagerResponse(confirm=True, interrupt=False, prompt=None, halt=False, selection=None),  # First iteration - normal confirmation
            ManagerResponse(confirm=True, interrupt=True, prompt="User feedback", halt=False, selection=None),  # Second iteration - interrupt
        ]

        # Run the orchestrator
        await selector._run()

        # Verify that _get_host_suggestion was called 3 times (including the END step)
        assert selector._get_host_suggestion.call_count == 3

        # Verify that _execute_step was called only once
        # This is because the second iteration had interrupt=True which should skip execution
        assert selector._execute_step.call_count == 1

        # Verify the first _execute_step call used the first step
        step_arg = selector._execute_step.call_args[1]["step"]
        assert step_arg.role == "TEST"
        assert step_arg.prompt == "Test prompt 1"

    @pytest.mark.asyncio
    async def test_cli_user_agent_interrupt_response(self):
        """Test that the CLIUserAgent correctly sets the interrupt flag."""
        from buttermilk.agents.ui.console import CLIUserAgent

        # Create a CLIUserAgent instance
        agent = CLIUserAgent(role="USER", description="Test User Agent")

        # Mock the callback_to_groupchat
        callback_mock = AsyncMock()
        agent.callback_to_groupchat = callback_mock

        # Mock ainput to simulate user inputs
        with patch("buttermilk.agents.ui.console.ainput") as mock_ainput:
            # First scenario: User provides feedback then confirms with empty line
            mock_ainput.side_effect = ["Some feedback", ""]

            # Start the polling task
            task = asyncio.create_task(agent._poll_input())

            # Wait a bit for the task to process both inputs
            await asyncio.sleep(0.2)

            # Cancel the task to clean up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Check that the callback was called with interrupt=True
            assert callback_mock.call_count == 1
            response_arg = callback_mock.call_args[0][0]
            assert isinstance(response_arg, ManagerResponse)
            assert response_arg.confirm is True
            assert response_arg.interrupt is True
            assert response_arg.prompt == "Some feedback"

            # Reset the mock for the next scenario
            callback_mock.reset_mock()

            # Second scenario: User provides empty line (simple confirmation)
            mock_ainput.side_effect = [""]

            # Start the polling task
            task = asyncio.create_task(agent._poll_input())

            # Wait a bit for the task to process the input
            await asyncio.sleep(0.2)

            # Cancel the task to clean up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Check that the callback was called with interrupt=False
            assert callback_mock.call_count == 1
            response_arg = callback_mock.call_args[0][0]
            assert isinstance(response_arg, ManagerResponse)
            assert response_arg.confirm is True
            assert response_arg.interrupt is False
            assert response_arg.prompt is None
