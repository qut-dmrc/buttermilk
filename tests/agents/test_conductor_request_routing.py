import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock, call

from buttermilk._core.contract import (
    ConductorRequest,
    StepRequest,
    AgentOutput,
)
from buttermilk.agents.flowcontrol.sequencer import Sequencer
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.agents.flowcontrol.explorer import ExplorerHost

# This is the same as using the @pytest.mark.anyio on all test functions in the module
pytestmark = pytest.mark.anyio


@pytest.fixture
def conductor_request():
    """Create a sample ConductorRequest object for testing."""
    return ConductorRequest(
        inputs={
            "participants": {
                "AGENT1": {"config": "some_config"},
                "AGENT2": {"config": "some_config"},
            },
            "task": "test task",
        },
        prompt="Test prompt",
    )


class TestConductorRequestRouting:
    """Tests that specifically check the routing of ConductorRequests to _get_next_step."""

    async def test_sequencer_handle_events_routes_conductor_request(self, conductor_request):
        """Test that Sequencer._handle_events routes ConductorRequest to _get_next_step."""
        # Create a mocked Sequencer instance
        # Mock the required methods
        with patch("buttermilk.agents.flowcontrol.sequencer.Sequencer._get_next_step") as mock_get_next_step:
            with patch("buttermilk.agents.flowcontrol.sequencer.Sequencer._check_completions"):
                # Set up mock response
                mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
                mock_get_next_step.return_value = mock_response

                # Call the method under test directly with mocked input and output
                # Note: we use self=mock_get_next_step as a workaround to test the method directly
                result = await Sequencer._handle_events(mock_get_next_step, conductor_request)

                # Check if the mock was called properly
                mock_get_next_step.assert_called_once_with(inputs=conductor_request)

            # The result should be the mock response
            assert result == mock_response

    async def test_host_handle_events_routes_conductor_request(self, conductor_request):
        """Test that LLMHostAgent._handle_events routes ConductorRequest to _get_next_step."""
        # Create a mocked LLMHostAgent instance
        # Mock the required methods
        with patch("buttermilk.agents.flowcontrol.host.LLMHostAgent._get_next_step") as mock_get_next_step:
            with patch("buttermilk.agents.flowcontrol.host.LLMHostAgent._check_completions"):
                # Set up mock response
                mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
                mock_get_next_step.return_value = mock_response

                # Call the method under test directly with mocked input and output
                # Note: we use self=mock_get_next_step as a workaround to test the method directly
                result = await LLMHostAgent._handle_events(mock_get_next_step, conductor_request)

                # Check if the mock was called properly
                mock_get_next_step.assert_called_once_with(inputs=conductor_request)

            # The result should be the mock response
            assert result == mock_response

    async def test_explorer_handle_events_routes_conductor_request(self, conductor_request):
        """Test that ExplorerHost._handle_events routes ConductorRequest to _get_next_step."""
        # Create a mocked ExplorerHost instance
        # Mock the required methods
        with patch("buttermilk.agents.flowcontrol.explorer.ExplorerHost._get_next_step") as mock_get_next_step:
            with patch("buttermilk.agents.flowcontrol.host.LLMHostAgent._check_completions"):
                # Set up mock response
                mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
                mock_get_next_step.return_value = mock_response

                # Call the method under test directly with mocked input and output
                # Note: we use self=mock_get_next_step as a workaround to test the method directly
                result = await ExplorerHost._handle_events(mock_get_next_step, conductor_request)

                # Check if the mock was called properly
                mock_get_next_step.assert_called_once_with(message=conductor_request)

            # The result should be the mock response
            assert result == mock_response

    async def test_explorer_choose_routes_to_process(self, conductor_request):
        """Test that ExplorerHost._choose routes to _process method."""
        # Create a mocked ExplorerHost instance
        # Mock the required method
        with patch("buttermilk.agents.flowcontrol.explorer.ExplorerHost._process") as mock_process:
            # Set up mock response
            step_request = StepRequest(role="AGENT1", prompt="", description="Test step")
            mock_process.return_value = step_request

            # Call the method under test directly with mocked input and output
            # Note: we use self=mock_process as a workaround to test the method directly
            result = await ExplorerHost._choose(mock_process, message=conductor_request)

            # Check if the mock was called properly
            mock_process.assert_called_once_with(message=conductor_request)

            # The result should be the mock step request
            assert result == step_request
