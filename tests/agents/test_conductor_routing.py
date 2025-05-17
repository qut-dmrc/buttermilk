"""Tests the internal routing logic of conductor agents, ensuring ConductorRequest
is handled correctly by delegating to step-choosing methods.
"""

from unittest.mock import AsyncMock, patch

import pytest

# Buttermilk core types
from buttermilk._core.contract import (
    AgentTrace,
    ConductorRequest,
    StepRequest,
)
from buttermilk.agents.flowcontrol.explorer import ExplorerHost
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.flowcontrol.llmhost import LLMHostAgent

pytestmark = pytest.mark.anyio


@pytest.fixture
def conductor_request() -> ConductorRequest:
    """Provides a sample ConductorRequest."""
    return ConductorRequest(inputs={"test": "data"}, prompt="test prompt")


class TestConductorRouting:
    """Tests focus on routing ConductorRequest within conductor agents."""

    async def test_sequencer_handle_events_routes_conductor_request(self, conductor_request: ConductorRequest):
        """Test Sequencer._handle_events calls _get_next_step for ConductorRequest."""
        # Arrange: Create a Sequencer instance with mocked dependencies
        # Patch __init__ to avoid dependencies, then add mocks manually
        with patch.object(HostAgent, "__init__", return_value=None):
            sequencer = HostAgent()
            # Mock methods called by _handle_events
            sequencer._get_next_step = AsyncMock(name="_get_next_step")
            sequencer._check_completions = AsyncMock(name="_check_completions")
            # Add necessary attributes if _handle_events reads them (e.g., _current_step_name)
            sequencer._current_step_name = "some_step"

        # Act: Call the method under test
        await sequencer._handle_events(conductor_request)

        # Assert: Verify _get_next_step was called correctly
        sequencer._check_completions.assert_called_once()  # Verify completion check happens
        sequencer._get_next_step.assert_called_once_with(message=conductor_request)

    async def test_host_handle_events_routes_conductor_request(self, conductor_request: ConductorRequest):
        """Test LLMHostAgent._handle_events calls _get_next_step for ConductorRequest."""
        # Arrange: Create LLMHostAgent instance with mocks
        with patch.object(LLMHostAgent, "__init__", return_value=None):
            host = LLMHostAgent()
            host._get_next_step = AsyncMock(name="_get_next_step")
            host._check_completions = AsyncMock(name="_check_completions")  # If LLMHostAgent uses it
            host._current_step_name = "some_step"  # Add needed attributes

        # Act
        await host._handle_events(conductor_request)

        # Assert
        # host._check_completions.assert_called_once() # Uncomment if LLMHostAgent uses it
        host._get_next_step.assert_called_once_with(message=conductor_request)

    async def test_explorer_handle_events_routes_conductor_request(self, conductor_request: ConductorRequest):
        """Test ExplorerHost._handle_events calls _get_next_step (inherited)."""
        # Arrange: Create ExplorerHost instance with mocks
        with patch.object(ExplorerHost, "__init__", return_value=None):
            explorer = ExplorerHost()
            # Explorer inherits from LLMHostAgent, mock same methods
            explorer._get_next_step = AsyncMock(name="_get_next_step")
            explorer._check_completions = AsyncMock(name="_check_completions")  # If inherited/used
            explorer._current_step_name = "some_step"  # Add needed attributes

        # Act
        await explorer._handle_events(conductor_request)

        # Assert: Behavior should be same as LLMHostAgent
        # explorer._check_completions.assert_called_once() # Uncomment if applicable
        explorer._get_next_step.assert_called_once_with(message=conductor_request)

    async def test_explorer_choose_calls_process(self, conductor_request: ConductorRequest):
        """Test that ExplorerHost._choose calls _process."""
        # Arrange: Create ExplorerHost instance with mocked _process
        with patch.object(ExplorerHost, "__init__", return_value=None):
            explorer = ExplorerHost()
            explorer._process = AsyncMock(
                name="_process", return_value=AgentTrace(agent_info="test", outputs=StepRequest(role="ANY", prompt="", description="")),
            )
            # Add any attributes _choose might need

        # Act: Call the method under test. Assuming _choose takes 'message' arg.
        # If it takes 'inputs', adjust the call signature.
        await explorer._choose(message=conductor_request)

        # Assert: Verify _process was called correctly
        explorer._process.assert_called_once()
        call_args, call_kwargs = explorer._process.call_args
        assert call_kwargs.get("message") == conductor_request  # Check message arg
