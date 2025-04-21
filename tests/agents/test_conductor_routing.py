import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import inspect

from buttermilk._core.contract import (
    ConductorRequest,
    StepRequest,
    AgentOutput,
)
from buttermilk.agents.flowcontrol.sequencer import Sequencer
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.agents.flowcontrol.explorer import ExplorerHost

pytestmark = pytest.mark.anyio


class TestConductorRouting:
    """Tests focusing specifically on the ConductorRequest routing to _get_next_step."""

    async def test_sequencer_handle_events_routes_conductor_request(self):
        """Test that Sequencer._handle_events calls _get_next_step when given a ConductorRequest."""
        # Get the source code for the method
        code = inspect.getsource(Sequencer._handle_events)

        # Verify the code contains call to _get_next_step when message is ConductorRequest
        assert "if isinstance(message, ConductorRequest)" in code
        assert "next_step = await self._get_next_step(inputs=message)" in code

        # Create a test instance that won't throw errors
        with patch.object(Sequencer, '__init__', return_value=None):
            sequencer = Sequencer()
            sequencer._get_next_step = AsyncMock()
            sequencer._check_completions = AsyncMock()

            # Create a test request
            request = ConductorRequest(inputs={"test": "data"}, prompt="test")

            # Call the method
            await sequencer._handle_events(request)

            # Verify _get_next_step was called with the right arguments
            sequencer._get_next_step.assert_called_once_with(message=request)

    async def test_host_handle_events_routes_conductor_request(self):
        """Test that LLMHostAgent._handle_events calls _get_next_step when given a ConductorRequest."""
        # Get the source code for the method
        code = inspect.getsource(LLMHostAgent._handle_events)

        # Verify the code contains call to _get_next_step when message is ConductorRequest
        assert "if isinstance(message, ConductorRequest)" in code
        assert "next_step = await self._get_next_step(message=message)" in code

        # Create a test instance that won't throw errors
        with patch.object(LLMHostAgent, '__init__', return_value=None):
            host = LLMHostAgent()
            host._get_next_step = AsyncMock()
            host._check_completions = AsyncMock()

            # Create a test request
            request = ConductorRequest(inputs={"test": "data"}, prompt="test")

            # Call the method
            await host._handle_events(request)

            # Verify _get_next_step was called with the right arguments
            host._get_next_step.assert_called_once_with(message=request)

    async def test_explorer_handle_events_routes_conductor_request(self):
        """Test that ExplorerHost._handle_events calls _get_next_step when given a ConductorRequest."""
        # ExplorerHost inherits _handle_events from LLMHostAgent, so we check that it's the same
        explorer_code = ExplorerHost._handle_events
        host_code = LLMHostAgent._handle_events

        # Verify it's the same method
        assert explorer_code == host_code, "ExplorerHost._handle_events should be the same as LLMHostAgent._handle_events"

        # Create a test instance that won't throw errors
        with patch.object(ExplorerHost, '__init__', return_value=None):
            explorer = ExplorerHost()
            explorer._get_next_step = AsyncMock()
            explorer._check_completions = AsyncMock()

            # Create a test request
            request = ConductorRequest(inputs={"test": "data"}, prompt="test")

            # Call the method
            await explorer._handle_events(request)

            # Verify _get_next_step was called with the right arguments
            explorer._get_next_step.assert_called_once_with(message=request)

    async def test_explorer_choose_calls_process(self):
        """Test that ExplorerHost._choose calls _process."""
        # Get the source code for the method
        code = inspect.getsource(ExplorerHost._choose)

        # Verify the code contains call to _process
        assert "step = await self._process(message=inputs)" in code

        # Create a test instance that won't throw errors
        with patch.object(ExplorerHost, '__init__', return_value=None):
            explorer = ExplorerHost()
            explorer._process = AsyncMock()

            # Create a test request
            request = ConductorRequest(inputs={"test": "data"}, prompt="test")

            # Call the method
            await explorer._choose(inputs=request)

            # Verify _process was called with the right arguments
            explorer._process.assert_called_once_with(message=request)
