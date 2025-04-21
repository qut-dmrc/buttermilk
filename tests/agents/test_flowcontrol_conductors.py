import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from buttermilk._core.contract import (
    ConductorRequest,
    ConductorResponse,
    StepRequest,
    AgentOutput,
    END,
    WAIT,
)
from buttermilk.agents.flowcontrol.sequencer import Sequencer
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.agents.flowcontrol.explorer import ExplorerHost


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


class TestSequencer:
    """Tests for Sequencer handling of ConductorRequest objects."""

    @pytest.mark.asyncio
    async def test_sequencer_handle_events_conductor_request(self, conductor_request):
        """
        Test that Sequencer._handle_events correctly handles ConductorRequest
        and routes it to _get_next_step.
        """
        sequencer = Sequencer(role="SEQUENCER", description="Test Sequencer")
        # Mock _get_next_step to avoid its implementation
        sequencer._get_next_step = AsyncMock()
        mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
        sequencer._get_next_step.return_value = mock_response

        # Call _handle_events with a ConductorRequest
        result = await sequencer._handle_events(conductor_request)

        # Verify _get_next_step was called with the request
        sequencer._get_next_step.assert_called_once_with(inputs=conductor_request)
        # Verify the result is the response from _get_next_step
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_sequencer_get_next_step(self, conductor_request):
        """
        Test that Sequencer._get_next_step returns an AgentOutput with a StepRequest.
        """
        sequencer = Sequencer(role="SEQUENCER")

        # Initialize with a mocked step generator that returns a fixed StepRequest
        async def mock_generator():
            yield StepRequest(role="AGENT1", prompt="", description="Test step")

        sequencer._step_generator = mock_generator()
        sequencer._step_completion_event = asyncio.Event()
        sequencer._step_completion_event.set()  # Ensure event is set to avoid waiting

        # Call _get_next_step
        result = await sequencer._get_next_step(conductor_request)

        # Verify the result is an AgentOutput with a StepRequest
        assert isinstance(result, AgentOutput)
        assert isinstance(result.outputs, StepRequest)
        assert result.outputs.role == "AGENT1"

        # Verify participants were set from the request
        assert sequencer._participants == conductor_request.inputs.get("participants")

        # Verify the completion tracking was reset
        assert sequencer._current_step_name == "AGENT1"
        assert len(sequencer._expected_agents_current_step) == 0
        assert len(sequencer._completed_agents_current_step) == 0
        assert not sequencer._step_completion_event.is_set()


class TestLLMHostAgent:
    """Tests for LLMHostAgent handling of ConductorRequest objects."""

    @pytest.mark.asyncio
    async def test_host_handle_events_conductor_request(self, conductor_request):
        """
        Test that LLMHostAgent._handle_events correctly handles ConductorRequest
        and routes it to _get_next_step.
        """
        host = LLMHostAgent(role="HOST", description="Test Host")
        # Mock _get_next_step to avoid its implementation
        host._get_next_step = AsyncMock()
        mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
        host._get_next_step.return_value = mock_response

        # Call _handle_events with a ConductorRequest
        result = await host._handle_events(conductor_request)

        # Verify _get_next_step was called with the request
        host._get_next_step.assert_called_once_with(inputs=conductor_request)
        # Verify the result is the response from _get_next_step
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_host_get_next_step(self, conductor_request):
        """
        Test that LLMHostAgent._get_next_step returns an AgentOutput with a StepRequest.
        """
        host = LLMHostAgent(role="HOST")

        # Initialize with a mocked step generator that returns a fixed StepRequest
        async def mock_generator():
            yield StepRequest(role="AGENT1", prompt="", description="Test step")

        host._step_generator = mock_generator()
        host._step_completion_event = asyncio.Event()
        host._step_completion_event.set()  # Ensure event is set to avoid waiting

        # Call _get_next_step
        result = await host._get_next_step(conductor_request)

        # Verify the result is an AgentOutput with a StepRequest
        assert isinstance(result, AgentOutput)
        assert isinstance(result.outputs, StepRequest)
        assert result.outputs.role == "AGENT1"

        # Verify participants were set from the request
        assert host._participants == conductor_request.inputs.get("participants")

        # Verify the completion tracking was reset
        assert host._current_step_name == "AGENT1"
        assert len(host._expected_agents_current_step) == 0
        assert len(host._completed_agents_current_step) == 0
        assert not host._step_completion_event.is_set()

    @pytest.mark.asyncio
    async def test_host_avoid_self_call(self, conductor_request):
        """
        Test that LLMHostAgent._get_next_step avoids calling itself.
        """
        host = LLMHostAgent(role="AGENT1", description="Test Host")  # Same role as the next step

        # Mock _choose to return StepRequest with role=AGENT1 (same as host)
        async def mock_choose(*args, **kwargs):
            return StepRequest(role="AGENT1", prompt="", description="Test step")

        host._choose = mock_choose

        # Set up test conditions
        host._step_completion_event = asyncio.Event()
        host._step_completion_event.set()

        # Call _get_next_step - should avoid calling itself
        with patch.object(
            host,
            "_choose",
            side_effect=[
                StepRequest(role="AGENT1", prompt="", description="Test step"),  # First call returns AGENT1
                StepRequest(role="AGENT2", prompt="", description="Test step"),  # Second call returns AGENT2
            ],
        ):
            result = await host._get_next_step(conductor_request)

            # Should return WAIT since the second choice is not in participants
            assert isinstance(result, AgentOutput)
            assert isinstance(result.outputs, StepRequest)
            assert result.outputs.role == WAIT


class TestExplorerHost:
    """Tests for ExplorerHost handling of ConductorRequest objects."""

    @pytest.mark.asyncio
    async def test_explorer_handle_events_conductor_request(self, conductor_request):
        """
        Test that ExplorerHost._handle_events correctly handles ConductorRequest
        and routes it to _get_next_step.
        """
        explorer = ExplorerHost(role="EXPLORER", description="Test Explorer")
        # Mock _get_next_step to avoid its implementation
        explorer._get_next_step = AsyncMock()
        mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
        explorer._get_next_step.return_value = mock_response

        # Call _handle_events with a ConductorRequest
        result = await explorer._handle_events(conductor_request)

        # Verify _get_next_step was called with the request
        explorer._get_next_step.assert_called_once_with(inputs=conductor_request)
        # Verify the result is the response from _get_next_step
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_explorer_get_next_step(self, conductor_request):
        """
        Test that ExplorerHost._get_next_step calls _process method.
        """
        explorer = ExplorerHost(role="EXPLORER")
        # Mock _process to return a StepRequest
        explorer._process = AsyncMock()
        mock_step = StepRequest(role="AGENT1", prompt="", description="Test step")
        explorer._process.return_value = mock_step

        # Initialize required attributes
        explorer._step_completion_event = asyncio.Event()
        explorer._step_completion_event.set()

        # Patch the parent _get_next_step method to avoid its implementation
        with patch.object(LLMHostAgent, "_get_next_step") as mock_parent_get_next_step:
            mock_response = AgentOutput(outputs=mock_step)
            mock_parent_get_next_step.return_value = mock_response

            # Call _get_next_step
            result = await explorer._get_next_step(conductor_request)

            # Verify _process was called
            explorer._process.assert_called_once()

            # Verify the result is from the parent method
            assert result == mock_response


@pytest.mark.asyncio
async def test_integration_with_orchestrator():
    """
    Integration test: simulate interactions between orchestrator's ConductorRequest and agents.
    """
    # Create agents
    sequencer = Sequencer(role="SEQUENCER", description="Test Sequencer")
    host = LLMHostAgent(role="HOST", description="Test Host")
    explorer = ExplorerHost(role="EXPLORER", description="Test Explorer")

    # Initialize them
    await sequencer.initialize()
    await host.initialize()
    await explorer.initialize()

    # Create a conductor request from orchestrator
    conductor_request = ConductorRequest(
        inputs={
            "participants": {
                "AGENT1": {"config": "some_config"},
                "AGENT2": {"config": "some_config"},
                "END": {"config": "some_config"},
            },
            "task": "integration test",
        },
        prompt="Integration test",
    )

    # Test each agent's handling of the request
    sequencer_response = await sequencer._handle_events(conductor_request)
    host_response = await host._handle_events(conductor_request)
    explorer_response = await explorer._handle_events(conductor_request)

    # Verify all responses are ConductorResponse objects with StepRequest outputs
    for response in [sequencer_response, host_response, explorer_response]:
        assert isinstance(response, AgentOutput)
        assert isinstance(response.outputs, StepRequest)
        # Response should return a step that's in participants or END
        assert response.outputs.role in ["AGENT1", "AGENT2", "END", WAIT]
