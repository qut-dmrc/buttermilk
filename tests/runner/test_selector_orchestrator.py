"""
Tests for the SelectorOrchestrator.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from buttermilk._core.contract import (
    AgentOutput,
    ConductorRequest,
    ManagerMessage,
    ManagerRequest,
    StepRequest,
)
from buttermilk._core.types import RunRequest
from buttermilk.runner.selector import SelectorOrchestrator


@pytest.fixture
def selector_config():
    """Basic config for testing the SelectorOrchestrator."""
    return {
        "name": "test_selector",
        "description": "Test selector orchestrator",
        "data": [],
        "agents": {
            "test": [
                (MagicMock(), MagicMock(id="test_agent", role="test")),
            ],
            "conductor": [
                (MagicMock(), MagicMock(id="conductor_agent", role="conductor")),
            ],
        },
        "params": {"task": "Test task"},
    }


class MockConfirmation:
    """Mock for confirmation queue."""

    def __init__(self, confirm=True, halt=False):
        self.confirm = confirm
        self.halt = halt


@pytest.mark.anyio
async def test_selector_init(selector_config):
    """Test that the SelectorOrchestrator initializes correctly."""
    orchestrator = SelectorOrchestrator(**selector_config)
    assert orchestrator.name == "test_selector"
    assert orchestrator.description == "Test selector orchestrator"
    assert len(orchestrator._agent_types) == 2  # test and conductor


@pytest.mark.anyio
async def test_setup(selector_config):
    """Test that the setup correctly initializes variants."""
    orchestrator = SelectorOrchestrator(**selector_config)

    # Mock the runtime for testing
    orchestrator._runtime = AsyncMock()
    orchestrator._user_confirmation = asyncio.Queue()
    orchestrator._topic = "test_topic"

    await orchestrator._setup()

    # Verify active variants were initialized
    assert len(orchestrator._active_variants) == 2
    assert "test" in orchestrator._active_variants
    assert "conductor" in orchestrator._active_variants

    # Verify welcome message was published
    assert orchestrator._runtime.publish_message.called
    args = orchestrator._runtime.publish_message.call_args[0]
    assert isinstance(args[0], ManagerMessage)
    assert "started" in args[0].content.lower()
    assert "test_selector" in args[0].content.lower()


@pytest.mark.anyio
async def test_wait_for_human(selector_config):
    """Test the wait_for_human method."""
    orchestrator = SelectorOrchestrator(**selector_config)
    orchestrator._user_confirmation = asyncio.Queue()

    # Put a confirmation in the queue
    confirmation = MockConfirmation(confirm=True)
    await orchestrator._user_confirmation.put(confirmation)

    # Get the confirmation
    result = await orchestrator._wait_for_human(timeout=1)
    assert result is True

    # Put a negative confirmation in the queue
    confirmation = MockConfirmation(confirm=False)
    await orchestrator._user_confirmation.put(confirmation)

    # Get the confirmation
    result = await orchestrator._wait_for_human(timeout=1)
    assert result is False

    # Test timeout
    with patch("asyncio.Queue.get_nowait", side_effect=asyncio.QueueEmpty):
        with patch("asyncio.sleep", return_value=None):
            result = await orchestrator._wait_for_human(timeout=0.1)
            assert result is False


@pytest.mark.anyio
async def test_in_the_loop(selector_config):
    """Test the in_the_loop method for user interaction."""
    orchestrator = SelectorOrchestrator(**selector_config)
    orchestrator._user_confirmation = asyncio.Queue()
    orchestrator._send_ui_message = AsyncMock()
    orchestrator._wait_for_human = AsyncMock(return_value=True)

    # Set up active variants
    orchestrator._active_variants = {
        "test": [
            (MagicMock(), MagicMock(id="test_agent1", role="test")),
            (MagicMock(), MagicMock(id="test_agent2", role="test")),
        ]
    }

    # Create a step request
    step = StepRequest(role="test", description="Test step", prompt="Test prompt")

    # Call the method
    result = await orchestrator._in_the_loop(step)

    # Verify message was sent to UI
    assert orchestrator._send_ui_message.called
    args = orchestrator._send_ui_message.call_args[0]
    assert isinstance(args[0], ManagerRequest)
    assert "test_agent1" in args[0].content
    assert "test_agent2" in args[0].content

    # Verify we got the user confirmation
    assert result is True


@pytest.mark.anyio
async def test_get_next_step(selector_config):
    """Test getting the next step from the conductor."""
    orchestrator = SelectorOrchestrator(**selector_config)
    orchestrator._topic = "test_topic"

    # Mock the _ask_agents method to return a step
    step = StepRequest(role="test", description="Test step", prompt="Test prompt")
    orchestrator._ask_agents = AsyncMock(return_value=[AgentOutput(outputs=step)])

    # Set up active variants
    orchestrator._active_variants = {
        "test": [
            (MagicMock(), MagicMock(id="test_agent1", role="test")),
            (MagicMock(), MagicMock(id="test_agent2", role="test")),
        ],
        "conductor": [
            (MagicMock(), MagicMock(id="conductor_agent", role="conductor")),
        ],
    }

    # Call the method
    result = await orchestrator._get_next_step()

    # Verify conductor was asked with correct context
    assert orchestrator._ask_agents.called
    args = orchestrator._ask_agents.call_args[0]
    assert args[0] == "conductor"
    message = args[1]
    assert isinstance(message, ConductorRequest)
    assert "task" in message.inputs
    assert "exploration_path" in message.inputs
    assert "available_agents" in message.inputs
    assert "results" in message.inputs

    # Verify we got the step back
    assert result == step


@pytest.mark.anyio
async def test_execute_step(selector_config):
    """Test executing a step with a specific agent variant."""
    orchestrator = SelectorOrchestrator(**selector_config)

    # Set up agent types
    agent_mock = MagicMock()
    agent_config_mock = MagicMock(id="test_agent", role="test")
    orchestrator._agent_types = {"test": [(agent_mock, agent_config_mock)]}

    # Mock the runtime
    orchestrator._runtime = AsyncMock()
    orchestrator._runtime.get.return_value = "test_agent_id"
    orchestrator._runtime.send_message.return_value = AgentOutput(outputs={"result": "Test result"})

    # Call the method
    result = await orchestrator._execute_step("test", MagicMock())

    # Verify agent was executed
    assert orchestrator._runtime.get.called
    assert orchestrator._runtime.send_message.called

    # Verify exploration path was updated
    assert len(orchestrator._exploration_path) == 1
    assert orchestrator._exploration_path[0].startswith("test_0_")

    # Verify exploration results were tracked
    assert len(orchestrator._exploration_results) == 1
    assert "test_agent" in orchestrator._exploration_results[orchestrator._exploration_path[0]]["agent"]
    assert "test" in orchestrator._exploration_results[orchestrator._exploration_path[0]]["role"]
    assert "result" in orchestrator._exploration_results[orchestrator._exploration_path[0]]["outputs"]


@pytest.mark.anyio
async def test_fetch_record(selector_config):
    """Test fetching a record."""
    with patch("buttermilk.agents.fetch.FetchRecord") as MockFetchRecord:
        # Setup mock return
        mock_fetch = AsyncMock()
        mock_output = MagicMock()
        mock_output.results = {"test": "result"}
        mock_fetch._run.return_value = mock_output
        MockFetchRecord.return_value = mock_fetch

        # Create orchestrator and request
        orchestrator = SelectorOrchestrator(**selector_config)
        orchestrator.data = [{"name": "test"}]
        request = RunRequest(record_id="test123", uri=None, prompt=None)

        # Call the method
        await orchestrator._fetch_record(request)

        # Verify FetchRecord was created with list of data
        MockFetchRecord.assert_called_once()
        args = MockFetchRecord.call_args[1]
        assert isinstance(args["data"], list)

        # Verify fetch was run
        assert mock_fetch._run.called
        assert orchestrator._records == {"test": "result"}


@pytest.mark.anyio
async def test_run_with_record_id(selector_config):
    """Test running with a record ID."""
    orchestrator = SelectorOrchestrator(**selector_config)

    # Mock necessary methods
    orchestrator._setup = AsyncMock()
    orchestrator._fetch_record = AsyncMock()
    orchestrator._get_next_step = AsyncMock(
        side_effect=[
            StepRequest(role="test", description="Test step", prompt="Test prompt"),
            asyncio.CancelledError(),  # Raise to stop the loop after first iteration
        ]
    )
    orchestrator._in_the_loop = AsyncMock(return_value=True)
    orchestrator._prepare_step = AsyncMock()
    orchestrator._execute_step = AsyncMock()
    orchestrator._cleanup = AsyncMock()

    # Create a request
    request = RunRequest(record_id="test123", uri=None, prompt=None)

    # Call the method
    with pytest.raises(asyncio.CancelledError):
        await orchestrator._run(request)

    # Verify methods were called in correct order
    orchestrator._setup.assert_called_once()
    orchestrator._fetch_record.assert_called_once_with(request)
    orchestrator._get_next_step.assert_called_once()
    orchestrator._in_the_loop.assert_called_once()
    orchestrator._prepare_step.assert_called_once()
    orchestrator._execute_step.assert_called_once()
    orchestrator._cleanup.assert_called_once()
