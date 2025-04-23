"""
Tests for the base Orchestrator concepts and potentially specific implementations like BatchOrchestrator.
"""

from unittest.mock import AsyncMock
import pytest

# Buttermilk core imports
from buttermilk._core.contract import AgentInput, StepRequest  # Needed for type hints potentially
from buttermilk._core.orchestrator import Orchestrator  # Base class
from buttermilk._core.variants import AgentVariants
from buttermilk._core.types import RunRequest  # Import RunRequest

# Specific orchestrator implementation being tested here
from buttermilk.runner.batch import BatchOrchestrator


@pytest.fixture
def simple_batch_orchestrator() -> BatchOrchestrator:
    """Creates a simple BatchOrchestrator instance for testing."""
    # Using BatchOrchestrator as it was in the original test file.
    # Note: This doesn't test AutogenOrchestrator or Selector specifically.
    return BatchOrchestrator(
        name="test_batch_flow",
        description="Test Batch Flow",
        agents={
            "STEP1": AgentVariants(  # Use uppercase role names as keys (consistent with validator)
                role="step1",  # Role within variants can be lowercase
                name="Step 1 Agent",
                description="Agent for Step 1",
                # Add agent_obj if BatchOrchestrator requires it for instantiation
            ),
            "STEP2": AgentVariants(
                role="step2",
                name="Step 2 Agent",
                description="Agent for Step 2",
            ),
        },
        params={"flow_param": "flow_value"},
        # Define a simple sequence for BatchOrchestrator if needed
        sequence=["STEP1", "STEP2"],  # BatchOrchestrator likely needs a sequence defined
    )


@pytest.mark.anyio
async def test_orchestrator_initialization(simple_batch_orchestrator: BatchOrchestrator):
    """Test that BatchOrchestrator initializes correctly."""
    assert simple_batch_orchestrator.name == "test_batch_flow"
    assert simple_batch_orchestrator.description == "Test Batch Flow"
    # Orchestrator validator converts keys to uppercase
    assert "STEP1" in simple_batch_orchestrator.agents
    assert "STEP2" in simple_batch_orchestrator.agents
    assert simple_batch_orchestrator.params == {"flow_param": "flow_value"}
    # Check sequence if applicable to BatchOrchestrator
    assert hasattr(simple_batch_orchestrator, "sequence")
    assert simple_batch_orchestrator.sequence == ["STEP1", "STEP2"]


@pytest.mark.anyio
async def test_orchestrator_call_method(simple_batch_orchestrator: BatchOrchestrator):
    """Test that calling the orchestrator instance invokes its run method."""
    # Mock the run method to check if __call__ delegates correctly
    simple_batch_orchestrator.run = AsyncMock()
    test_request = RunRequest(prompt="start run")

    # Call the orchestrator instance
    await simple_batch_orchestrator(request=test_request)

    # Assert that the run method was called once with the correct request
    simple_batch_orchestrator.run.assert_called_once_with(request=test_request)


# Removed outdated tests:
# - test_prepare_step_basic
# - test_prepare_step_with_history
# - test_prepare_step_message
# - test_parse_history
# (These tested methods that are no longer part of the base Orchestrator)
