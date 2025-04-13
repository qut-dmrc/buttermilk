from unittest.mock import AsyncMock

import pytest

from buttermilk._core.contract import AgentInput, StepRequest
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.variants import AgentVariants


class SimpleOrchestrator(Orchestrator):
    """Simple implementation of Orchestrator for testing."""

    async def run(self, request=None):
        """Simple implementation that just runs steps in sequence."""
        self.run_called = True
        self.request = request


@pytest.fixture
def simple_orchestrator():
    """Create a simple orchestrator for testing."""
    return SimpleOrchestrator(
        flow_name="test_flow",
        description="Test flow",
        agents={
            "step1": AgentVariants(
                description="Step 1",
                inputs={"key1": "value1"},
            ),
            "step2": AgentVariants(
                description="Step 2",
                inputs={"key2": "value2", "history": True},
            ),
        },
        params={"param1": "value1"},
    )


@pytest.mark.asyncio
async def test_orchestrator_initialization(simple_orchestrator):
    """Test that orchestrator initializes correctly."""
    assert simple_orchestrator.flow_name == "test_flow"
    assert simple_orchestrator.description == "Test flow"
    assert "step1" in simple_orchestrator.agents
    assert "step2" in simple_orchestrator.agents
    assert simple_orchestrator.params == {"param1": "value1"}


@pytest.mark.asyncio
async def test_prepare_step_basic(simple_orchestrator):
    """Test that _prepare_step returns the expected inputs."""
    inputs = await simple_orchestrator._prepare_step(StepRequest(role="step1"))
    assert "key1" in inputs
    assert inputs["key1"] == "value1"


@pytest.mark.asyncio
async def test_prepare_step_with_history(simple_orchestrator):
    """Test that _prepare_step handles special variables correctly."""
    # Add some history
    simple_orchestrator.history = ["message1", "message2"]

    inputs = await simple_orchestrator._prepare_step(StepRequest(role="step2"))
    assert "key2" in inputs
    assert inputs["key2"] == "value2"
    assert "history" in inputs
    assert inputs["history"] == "message1\nmessage2"


@pytest.mark.asyncio
async def test_prepare_step_message(simple_orchestrator):
    """Test that _prepare_step creates a proper AgentInput."""
    step = StepRequest(
        role="step1",
        prompt="Test prompt",
        source="test_source",
        arguments={"extra_input": "extra_value"},
    )
    message = await simple_orchestrator._prepare_step(step)

    assert isinstance(message, AgentInput)
    assert message.role == "test_flow"
    assert message.source == "test_source"
    assert message.content == "Test prompt"
    assert "key1" in message.inputs
    assert message.inputs["key1"] == "value1"
    assert "extra_input" in message.inputs
    assert message.inputs["extra_input"] == "extra_value"
    assert "prompt" in message.inputs
    assert message.inputs["prompt"] == "Test prompt"


@pytest.mark.asyncio
async def test_call_method(simple_orchestrator):
    """Test that calling the orchestrator invokes run."""
    simple_orchestrator.run = AsyncMock()
    await simple_orchestrator("test_request")

    simple_orchestrator.run.assert_called_once_with(request="test_request")


@pytest.mark.asyncio
async def test_parse_history():
    """Test the _parse_history validator."""
    # Test with list of strings
    result = Orchestrator._parse_history(["message1", "message2"])
    assert result == ["message1", "message2"]

    # Test with list of dicts
    result = Orchestrator._parse_history([
        {"type": "user", "content": "hello"},
        {"type": "bot", "content": "hi there"},
    ])
    assert result == ["user: hello", "bot: hi there"]

    # Test with mixed
    result = Orchestrator._parse_history([
        "plain message",
        {"type": "user", "content": "dict message"},
    ])
    assert result == ["plain message", "user: dict message"]
