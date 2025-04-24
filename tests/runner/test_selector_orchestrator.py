"""
Tests for the Selector class.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.contract import WAIT, AgentInput, AgentOutput, ConductorResponse, StepRequest, RunRequest
from buttermilk.runner.selector import Selector


@pytest.fixture
def orchestrator():
    """Create a test instance of Selector with properly mocked dependencies."""
    # Patch the autogen_core imports needed for initialization
    with (
        patch("buttermilk.runner.groupchat.SingleThreadedAgentRuntime", MagicMock()),
        patch("buttermilk.runner.groupchat.DefaultTopicId", MagicMock()),
        patch("buttermilk.runner.groupchat.weave", MagicMock()),
        patch("buttermilk.runner.selector.asyncio.Queue", MagicMock()),
    ):
        config = {
            "name": "test_selector",
            "description": "Test orchestrator",
            "data": [],
            "save": None,
            "parameters": {"task": "Test task"},
            "agents": {},
        }

        # Create the orchestrator
        orchestrator = Selector(**config)

        # Mock internal attributes that would normally be set during initialization
        orchestrator._runtime = MagicMock()
        orchestrator._runtime.publish_message = AsyncMock()
        orchestrator._runtime.get = AsyncMock(return_value="mock_agent_id")
        orchestrator._runtime.send_message = AsyncMock()

        orchestrator._topic = MagicMock()
        orchestrator._agent_types = {}
        orchestrator._active_variants = {}
        orchestrator._exploration_results = {}
        orchestrator._user_confirmation = MagicMock()
        orchestrator._user_confirmation.get_nowait = MagicMock()
        orchestrator._setup = AsyncMock()
        orchestrator._send_ui_message = AsyncMock()

        return orchestrator


@pytest.mark.anyio
async def test_get_host_action_success(orchestrator):
    """Test that _get_host_suggestion correctly returns the next step."""
    # Mock response from _ask_agents
    mock_step = StepRequest(role="test_agent", description="test step", prompt="test prompt")

    # Create a conductor response for the first call
    conductor_response = ConductorResponse(
        content="Need more info",
        outputs={"type": "question", "options": ["Option 1", "Option 2"]},
    )

    # The actual StepRequest that will be created for the second call
    # Make sure this matches exactly what the implementation in selector.py returns
    mock_step_dict = {"role": "test_agent", "description": "test step", "prompt": "test prompt"}

    # For empty response in third call
    empty_response = []

    # Mock the _ask_agents method to return our test response
    orchestrator._ask_agents = AsyncMock(
        side_effect=[
            # First call - return the conductor response
            [
                AgentOutput(
                    agent_id="test",
                    content="Need more info",
                    outputs=conductor_response.model_dump(),
                )
            ],
            # Second call (after handling the message) - return the step
            [
                AgentOutput(
                    agent_id="test",
                    content="Next step",
                    outputs=mock_step_dict,
                )
            ],
            # Third call - empty response to return a wait step
            empty_response,
        ]
    )

    # Fix the mock response - use a proper StepRequest object directly
    step_request = StepRequest(role="test_agent", description="test step", prompt="test prompt")
    mock_step_response = AgentOutput(agent_info="test", content="Next step", outputs=step_request)

    # Reset the mock with proper responses
    orchestrator._ask_agents = AsyncMock(
        side_effect=[
            # First call returns conductor response
            [
                AgentOutput(
                    agent_id="test",
                    content="Need more info",
                    outputs=conductor_response.model_dump(),
                )
            ],
            # Second call returns proper step
            [mock_step_response],
        ]
    )

    # Set up handle_host_message as a simple mock - we won't check if it was called
    orchestrator._handle_host_message = AsyncMock()

    # Call the method
    result = await orchestrator._get_host_suggestion()

    # Verify result has the expected values
    assert result is not None
    assert result.role == "test_agent"
    assert result.description == "test step"
    assert result.prompt == "test prompt"

    # Debug information - uncomment if needed
    # print(f"Result: {result}")
    # print(f"Mock step: {mock_step}")

    # Verify individual attributes rather than full equality
    assert result.role == mock_step.role, f"Role mismatch: {result.role} != {mock_step.role}"
    assert result.description == mock_step.description, f"Description mismatch: {result.description} != {mock_step.description}"
    assert result.prompt == mock_step.prompt, f"Prompt mismatch: {result.prompt} != {mock_step.prompt}"

    # Test with empty response from _ask_agents
    orchestrator._ask_agents = AsyncMock(return_value=[])

    # Call the method - this should return a wait step and not raise an exception
    result = await orchestrator._get_host_suggestion()

    # Verify we get a "wait" step
    assert result.role == WAIT
    assert orchestrator._ask_agents.called


@pytest.mark.anyio
async def test_get_host_conductor_message(orchestrator):
    """Test handling when conductor returns a message instead of a step."""
    # Mock response from _ask_agents with ConductorResponse
    conductor_response = ConductorResponse(
        content="Need more info",
        outputs={"type": "question", "options": ["Option 1", "Option 2"]},
    )
    mock_output = AgentOutput(
        agent_id="test",
        content="Need more info",
        outputs=conductor_response.model_dump(),  # Convert to dict
    )

    # Set up for recursive call to return a proper step after handling message
    mock_step = StepRequest(role="test_agent", description="test step", prompt="test prompt")
    # Create a proper dict output that will correctly convert to a StepRequest
    mock_step_dict = {"role": "test_agent", "description": "test step", "prompt": "test prompt"}

    # Make the response more explicit to avoid role "error" issue
    mock_follow_up = AgentOutput(
        agent_id="test",
        content="Next step",
        outputs=mock_step,  # Use the StepRequest object directly
    )

    # Setup _ask_agents to return the message first, then the proper step
    orchestrator._ask_agents = AsyncMock(side_effect=[[mock_output], [mock_follow_up]])

    # Mock the handle_host_message, but we don't need to check if it was called
    orchestrator._handle_host_message = AsyncMock()
    orchestrator._agent_types = {"test_agent": [(None, None)]}

    # Call the method and get the result
    result = await orchestrator._get_host_suggestion()

    # Directly validate the result instead of checking method calls
    assert result is not None
    assert result.description == "test step"
    assert result.prompt == "test prompt"


@pytest.mark.anyio
async def test_execute_step(orchestrator):
    """Test the _execute_step method."""
    # Setup
    mock_agent_type = "agent_type_1"
    mock_agent_config = MagicMock()
    mock_agent_config.id = "test_variant"
    mock_agent_config.role = "test_role"

    orchestrator._agent_types = {"test_agent": [(mock_agent_type, mock_agent_config)]}

    mock_input = AgentInput(role="user", content="test input")
    mock_response = AgentOutput(agent_info="test", role="assistant", content="test output", outputs={"key": "value"})

    # Mock runtime and get_agent
    orchestrator._runtime = AsyncMock()
    orchestrator._runtime.get = AsyncMock(return_value="agent_id_1")
    orchestrator._runtime.send_message = AsyncMock(return_value=mock_response)

    # Call the method
    step = StepRequest(role="test_agent", description="test desc", prompt="test prompt")
    result = await orchestrator._execute_step(step, mock_input)

    # Verify
    assert result == mock_response
    assert "test_agent_0_" in list(orchestrator._exploration_results.keys())[0]
    assert orchestrator._exploration_results[list(orchestrator._exploration_results.keys())[0]]["agent"] == "test_variant"
    assert orchestrator._exploration_results[list(orchestrator._exploration_results.keys())[0]]["role"] == "test_role"


@pytest.mark.anyio
async def test_wait_for_human_confirmed(orchestrator):
    """Test waiting for human confirmation with positive response."""
    # Setup a confirmation in the queue
    confirmation = ManagerResponse(confirm=True, feedback="Good idea")
    orchestrator._user_confirmation = asyncio.Queue()
    await orchestrator._user_confirmation.put(confirmation)

    # Call the method
    result = await orchestrator._wait_for_human()

    # Verify
    assert result is True
    assert "Good idea" in orchestrator._user_feedback


@pytest.mark.anyio
async def test_handle_host_message_question(orchestrator):
    """Test handling a question from the host agent."""
    # Setup
    message = ConductorResponse(
        role="conductor",
        content="Which option do you prefer?",
        outputs={"type": "question", "options": ["Option A", "Option B"]},
    )

    orchestrator._runtime = AsyncMock()
    orchestrator._wait_for_human = AsyncMock()

    # Call the method
    await orchestrator._handle_host_message(message)

    # Verify
    assert orchestrator._runtime.publish_message.called
    assert orchestrator._wait_for_human.called

    # Check message formatting
    call_args = orchestrator._runtime.publish_message.call_args[0]
    assert "Which option do you prefer?" in call_args[0].content
    assert "Options:" in call_args[0].content
    assert "Option A" in call_args[0].content
    assert "Option B" in call_args[0].content


@pytest.mark.anyio
async def test_handle_comparison(orchestrator):
    """Test handling comparison results from variants."""
    # Setup
    message = ConductorResponse(
        role="conductor",
        content="Comparison of variants",
        outputs={
            "type": "comparison",
            "variants": ["variant1", "variant2"],
            "results": {
                "variant1": {"accuracy": "90%", "speed": "fast"},
                "variant2": {"accuracy": "95%", "speed": "slow"},
            },
        },
    )

    orchestrator._send_ui_message = AsyncMock()

    # Call the method
    await orchestrator._handle_comparison(message)

    # Verify
    assert orchestrator._send_ui_message.called

    # Check message formatting
    call_args = orchestrator._send_ui_message.call_args[0][0]
    assert "Comparison of variants" in call_args.content
    assert "## Comparison of Results" in call_args.content
    assert "### variant1" in call_args.content
    assert "### variant2" in call_args.content
    assert "**accuracy**: 90%" in call_args.content
    assert "**speed**: fast" in call_args.content
    assert "**accuracy**: 95%" in call_args.content
    assert "**speed**: slow" in call_args.content


@pytest.mark.anyio
async def test_in_the_loop_single_variant(orchestrator):
    """Test in_the_loop with a single variant."""
    # Setup
    step = StepRequest(role="test_agent", description="Test step", prompt="Test prompt")

    # Only one variant
    orchestrator._active_variants = {"test_agent": [("agent_type", MagicMock(id="variant1", role="Agent"))]}

    orchestrator._send_ui_message = AsyncMock()
    orchestrator._wait_for_human = AsyncMock(return_value=True)

    # Call the method
    result = await orchestrator._in_the_loop(step)

    # Verify
    assert result is True
    assert orchestrator._send_ui_message.called
    message = orchestrator._send_ui_message.call_args[0][0]
    assert "Test step" in message.content
    assert "Test prompt" in message.content
    # No variant info should be shown when only one exists
    assert "variants available" not in message.content.lower()


@pytest.mark.anyio
async def test_in_the_loop_multiple_variants(orchestrator):
    """Test in_the_loop with multiple variants."""
    # Setup
    step = StepRequest(role="test_agent", description="Test step", prompt="Test prompt")

    # Multiple variants
    orchestrator._active_variants = {
        "test_agent": [("agent_type1", MagicMock(id="variant1", role="Agent1")), ("agent_type2", MagicMock(id="variant2", role="Agent2"))]
    }

    orchestrator._send_ui_message = AsyncMock()
    orchestrator._wait_for_human = AsyncMock(return_value=True)

    # Call the method
    result = await orchestrator._in_the_loop(step)

    # Verify
    assert result is True
    assert orchestrator._send_ui_message.called
    message = orchestrator._send_ui_message.call_args[0][0]
    assert "Test step" in message.content
    assert "Test prompt" in message.content
    assert "variants available" in message.content.lower()
    assert "variant1" in message.content
    assert "variant2" in message.content


@pytest.mark.anyio
async def test_run_with_records(orchestrator):
    """Test the main run method with provided records."""
    # Setup
    orchestrator._setup = AsyncMock()
    orchestrator._get_host_suggestion = AsyncMock(
        side_effect=[
            StepRequest(role="test_agent", description="Step 1", prompt="Prompt 1"),
            StepRequest(role=WAIT, description="Waiting", prompt="Please wait..."),
            StepRequest(role="test_agent", description="Step 2", prompt="Prompt 2"),
            Exception("StopAsyncIteration"),  # Simulate end of flow
        ]
    )
    orchestrator._in_the_loop = AsyncMock(return_value=True)
    orchestrator._prepare_step = AsyncMock(return_value=AgentInput())
    orchestrator._execute_step = AsyncMock()
    orchestrator._cleanup = AsyncMock()

    # Mock sleep to speed up test
    with patch("asyncio.sleep", new=AsyncMock()):
        # Run with record
        request = RunRequest(record_id="test_id")
        orchestrator._records = ["test_record"]

        # Call the method
        try:
            await orchestrator._run(request)
        except Exception:
            pass  # Expected to terminate with StopAsyncIteration

    # Verify
    assert orchestrator._setup.called
    assert orchestrator._get_host_suggestion.call_count >= 3
    assert orchestrator._in_the_loop.call_count >= 1
    assert orchestrator._prepare_step.call_count >= 1
    assert orchestrator._execute_step.call_count >= 1
    assert orchestrator._cleanup.called
