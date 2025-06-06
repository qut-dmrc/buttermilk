"""Tests for AutogenAgentAdapter integration with Autogen runtime."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Autogen imports
from autogen_core import DefaultTopicId, MessageContext, SingleThreadedAgentRuntime

# Buttermilk core types
from buttermilk._core.agent import AgentInput, AgentTrace
from buttermilk._core.config import AgentConfig

# Specific agent classes
from buttermilk.agents.judge import Judge, Reasons

# The adapter under test
from buttermilk.libs.autogen import AutogenAgentAdapter

# --- Test Fixtures ---


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Provides a basic AgentConfig."""
    # Ensure a model is specified, needed by LLMAgent subclasses like Judge
    return AgentConfig(
        role="judge",
        name="Test Judge Agent",
        description="A test judge agent",
        parameters={"model": "mock_llm_model"},  # Mock model is fine if _process is mocked
    )


@pytest.fixture
def mock_message_context() -> MagicMock:
    """Create a mock MessageContext."""
    context = MagicMock(spec=MessageContext)
    context.topic_id = DefaultTopicId(type="test-topic")
    context.sender = "mock_sender/instance"
    context.message_id = "mock-message-id"
    context.cancellation_token = None
    context.is_rpc = False  # Typically False for standard message handling
    return context


@pytest.fixture
def agent_input_message() -> AgentInput:
    """Provides a sample AgentInput message."""
    return AgentInput(
        inputs={"prompt": "Evaluate this content based on criteria.", "content": "Some content to evaluate"},
        parameters={},  # Add task-specific parameters if needed
    )


# --- Test Class ---


@pytest.mark.anyio
class TestAutogenAdapterIntegration:
    async def test_adapter_initialization(self, mock_agent_config):
        """Test adapter initializes the underlying agent correctly."""
        # Mock LLM client lookup needed by Judge (which inherits from LLMAgent)
        with patch("buttermilk.bm.bm.llms.get_autogen_chat_client", return_value=MagicMock()):
            adapter = AutogenAgentAdapter(
                topic_type="test_topic",  # Provide topic_type
                agent_cls=Judge,  # Use correct param name
                agent_cfg=mock_agent_config,
            )
            # Initialization happens within adapter's __init__ now
            await adapter.agent.initialize()  # Ensure any async init in agent runs

        assert isinstance(adapter.agent, Judge)
        assert adapter.agent.name == mock_agent_config.name
        assert adapter.agent.role == mock_agent_config.role
        assert adapter.topic_id.type == "test_topic"

    async def test_buttermilk_handler_invocation_via_adapter(self, mock_agent_config, agent_input_message, mock_message_context):
        """Test that adapter's message handler calls the correct agent method."""
        # Mock the specific method decorated with @buttermilk_handler in Judge
        # The documented Judge uses evaluate_content (renamed from handle_agent_input)
        mock_judge = MagicMock(spec=Judge)
        mock_judge.evaluate_content = AsyncMock(name="evaluate_content")
        mock_response = AgentTrace(
            agent_info=mock_judge._cfg,
            outputs=Reasons(prediction=True, reasons=[], confidence="low", conclusion="mocked"),
        )
        mock_judge.evaluate_content.return_value = mock_response
        # Set necessary attributes
        mock_judge.id = "mock_judge_id"
        mock_judge.role = "judge"
        mock_judge.description = "Mock Judge"
        mock_judge.initialize = AsyncMock()  # Needs initialize method

        # Create adapter with the *mock* agent instance
        adapter = AutogenAgentAdapter(topic_type="test_topic", agent=mock_judge)
        adapter.publish_message = AsyncMock()  # Mock publishing

        # Directly call the adapter's handler for AgentInput
        # The adapter's internal routing logic finds the @buttermilk_handler method
        result = await adapter.handle_invocation(agent_input_message, mock_message_context)

        # Assert that the decorated method on the mock agent was called
        mock_judge.evaluate_content.assert_called_once()
        # Check args passed to the agent's method (should not include context directly)
        call_args, call_kwargs = mock_judge.evaluate_content.call_args
        assert call_kwargs.get("message") == agent_input_message
        # Verify the handler returned the result from the agent method (it might not, depending on adapter logic)
        # The current adapter's handle_invocation calls __call__ which calls _process,
        # Let's adjust the test to mock __call__ as that's what the handler invokes.

        # --- Re-run with __call__ mocked ---
        mock_judge.reset_mock()  # Reset mocks
        mock_judge.__call__ = AsyncMock(name="__call__", return_value=mock_response)  # Mock __call__ instead
        adapter = AutogenAgentAdapter(topic_type="test_topic", agent=mock_judge)
        adapter.publish_message = AsyncMock()

        result = await adapter.handle_invocation(agent_input_message, mock_message_context)

        # Assert __call__ was invoked by the handler
        mock_judge.__call__.assert_called_once()
        call_args, call_kwargs = mock_judge.__call__.call_args
        assert call_kwargs.get("message") == agent_input_message  # Check message passed to __call__

        # Assert the result returned by the handler is the result from __call__
        assert result == mock_response
        # Assert task processing events were published (checked in adapter tests, less critical here)
        assert adapter.publish_message.call_count >= 2  # Start and Complete

    async def test_autogen_registration_and_get(self, mock_agent_config):
        """Test agent registration and retrieval using the adapter."""
        runtime = SingleThreadedAgentRuntime()
        agent_type_id = "test-judge-reg"  # Use a unique ID for registration type

        # Define the factory function
        def adapter_factory():
            # Mock LLM client lookup inside the factory
            with patch("buttermilk.bm.bm.llms.get_autogen_chat_client", return_value=MagicMock()):
                return AutogenAgentAdapter(
                    topic_type="reg_topic",  # Provide topic_type
                    agent_cls=Judge,  # Use agent_cls
                    agent_cfg=mock_agent_config,
                )

        # Register using the static method
        registered_agent_type = await AutogenAgentAdapter.register(
            runtime=runtime,
            id=agent_type_id,  # The ID to register this type under
            factory=adapter_factory,
        )

        # Verify registration details
        assert registered_agent_type is not None
        assert registered_agent_type.type == agent_type_id

        # Get an instance of the registered agent from the runtime
        # This implicitly calls the factory
        agent_instance_id = await runtime.get(registered_agent_type)
        assert agent_instance_id is not None
        assert isinstance(agent_instance_id, str)  # Runtime stores instance IDs as strings

        # Optional: Check if runtime has the instance (internal detail)
        # assert agent_instance_id in runtime._agent_threads # Accessing private member

        # To further test, you could potentially send a message via the runtime
        # to the retrieved agent_instance_id and assert the response, but that
        # becomes a more complex runtime integration test. This test primarily
        # verifies the registration mechanism itself works.
