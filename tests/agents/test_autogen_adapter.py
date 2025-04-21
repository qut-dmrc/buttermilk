import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from autogen_core import MessageContext

from buttermilk._core.agent import Agent, AgentConfig
from buttermilk._core.contract import (
    ConductorRequest,
    ConductorResponse,
    StepRequest,
    AgentOutput,
    TaskProcessingComplete,
    TaskProcessingStarted,
)
from buttermilk.agents.flowcontrol.sequencer import Sequencer
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.agents.flowcontrol.explorer import ExplorerHost
from buttermilk.libs.autogen import AutogenAgentAdapter


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


@pytest.fixture
def mock_message_context():
    """Create a mock MessageContext for testing."""
    context = MagicMock(spec=MessageContext)
    context.cancellation_token = None
    context.sender = "test_sender/123"
    context.topic_id = "test_topic"
    return context


class TestAutogenAgentAdapter:
    """Tests for AutogenAgentAdapter with focus on ConductorRequest handling."""

    @pytest.mark.anyio
    async def test_adapter_initialization_with_sequencer(self):
        """
        Test that adapter can be initialized with a Sequencer agent.
        """
        sequencer_config = AgentConfig(role="SEQUENCER", name="Test Sequencer", description="Test Sequencer")
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent_cls=Sequencer,
            agent_cfg=sequencer_config,
        )

        # Verify the adapter was initialized with a Sequencer agent
        assert isinstance(adapter.agent, Sequencer)
        assert adapter.agent.role == "SEQUENCER"
        assert adapter.topic_id.type == "test_topic"

    @pytest.mark.anyio
    async def test_adapter_initialization_with_host(self):
        """
        Test that adapter can be initialized with a Host agent.
        """
        host_config = AgentConfig(role="HOST", name="Test Host", description="Test Host")
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent_cls=LLMHostAgent,
            agent_cfg=host_config,
        )

        # Verify the adapter was initialized with a Host agent
        assert isinstance(adapter.agent, LLMHostAgent)
        assert adapter.agent.role == "HOST"
        assert adapter.topic_id.type == "test_topic"
        assert adapter.is_manager  # Host agents should be recognized as managers

    @pytest.mark.anyio
    async def test_adapter_initialization_with_explorer(self):
        """
        Test that adapter can be initialized with an Explorer agent.
        """
        explorer_config = AgentConfig(role="EXPLORER", name="Test Explorer", description="Test Explorer")
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent_cls=ExplorerHost,
            agent_cfg=explorer_config,
        )

        # Verify the adapter was initialized with an Explorer agent
        assert isinstance(adapter.agent, ExplorerHost)
        assert adapter.agent.role == "EXPLORER"
        assert adapter.topic_id.type == "test_topic"
        assert adapter.is_manager  # Explorer agents should be recognized as managers

    @pytest.mark.anyio
    async def test_handle_conductor_request_with_sequencer(self, conductor_request, mock_message_context):
        """
        Test that adapter correctly routes ConductorRequest to the Sequencer agent.
        """
        # Set up a Sequencer with a mocked call method
        sequencer = Sequencer(role="SEQUENCER", description="Test Sequencer", name="test sequencer")
        sequencer.__call__ = AsyncMock()

        # Adding __call__ parameter definitions
        async def mock_call(message=None, cancellation_token=None, public_callback=None, message_callback=None, source=""):
            return mock_response

        sequencer.__call__.side_effect = mock_call
        mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
        sequencer.__call__.return_value = mock_response

        # Create adapter with mocked agent
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=sequencer,
        )
        adapter.publish_message = AsyncMock()  # Mock publish_message

        # Call handle_conductor_request
        result = await adapter.handle_conductor_request(conductor_request, mock_message_context)

        # Verify the agent's call method was called with the request
        sequencer.__call__.assert_called_once()
        call_args = sequencer.__call__.call_args[1]
        assert call_args["message"] == conductor_request
        assert call_args["cancellation_token"] == mock_message_context.cancellation_token

        # Verify the result is the response from the agent
        assert result == mock_response

    @pytest.mark.anyio
    async def test_handle_conductor_request_with_host(self, conductor_request, mock_message_context):
        """
        Test that adapter correctly routes ConductorRequest to the Host agent.
        """
        # Set up a Host with a mocked call method
        host = LLMHostAgent(role="HOST", description="Test Host", name="test")
        host.__call__ = AsyncMock()

        # Adding __call__ parameter definitions
        async def mock_call(message=None, cancellation_token=None, public_callback=None, message_callback=None, source=""):
            return mock_response

        host.__call__.side_effect = mock_call
        mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
        host.__call__.return_value = mock_response

        # Create adapter with mocked agent
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=host,
        )
        adapter.publish_message = AsyncMock()  # Mock publish_message

        # Call handle_conductor_request
        result = await adapter.handle_conductor_request(conductor_request, mock_message_context)

        # Verify the agent's call method was called with the request
        host.__call__.assert_called_once()
        call_args = host.__call__.call_args[1]
        assert call_args["message"] == conductor_request
        assert call_args["cancellation_token"] == mock_message_context.cancellation_token

        # Verify the result is the response from the agent
        assert result == mock_response

    @pytest.mark.anyio
    async def test_handle_conductor_request_with_explorer(self, conductor_request, mock_message_context):
        """
        Test that adapter correctly routes ConductorRequest to the Explorer agent.
        """
        # Set up an Explorer with a mocked call method
        explorer = ExplorerHost(role="EXPLORER", description="Test Explorer", name="test")
        explorer.__call__ = AsyncMock()

        # Adding __call__ parameter definitions
        async def mock_call(message=None, cancellation_token=None, public_callback=None, message_callback=None, source=""):
            return mock_response

        explorer.__call__.side_effect = mock_call
        mock_response = AgentOutput(outputs=StepRequest(role="AGENT1", prompt="", description="Test step"))
        explorer.__call__.return_value = mock_response

        # Create adapter with mocked agent
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=explorer,
        )
        adapter.publish_message = AsyncMock()  # Mock publish_message

        # Call handle_conductor_request
        result = await adapter.handle_conductor_request(conductor_request, mock_message_context)

        # Verify the agent's call method was called with the request
        explorer.__call__.assert_called_once()
        call_args = explorer.__call__.call_args[1]
        assert call_args["message"] == conductor_request
        assert call_args["cancellation_token"] == mock_message_context.cancellation_token

        # Verify the result is the response from the agent
        assert result == mock_response

    @pytest.mark.anyio
    async def test_handle_invocation_sends_task_events(self, mock_message_context):
        """
        Test that adapter sends TaskProcessingStarted and TaskProcessingComplete events
        when handling an invocation.
        """
        # Set up a Sequencer with a mocked call method
        sequencer = Sequencer(role="SEQUENCER", description="Test Sequencer", name="test")
        sequencer.__call__ = AsyncMock()

        # Adding __call__ parameter definitions
        async def mock_call(message=None, cancellation_token=None, public_callback=None, message_callback=None, source=""):
            return mock_response

        sequencer.__call__.side_effect = mock_call
        mock_response = AgentOutput(content="Test response")
        sequencer.__call__.return_value = mock_response

        # Create adapter with mocked agent
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=sequencer,
        )
        adapter.publish_message = AsyncMock()  # Mock publish_message

        # Call handle_invocation
        agent_input = MagicMock()
        result = await adapter.handle_invocation(agent_input, mock_message_context)

        # Verify TaskProcessingStarted was published
        assert adapter.publish_message.call_count >= 2  # At least called for start and complete events

        # Check the first call is TaskProcessingStarted
        first_call_args = adapter.publish_message.call_args_list[0][0]
        assert isinstance(first_call_args[0], TaskProcessingStarted)
        assert first_call_args[0].agent_id == sequencer.id
        assert first_call_args[0].role == adapter.type

        # Check TaskProcessingComplete was published
        # The last call should be TaskProcessingComplete
        last_call_args = adapter.publish_message.call_args_list[-1][0]
        assert isinstance(last_call_args[0], TaskProcessingComplete)
        assert last_call_args[0].agent_id == sequencer.id
        assert last_call_args[0].role == adapter.type
        assert last_call_args[0].more_tasks_remain is False
        assert last_call_args[0].is_error is False

        # Verify the result is the response from the agent
        assert result == mock_response


@pytest.mark.anyio
async def test_integration_autogen_adapter_with_conductor_request():
    """
    Integration test: verify AutogenAgentAdapter properly integrates agents
    with the Autogen runtime for handling ConductorRequests.
    """
    # Create a real Sequencer agent
    sequencer = Sequencer(role="SEQUENCER", description="Test Sequencer", name="test")
    await sequencer.initialize()

    # Create an adapter with the Sequencer
    adapter = AutogenAgentAdapter(
        topic_type="test_topic",
        agent=sequencer,
    )

    # Mock the publish_message method to avoid actual publishing
    adapter.publish_message = AsyncMock()

    # Create a conductor request
    conductor_request = ConductorRequest(
        inputs={
            "participants": {
                "AGENT1": {"config": "some_config"},
                "AGENT2": {"config": "some_config"},
            },
            "task": "integration test",
        },
        prompt="Integration test",
    )

    # Create a mock context for the message handler
    context = MagicMock(spec=MessageContext)
    context.cancellation_token = None
    context.sender = "test_sender/123"
    context.topic_id = "test_topic"

    # Call the handle_conductor_request method
    response = await adapter.handle_conductor_request(conductor_request, context)

    # Verify the response is an AgentOutput with StepRequest
    assert isinstance(response, AgentOutput)
    assert isinstance(response.outputs, StepRequest)

    # Verify the step is either AGENT1, AGENT2, or END
    assert response.outputs.role in ["AGENT1", "AGENT2", "END"]
