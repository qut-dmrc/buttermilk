"""Tests for the AutogenAgentAdapter which bridges Buttermilk agents and Autogen runtime."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Autogen imports
from autogen_core import MessageContext

# Buttermilk core types
from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.constants import END, WAIT
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    ConductorRequest,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
)
from buttermilk.agents.flowcontrol.explorer import ExplorerHost
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.flowcontrol.llmhost import LLMHostAgent

# The class under test
from buttermilk.libs.autogen import AutogenAgentAdapter

# --- Fixtures ---


@pytest.fixture
def conductor_request() -> ConductorRequest:
    """Create a sample ConductorRequest object for testing."""
    return ConductorRequest(
        inputs={
            "task": "test task",
            "prompt": "Test prompt for conductor",
        },
        participants={
            "AGENT1": "First test agent",
            "AGENT2": "Second test agent",
        },
    )


@pytest.fixture
def agent_input() -> AgentInput:
    """Create a sample AgentInput object for testing invocation."""
    return AgentInput(
        inputs={"data": "sample input", "prompt": "Test prompt for invocation"},
    )


@pytest.fixture
def mock_message_context() -> MagicMock:
    """Create a mock MessageContext for testing message handlers."""
    context = MagicMock(spec=MessageContext)
    context.cancellation_token = None
    context.sender = "test_sender/instance_id"
    context.topic_id = "test_topic_id"
    return context


# --- Test Class ---


@pytest.mark.anyio
class TestAutogenAgentAdapter:
    """Tests for AutogenAgentAdapter."""

    # --- Initialization Tests ---

    async def test_adapter_initialization_with_HostAgent(self):
        """Test adapter initialization with a HostAgent agent."""
        HostAgent_config = AgentConfig(role="HostAgent", name="Test HostAgent", description="Test HostAgent")
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent_cls=HostAgent,
            agent_cfg=HostAgent_config,
        )
        await adapter.agent.initialize()  # Initialize agent after adapter creation

        assert isinstance(adapter.agent, HostAgent)
        assert adapter.agent.role == "HostAgent"
        assert adapter.topic_id.type == "test_topic"
        assert adapter.is_manager, "HostAgent should be identified as a manager"

    async def test_adapter_initialization_with_host(self):
        """Test adapter initialization with an LLMHostAgent."""
        host_config = AgentConfig(role="HOST", name="Test Host", description="Test Host", parameters={"model": "mock_model"})
        with patch("buttermilk.buttermilk.llms.get_autogen_chat_client", return_value=MagicMock()):
            adapter = AutogenAgentAdapter(
                topic_type="test_topic",
                agent_cls=LLMHostAgent,
                agent_cfg=host_config,
            )
            await adapter.agent.initialize(callback_to_groupchat=AsyncMock())  # Initialize agent

        assert isinstance(adapter.agent, LLMHostAgent)
        assert adapter.agent.role == "host"
        assert adapter.topic_id.type == "test_topic"
        assert adapter.is_manager, "LLMHostAgent should be identified as a manager"

    async def test_adapter_initialization_with_explorer(self):
        """Test adapter initialization with an ExplorerHost agent."""
        explorer_config = AgentConfig(role="EXPLORER", name="Test Explorer", description="Test Explorer", parameters={"model": "mock_model"})
        with patch("buttermilk.buttermilk.llms.get_autogen_chat_client", return_value=MagicMock()):
            adapter = AutogenAgentAdapter(
                topic_type="test_topic",
                agent_cls=ExplorerHost,
                agent_cfg=explorer_config,
            )
            await adapter.agent.initialize()  # Initialize agent

        assert isinstance(adapter.agent, ExplorerHost)
        assert adapter.agent.role == "explorer"
        assert adapter.topic_id.type == "test_topic"
        assert adapter.is_manager, "ExplorerHost should be identified as a manager"

    # --- Handler Tests ---

    async def test_handle_conductor_request_routes_to_agent_call(self, conductor_request, mock_message_context):
        """Test adapter routes ConductorRequest to the wrapped agent's __call__."""
        mock_agent = MagicMock(spec=Agent)
        mock_agent.__call__ = AsyncMock(name="__call__")
        mock_response = AgentTrace(
            agent_info="mock_agent_id", role="test", outputs=StepRequest(role="AGENT1", prompt="Next step", description="Desc"),
        )
        mock_agent.__call__.return_value = mock_response
        mock_agent.id = "mock_agent_id"
        mock_agent.role = "test"
        mock_agent.description = "mock"
        mock_agent.initialize = AsyncMock()

        adapter = AutogenAgentAdapter(topic_type="test_topic", agent=mock_agent)
        adapter.publish_message = AsyncMock()

        result = await adapter.handle_conductor_request(conductor_request, mock_message_context)

        mock_agent.__call__.assert_called_once()
        call_kwargs = mock_agent.__call__.call_args.kwargs
        assert call_kwargs.get("message") == conductor_request
        assert call_kwargs.get("cancellation_token") == mock_message_context.cancellation_token
        assert callable(call_kwargs.get("public_callback"))
        assert callable(call_kwargs.get("message_callback"))
        assert call_kwargs.get("source") == "test_sender"
        assert result == mock_response
        adapter.publish_message.assert_not_called()

    async def test_handle_invocation_sends_task_events(self, agent_input, mock_message_context):
        """Test adapter sends TaskProcessingStarted/Complete during handle_invocation."""
        mock_agent = MagicMock(spec=Agent)
        mock_agent.__call__ = AsyncMock(name="__call__")
        mock_response = AgentTrace(agent_info="mock_agent_id", role="test_role", outputs={"result": "done"})
        mock_agent.__call__.return_value = mock_response
        mock_agent.id = "mock_agent_id"
        mock_agent.role = "test_role"
        mock_agent.description = "mock"
        mock_agent.initialize = AsyncMock()

        adapter = AutogenAgentAdapter(topic_type="test_topic", agent=mock_agent)
        # Mock the 'type' attribute which might be set during registration
        adapter.type = mock_agent.role
        adapter.publish_message = AsyncMock()

        result = await adapter.handle_invocation(agent_input, mock_message_context)

        mock_agent.__call__.assert_called_once()
        call_kwargs = mock_agent.__call__.call_args.kwargs
        assert call_kwargs.get("message") == agent_input

        assert adapter.publish_message.call_count >= 2
        start_call_args = adapter.publish_message.call_args_list[0].kwargs
        assert isinstance(start_call_args.get("message"), TaskProcessingStarted)
        assert start_call_args["message"].agent_id == mock_agent.id
        assert start_call_args["message"].role == mock_agent.role  # Check role in event
        assert start_call_args.get("topic_id") == adapter.topic_id

        complete_call_args = adapter.publish_message.call_args_list[-1].kwargs
        assert isinstance(complete_call_args.get("message"), TaskProcessingComplete)
        assert complete_call_args["message"].agent_id == mock_agent.id
        assert complete_call_args["message"].role == mock_agent.role  # Check role in event
        assert complete_call_args["message"].is_error is False
        assert complete_call_args.get("topic_id") == adapter.topic_id

        assert result == mock_response

    # --- Integration Test ---

    async def test_integration_adapter_with_HostAgent_conductor_request(self, conductor_request, mock_message_context):
        """Integration test: verify adapter routes ConductorRequest to a real HostAgent."""
        HostAgent = HostAgent(role="HostAgent", name="Real HostAgent", description="Integration Test HostAgent")
        await HostAgent.initialize()  # Initialize the real agent

        adapter = AutogenAgentAdapter(topic_type="integration_topic", agent=HostAgent)
        adapter.publish_message = AsyncMock()  # Mock publish to prevent side effects

        response = await adapter.handle_conductor_request(conductor_request, mock_message_context)

        assert isinstance(response, AgentTrace)
        assert not response.is_error, f"HostAgent returned error: {response.error}"
        assert isinstance(response.outputs, StepRequest)
        participants = conductor_request.inputs["participants"]
        assert response.outputs.role in list(participants.keys()) + [END, WAIT]
        assert response.outputs.role != HostAgent.role.upper()
        adapter.publish_message.assert_not_called()
