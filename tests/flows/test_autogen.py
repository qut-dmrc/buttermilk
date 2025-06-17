from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.constants import CONDUCTOR
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
)
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.libs.autogen import AutogenAgentAdapter


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=Agent)
    agent.agent_id = "test_agent"
    agent.description = "Test agent description"
    agent.invoke = AsyncMock(
        return_value=AgentTrace(
            agent_id="test",
            agent_info=AgentConfig(role="test"),
            inputs=AgentInput(),
        ),
    )
    agent.initialize = AsyncMock()
    agent._listen = AsyncMock(return_value=None)
    return agent


@pytest.fixture
def agent_adapter(mock_agent):
    """Create an AutogenAgentAdapter with a mock agent."""
    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=mock_agent,
        )
        return adapter


@pytest.mark.anyio
async def test_agent_adapter_init_with_agent():
    """Test AutogenAgentAdapter initialization with an agent."""
    agent = MagicMock(spec=Agent)
    agent.description = "Test description"
    agent.initialize = AsyncMock()

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(topic_type="test_topic", agent=agent)

    assert adapter.agent == agent
    assert adapter.topic_id.type == "test_topic"


@pytest.mark.anyio
async def test_agent_adapter_init_with_config():
    """Test AutogenAgentAdapter initialization with config."""
    mock_agent_cls = MagicMock()
    mock_agent_cls.__name__ = "MockAgent"
    mock_agent_instance = MagicMock(spec=Agent)
    mock_agent_instance.agent_id = "test_agent"
    mock_agent_instance.description = "Test description"
    mock_agent_instance.initialize = AsyncMock()
    mock_agent_cls.return_value = mock_agent_instance

    agent_cfg = AgentConfig(name="test_id")

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent_cls=mock_agent_cls,
            agent_cfg=agent_cfg,
        )

    mock_agent_cls.assert_called_once()
    assert adapter.agent == mock_agent_instance


@pytest.mark.anyio
async def test_agent_adapter_process_request(agent_adapter):
    """Test handle_invocation method."""
    message = AgentInput(inputs={"content": "test input"})
    ctx = MagicMock()
    ctx.cancellation_token = None
    ctx.sender = "test_sender"
    ctx.topic_id = "test_topic"

    # Non-conductor agent (publishes message)
    with patch.object(agent_adapter, "publish_message", new_callable=AsyncMock):
        result = await agent_adapter.handle_invocation(message, ctx)

    agent_adapter.agent.invoke.assert_called_once()
    assert result == agent_adapter.agent.invoke.return_value


@pytest.mark.anyio
async def test_agent_adapter_process_request_conductor():
    """Test handle_invocation method for conductor agent."""
    agent = MagicMock(spec=Agent)
    agent.agent_id = f"{CONDUCTOR}-test"
    agent.description = "Test conductor"
    agent.invoke = AsyncMock(
        return_value=AgentTrace(
            agent_id="test",
            agent_info=AgentConfig(role="test"),
            inputs=AgentInput(),
        ),
    )
    agent.initialize = AsyncMock()

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=agent,
        )

    message = AgentInput(inputs={"content": "test input"})
    ctx = MagicMock()
    ctx.cancellation_token = None
    ctx.sender = "test_sender"
    ctx.topic_id = "test_topic"

    # Test conductor behavior
    with patch.object(adapter, "publish_message", new_callable=AsyncMock):
        result = await adapter.handle_invocation(message, ctx)

    adapter.agent.invoke.assert_called_once()
    assert result == adapter.agent.invoke.return_value


@pytest.mark.anyio
async def test_agent_adapter_handle_output(agent_adapter):
    """Test handle_groupchat_message method."""
    # Create a message context with a different sender
    ctx = MagicMock()
    ctx.sender = "different_agent"
    ctx.topic_id = "test_topic"
    ctx.cancellation_token = None

    message = AgentTrace(
        agent_id="test",
        agent_info=AgentConfig(role="test"),
        inputs=AgentInput(),
    )

    await agent_adapter.handle_groupchat_message(message, ctx)

    # Should call _listen since this is a group chat message
    agent_adapter.agent._listen.assert_called_once()


@pytest.mark.anyio
async def test_agent_adapter_handle_output_from_self(agent_adapter):
    """Test handle_groupchat_message method with message from self."""
    # Create a message context
    ctx = MagicMock()
    ctx.sender = agent_adapter.id.type
    ctx.topic_id = "test_topic"
    ctx.cancellation_token = None

    message = AgentTrace(
        agent_id="test",
        agent_info=AgentConfig(role="test"),
        inputs=AgentInput(),
    )

    await agent_adapter.handle_groupchat_message(message, ctx)

    # Should still call _listen for group chat messages regardless of sender
    agent_adapter.agent._listen.assert_called_once()


@pytest.mark.anyio
async def test_agent_adapter_handle_input_ui_agent():
    """Test handle_control_message method with UI agent."""
    ui_agent = MagicMock(spec=UIAgent)
    ui_agent.agent_id = "ui_test"
    ui_agent.description = "Test UI agent"
    ui_agent.initialize = AsyncMock()
    ui_agent._handle_events = AsyncMock(return_value=None)

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=ui_agent,
        )

    from buttermilk._core.contract import HeartBeat
    message = HeartBeat()
    ctx = MagicMock()
    ctx.cancellation_token = None
    ctx.sender = "test_sender"
    ctx.topic_id = "test_topic"

    result = await adapter.handle_control_message(message, ctx)
    ui_agent._handle_events.assert_called_once()
    assert result is None


@pytest.mark.anyio
async def test_agent_adapter_handle_input_normal_agent():
    """Test handle_control_message method with normal agent."""
    agent = MagicMock(spec=Agent)
    agent.agent_id = "normal_test"
    agent.description = "Test agent"
    agent.initialize = AsyncMock()
    agent._handle_events = AsyncMock(return_value=None)

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=agent,
        )

    from buttermilk._core.contract import HeartBeat
    message = HeartBeat()
    ctx = MagicMock()
    ctx.cancellation_token = None
    ctx.sender = "test_sender"
    ctx.topic_id = "test_topic"

    result = await adapter.handle_control_message(message, ctx)
    agent._handle_events.assert_called_once()
    assert result is None
