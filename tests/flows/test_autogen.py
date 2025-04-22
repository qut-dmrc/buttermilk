from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.agent import Agent, AgentConfig
from buttermilk._core.contract import (
    CONDUCTOR,
    AgentInput,
    AgentOutput,
    UserInstructions,
)
from buttermilk.libs.autogen import AutogenAgentAdapter


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=Agent)
    agent.id = "test_agent"
    agent.description = "Test agent description"
    agent.__call__ = AsyncMock(
        return_value=AgentOutput(
            agent_id="test",
            content="Test output",
        )
    )
    agent.initialize = AsyncMock()
    agent.receive_output = AsyncMock(return_value=None)
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
    mock_agent_instance = MagicMock(spec=Agent)
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
    """Test process_request method."""
    message = AgentInput(content="test input")

    # Non-conductor agent (publishes message)
    with patch.object(agent_adapter, "publish_message", new_callable=AsyncMock) as mock_publish:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await agent_adapter._process_request(message)

    agent_adapter.agent.__call__.assert_called_once_with(message)
    mock_publish.assert_called_once()
    assert result == agent_adapter.agent.__call__.return_value


@pytest.mark.anyio
async def test_agent_adapter_process_request_conductor():
    """Test process_request method for conductor agent."""
    agent = MagicMock(spec=Agent)
    agent.id = f"{CONDUCTOR}-test"
    agent.description = "Test conductor"
    agent.__call__ = AsyncMock(
        return_value=AgentOutput(
            agent_id="test",
            content="Test output",
        )
    )
    agent.initialize = AsyncMock()

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=agent,
        )

    # Set ID to be a conductor
    adapter.id.type = f"{CONDUCTOR}-test"

    message = AgentInput(content="test input")

    # Conductor agent (doesn't publish message)
    with patch.object(adapter, "publish_message", new_callable=AsyncMock) as mock_publish:
        result = await adapter._process_request(message)

    adapter.agent.__call__.assert_called_once_with(message)
    mock_publish.assert_not_called()
    assert result == adapter.agent.__call__.return_value


@pytest.mark.anyio
async def test_agent_adapter_handle_output(agent_adapter):
    """Test handle_output method."""
    # Create a message context with a different sender
    ctx = MagicMock()
    ctx.sender.type = "different_agent"

    message = AgentOutput(agent_id="test", content="test output")

    await agent_adapter.handle_output(message, ctx)

    # Should call receive_output since message is from someone else
    agent_adapter.agent.receive_output.assert_called_once_with(message)


@pytest.mark.anyio
async def test_agent_adapter_handle_output_from_self(agent_adapter):
    """Test handle_output method with message from self."""
    # Create a message context with self as sender
    ctx = MagicMock()
    ctx.sender.type = agent_adapter.id

    message = AgentOutput(agent_id="test", content="test output")

    await agent_adapter.handle_output(message, ctx)

    # Should not call receive_output since message is from self
    agent_adapter.agent.receive_output.assert_not_called()


@pytest.mark.anyio
async def test_agent_adapter_handle_input_ui_agent():
    """Test handle_input method with UI agent."""
    ui_agent = MagicMock(spec=UIAgent)
    ui_agent.description = "Test UI agent"
    ui_agent.initialize = AsyncMock()

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=ui_agent,
        )

    callback = adapter.handle_input()
    assert callback is not None

    # Test the callback
    with patch.object(adapter, "publish_message", new_callable=AsyncMock) as mock_publish:
        message = UserInstructions(content="test input")
        await callback(message)

        mock_publish.assert_called_once_with(message, topic_id=adapter.topic_id)


@pytest.mark.anyio
async def test_agent_adapter_handle_input_normal_agent():
    """Test handle_input method with normal agent."""
    agent = MagicMock(spec=Agent)
    agent.description = "Test agent"
    agent.initialize = AsyncMock()

    with patch("asyncio.create_task"):
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent=agent,
        )

    callback = adapter.handle_input()
    assert callback is None
