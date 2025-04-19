from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.contract import AgentInput, AgentOutput, ManagerRequest
from buttermilk.agents.ui.slackthreadchat import (
    SlackUIAgent,
)
from buttermilk.libs.slack import SlackContext


@pytest.fixture
def slack_context():
    """Create a mock Slack context for testing."""
    return SlackContext(
        channel_id="test_channel",
        thread_ts="test_thread",
        user_id="test_user",
        event_ts="test_event",
        say=AsyncMock(),
    )


@pytest.fixture
def slack_app():
    """Create a mock Slack app for testing."""
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_update = AsyncMock()
    app.client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {"user": "U123", "text": "Hello"},
                {"user": "bot", "text": "Hi there!"},
            ],
        },
    )
    app.message = MagicMock(return_value=lambda f: f)
    app.action = MagicMock(return_value=lambda f: f)
    return app


@pytest.fixture
def slack_ui_agent(slack_app, slack_context):
    """Create a SlackUIAgent with mocked dependencies."""
    agent = SlackUIAgent(
        id="test_agent",
        description="Test agent for Slack",
    )
    agent.app = slack_app
    agent.context = slack_context
    return agent


@pytest.mark.anyio
async def test_slack_ui_agent_initialization(slack_ui_agent):
    """Test that the SlackUIAgent initializes correctly."""
    input_callback = AsyncMock()

    with patch(
        "buttermilk.agents.ui.slackthreadchat.register_chat_thread_handler",
    ) as mock_register:
        await slack_ui_agent.initialize(input_callback=input_callback)

    mock_register.assert_called_once_with(
        slack_ui_agent.context.thread_ts,
        slack_ui_agent,
    )
    assert slack_ui_agent._input_callback == input_callback


@pytest.mark.anyio
async def test_slack_ui_agent_send_to_thread(slack_ui_agent):
    """Test sending messages to a Slack thread."""
    with patch(
        "buttermilk.agents.ui.slackthreadchat.post_message_with_retry",
        new_callable=AsyncMock,
    ) as mock_post:
        await slack_ui_agent.send_to_thread(text="Test message")

    mock_post.assert_called_once_with(
        app=slack_ui_agent.app,
        context=slack_ui_agent.context,
        text="Test message",
        blocks=None,
    )


@pytest.mark.anyio
async def test_slack_ui_agent_receive_output_agent_output(slack_ui_agent):
    """Test handling of AgentOutput messages."""
    message = AgentOutput(
        content="Test output",
        outputs={"key": "value"},
    )

    with patch(
        "buttermilk.agents.ui.slackthreadchat.format_slack_message",
    ) as mock_format:
        mock_format.return_value = {"text": "Formatted text", "blocks": []}
        with patch.object(
            slack_ui_agent,
            "send_to_thread",
            new_callable=AsyncMock,
        ) as mock_send:
            await slack_ui_agent.receive_output(message)

    mock_format.assert_called_once()
    mock_send.assert_called_once_with(text="Formatted text", blocks=[])


@pytest.mark.anyio
async def test_slack_ui_agent_receive_output_format_error(slack_ui_agent):
    """Test handling of formatting errors."""
    message = AgentOutput(
        content="Test output",
        outputs={"key": "value"},
    )

    with patch(
        "buttermilk.agents.ui.slackthreadchat.format_slack_message",
    ) as mock_format:
        mock_format.side_effect = Exception("Formatting error")
        with patch.object(
            slack_ui_agent,
            "send_to_thread",
            new_callable=AsyncMock,
        ) as mock_send:
            await slack_ui_agent.receive_output(message)

    mock_send.assert_called_once_with(text="Test output")


@pytest.mark.anyio
async def test_request_user_input_boolean(slack_ui_agent):
    """Test requesting user input with boolean options."""
    message = ManagerRequest(
        role="tester",
        content="Do you want to continue?",
        options=True,
    )

    with patch("buttermilk.agents.ui.slackthreadchat.confirm_bool") as mock_confirm:
        mock_confirm.return_value = {"text": "Confirm?", "blocks": []}
        with patch.object(
            slack_ui_agent,
            "send_to_thread",
            new_callable=AsyncMock,
        ) as mock_send:
            mock_send.return_value = MagicMock(data={"ts": "message_ts"})
            await slack_ui_agent._request_user_input(message)

    mock_confirm.assert_called_once()
    mock_send.assert_called_once_with(text="Confirm?", blocks=[])
    assert slack_ui_agent._current_input_message is not None


@pytest.mark.anyio
async def test_request_user_input_options_list(slack_ui_agent):
    """Test requesting user input with a list of options."""
    message = ManagerRequest(
        content="Choose an option:",
        options=["Option 1", "Option 2", "Option 3"],
    )

    with patch("buttermilk.agents.ui.slackthreadchat.confirm_options") as mock_confirm:
        mock_confirm.return_value = {"text": "Choose:", "blocks": []}
        with patch.object(
            slack_ui_agent,
            "send_to_thread",
            new_callable=AsyncMock,
        ) as mock_send:
            mock_send.return_value = MagicMock(data={"ts": "message_ts"})
            await slack_ui_agent._request_user_input(message)

    mock_confirm.assert_called_once()
    mock_send.assert_called_once_with(text="Choose:", blocks=[])
    assert slack_ui_agent._current_input_message is not None


@pytest.mark.anyio
async def test_update_existing_input_message(slack_ui_agent):
    """Test updating an existing input message instead of creating a new one."""
    # First set an existing input message
    slack_ui_agent._current_input_message = MagicMock(data={"ts": "existing_ts"})

    message = ManagerRequest(
        content="New question?",
        options=True,
    )

    with patch("buttermilk.agents.ui.slackthreadchat.confirm_bool") as mock_confirm:
        mock_confirm.return_value = {"text": "Confirm?", "blocks": []}
        await slack_ui_agent._request_user_input(message)

    slack_ui_agent.app.client.chat_update.assert_called_once_with(
        channel=slack_ui_agent.context.channel_id,
        ts="existing_ts",
        text="Confirm?",
        blocks=[],
    )


@pytest.mark.anyio
async def test_process_method(slack_ui_agent):
    """Test the _process method that handles agent input."""
    input_data = AgentInput(
        content="Test input",
    )

    with patch.object(
        slack_ui_agent,
        "_request_user_input",
        new_callable=AsyncMock,
    ) as mock_request:
        result = await slack_ui_agent._process(input_data)

    mock_request.assert_called_once_with(input_data)
    assert result is None


def test_register_chat_thread_handler():
    """Test the registration of Slack thread handlers."""
    thread_ts = "test_thread"
    agent = MagicMock()
    agent.app = MagicMock()
    agent.app.message = MagicMock(return_value=lambda f: f)
    agent.app.action = MagicMock(return_value=lambda f: f)
    agent.context = MagicMock(thread_ts=thread_ts)

    register_chat_thread_handler(thread_ts, agent)

    # Verify handlers were registered
    assert agent.app.message.call_count == 1
    assert agent.app.action.call_count == 2  # confirm and cancel actions


@pytest.mark.anyio
async def test_handle_confirm_action():
    """Test the confirm action handler function."""
    # This is a complex test as it involves closures
    # We'll need to extract the handler and test it directly
    thread_ts = "test_thread"
    agent = MagicMock()
    agent._current_input_message = MagicMock()
    agent._input_callback = AsyncMock()
    agent.context = MagicMock(channel_id="test_channel")

    # Create a mock client
    client = AsyncMock()

    # Extract the handler function by mocking the decorator
    confirm_handler = None

    def action_decorator(action_id):
        def wrapper(func):
            nonlocal confirm_handler
            if action_id == "confirm_action":
                confirm_handler = func
            return func

        return wrapper

    agent.app.action = action_decorator

    # Register handlers
    register_chat_thread_handler(thread_ts, agent)

    # Now test the extracted handler
    ack = AsyncMock()
    body = {
        "message": {
            "thread_ts": thread_ts,
            "ts": "message_ts",
        },
    }

    await confirm_handler(ack, body, client)

    ack.assert_called_once()
    client.chat_update.assert_called_once()
    agent._input_callback.assert_called_once()
    assert agent._current_input_message is None
