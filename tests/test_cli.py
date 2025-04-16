import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk.runner.cli import main


@pytest.fixture
def mock_hydra():
    """Create a mock Hydra for testing the CLI."""
    mock_instantiate = MagicMock()
    mock_main = MagicMock()

    with patch("buttermilk.runner.cli.hydra", autospec=True) as mock_hydra:
        mock_hydra.utils.instantiate = mock_instantiate
        mock_hydra.main = lambda version_base, config_path, config_name: lambda f: mock_main
        yield mock_hydra, mock_instantiate, mock_main


@pytest.fixture
def mock_console_app():
    """Create mock environment for console app testing."""
    mock_objs = MagicMock()
    mock_objs.ui = "console"
    mock_objs.flows = {
        "test_flow": {
            "orchestrator": "Selector",
            "name": "test_flow",
            "description": "Test flow",
        },
    }

    mock_orchestrator = AsyncMock()
    mock_selector = MagicMock(return_value=mock_orchestrator)

    with patch("buttermilk.runner.cli.globals") as mock_globals:
        mock_globals.return_value = {"Selector": mock_selector}
        yield mock_objs, mock_orchestrator, mock_selector


@pytest.fixture
def mock_slack_app():
    """Create mock environment for slack app testing."""
    mock_objs = MagicMock()
    mock_objs.ui = "slackbot"
    mock_objs.bm = MagicMock()
    mock_objs.bm.credentials = {
        "MODBOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
    }
    mock_objs.flows = {
        "test_flow": {
            "orchestrator": "AutogenOrchestrator",
            "name": "test_flow",
        },
    }

    mock_slack_app = MagicMock()
    mock_handler = MagicMock()
    mock_init_slack = MagicMock(return_value=(mock_slack_app, mock_handler))
    mock_register_handlers = AsyncMock()

    with patch("buttermilk.runner.cli.initialize_slack_bot", mock_init_slack), \
         patch("buttermilk.runner.cli.register_handlers", mock_register_handlers), \
         patch("buttermilk.runner.cli.asyncio", autospec=True) as mock_asyncio:

        mock_asyncio.Queue = MagicMock()
        mock_asyncio.sleep = AsyncMock()

        yield (
            mock_objs,
            mock_slack_app,
            mock_handler,
            mock_init_slack,
            mock_register_handlers,
            mock_asyncio,
        )


def test_main_console_mode(mock_hydra, mock_console_app):
    """Test CLI main function in console mode."""
    _, mock_instantiate, mock_main = mock_hydra
    mock_objs, mock_orchestrator, mock_selector = mock_console_app

    mock_instantiate.return_value = mock_objs
    mock_cfg = MagicMock()
    mock_cfg.flow = "test_flow"

    # Call the function through the mock
    main(mock_cfg)

    # Verify Sequencer was created with correct params
    mock_selector.assert_called_once()
    # Verify we tried to run it
    mock_orchestrator.run.assert_called_once()


def test_main_slack_mode_environment_vars(mock_hydra, mock_slack_app):
    """Test CLI main function sets environment variables in slack mode."""
    _, mock_instantiate, mock_main = mock_hydra
    (
        mock_objs,
        mock_slack_app,
        mock_handler,
        mock_init_slack,
        mock_register_handlers,
        mock_asyncio,
    ) = mock_slack_app

    mock_instantiate.return_value = mock_objs
    mock_cfg = MagicMock()

    # Clear env vars if present
    bot_token_backup = os.environ.pop("SLACK_BOT_TOKEN", None)
    app_token_backup = os.environ.pop("SLACK_APP_TOKEN", None)

    try:
        # Call the function through the mock
        main(mock_cfg)

        # Check if env vars were set
        assert os.environ["SLACK_BOT_TOKEN"] == "xoxb-test"
        assert os.environ["SLACK_APP_TOKEN"] == "xapp-test"
    finally:
        # Restore environment if needed
        if bot_token_backup:
            os.environ["SLACK_BOT_TOKEN"] = bot_token_backup
        else:
            os.environ.pop("SLACK_BOT_TOKEN", None)

        if app_token_backup:
            os.environ["SLACK_APP_TOKEN"] = app_token_backup
        else:
            os.environ.pop("SLACK_APP_TOKEN", None)


def test_main_slack_mode_initialization(mock_hydra, mock_slack_app):
    """Test slack app initialization in CLI main function."""
    _, mock_instantiate, mock_main = mock_hydra
    (
        mock_objs,
        mock_slack_app,
        mock_handler,
        mock_init_slack,
        mock_register_handlers,
        mock_asyncio,
    ) = mock_slack_app

    mock_instantiate.return_value = mock_objs
    mock_cfg = MagicMock()

    # Call the function through the mock
    main(mock_cfg)

    # Verify slack app initialization
    mock_init_slack.assert_called_once_with(
        bot_token="xoxb-test",
        app_token="xapp-test",
        loop=mock_asyncio.get_event_loop.return_value,
    )

    # Verify handler registration
    mock_register_handlers.assert_called_once_with(
        slack_app=mock_slack_app,
        flows=mock_objs.flows,
        orchestrator_tasks=mock_asyncio.Queue.return_value,
    )

    # Verify app started
    mock_handler.start_async.assert_called_once()


def test_main_slack_mode_event_loop(mock_hydra, mock_slack_app):
    """Test event loop behavior in slack mode."""
    _, mock_instantiate, mock_main = mock_hydra
    (
        mock_objs,
        mock_slack_app,
        mock_handler,
        mock_init_slack,
        mock_register_handlers,
        mock_asyncio,
    ) = mock_slack_app

    mock_instantiate.return_value = mock_objs
    mock_cfg = MagicMock()

    # Call the function through the mock
    main(mock_cfg)

    # Verify loop configuration
    mock_loop = mock_asyncio.get_event_loop.return_value
    assert mock_loop.slow_callback_duration == 1.0

    # Verify we created a task for handler.start_async
    mock_loop.create_task.assert_called_with(mock_handler.start_async.return_value)

    # Verify we ran the runloop
    mock_loop.run_until_complete.assert_called_once()
