"""
TDD Tests for ws_debug_cli.py Refactoring

GitHub Issue #274: Deprecate legacy CLI commands, keep only puppet mode + infrastructure commands

These tests validate the refactored structure BEFORE implementation:
- Legacy commands MUST be removed: start, send, wait, clear-session, session, start_debug, start_server, list_flows
- Infrastructure commands MUST remain: logs, list-logs, test-connection
- Core classes MUST remain functional for puppet mode: NonInteractiveDebugClient

Expected behavior: Tests should FAIL initially (TDD approach)
After refactoring: Tests should PASS
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from buttermilk.debug.ws_debug_cli import NonInteractiveDebugClient, cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_flow_test_client():
    """Mock FlowTestClient for WebSocket operations."""
    with patch("buttermilk.debug.ws_debug_cli.FlowTestClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.session_id = "test-session-123"
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.start_flow = AsyncMock()
        mock_client.send_manager_response = AsyncMock()
        mock_client.ws = MagicMock()
        mock_client.ws.send_json = AsyncMock()

        # Mock the collector for message tracking
        mock_collector = MagicMock()
        mock_collector.all_messages = []
        mock_client.collector = mock_collector

        mock_client_class.return_value = mock_client
        yield mock_client


class TestInfrastructureCommandsRemain:
    """Test suite: Infrastructure commands must still work after refactoring."""

    def test_logs_command_exists_and_functions(self, cli_runner, tmp_path):
        """Test: 'logs' command should still be available and functional."""
        # Create a mock log file
        log_file = tmp_path / "bm_test_project_12345.jsonl"
        log_content = '{"level": "INFO", "message": "Test log entry"}\n'
        log_file.write_text(log_content)

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = [str(log_file)]
            result = cli_runner.invoke(cli, ["logs", "--lines", "10"])

        # Command should execute successfully
        assert result.exit_code == 0, f"logs command failed: {result.output}"
        assert "Test log entry" in result.output or "bm_test_project" in result.output

    def test_logs_command_with_json_output(self, cli_runner, tmp_path):
        """Test: 'logs' command should support JSON output."""
        log_file = tmp_path / "bm_test_project_12345.jsonl"
        log_content = '{"level": "INFO", "message": "Test message"}\n'
        log_file.write_text(log_content)

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = [str(log_file)]
            result = cli_runner.invoke(cli, ["--json-output", "logs"])

        assert result.exit_code == 0
        # Should be valid JSON
        output = json.loads(result.output)
        assert "log_file" in output or "error" in output

    def test_logs_command_with_level_filter(self, cli_runner, tmp_path):
        """Test: 'logs' command should support log level filtering."""
        log_file = tmp_path / "bm_test_project_12345.jsonl"
        log_file.write_text('{"level": "ERROR", "message": "Error message"}\n')

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = [str(log_file)]
            result = cli_runner.invoke(cli, ["logs", "--level", "ERROR"])

        assert result.exit_code == 0

    def test_list_logs_command_exists_and_functions(self, cli_runner, tmp_path):
        """Test: 'list-logs' command should still be available and functional."""
        # Create mock log files
        log_file1 = tmp_path / "bm_project_123.jsonl"
        log_file2 = tmp_path / "bm_project_456.jsonl"
        log_file1.write_text("")
        log_file2.write_text("")

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = [str(log_file1), str(log_file2)]
            result = cli_runner.invoke(cli, ["list-logs"])

        # Command should execute successfully
        assert result.exit_code == 0, f"list-logs command failed: {result.output}"
        # Should show log files or appropriate message
        assert "bm_" in result.output or "No" in result.output

    def test_list_logs_command_with_count_option(self, cli_runner, tmp_path):
        """Test: 'list-logs' command should support --count option."""
        log_files = []
        for i in range(5):
            log_file = tmp_path / f"bm_project_{i}.jsonl"
            log_file.write_text("")
            log_files.append(str(log_file))

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = log_files
            result = cli_runner.invoke(cli, ["list-logs", "--count", "3"])

        assert result.exit_code == 0

    def test_list_logs_command_with_json_output(self, cli_runner, tmp_path):
        """Test: 'list-logs' command should support JSON output."""
        log_file = tmp_path / "bm_project_123.jsonl"
        log_file.write_text("")

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = [str(log_file)]
            result = cli_runner.invoke(cli, ["--json-output", "list-logs"])

        assert result.exit_code == 0
        # Should be valid JSON
        output = json.loads(result.output)
        assert "log_files" in output or isinstance(output, dict)

    def test_test_connection_command_exists_and_functions(self, cli_runner, mock_flow_test_client):
        """Test: 'test-connection' command should still be available and functional."""
        # Mock successful connection
        mock_flow_test_client.connect.return_value = True

        result = cli_runner.invoke(cli, ["test-connection"])

        # Command should execute successfully
        assert result.exit_code == 0, f"test-connection command failed: {result.output}"
        # Should show connection status
        assert "connect" in result.output.lower() or "ws://" in result.output

    def test_test_connection_command_with_custom_host_port(self, cli_runner, mock_flow_test_client):
        """Test: 'test-connection' command should support custom host and port."""
        mock_flow_test_client.connect.return_value = True

        result = cli_runner.invoke(cli, ["--host", "example.com", "--port", "9000", "test-connection"])

        assert result.exit_code == 0

    def test_test_connection_command_with_json_output(self, cli_runner, mock_flow_test_client):
        """Test: 'test-connection' command should support JSON output."""
        mock_flow_test_client.connect.return_value = True

        result = cli_runner.invoke(cli, ["--json-output", "test-connection"])

        assert result.exit_code == 0
        # Should be valid JSON
        output = json.loads(result.output)
        assert "status" in output or "url" in output


class TestLegacyCommandsRemoved:
    """Test suite: Legacy CLI commands must be removed."""

    def test_start_command_removed(self, cli_runner):
        """Test: 'start' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["start", "test_flow", "test query"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'start' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_send_command_removed(self, cli_runner):
        """Test: 'send' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["send", "test message"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'send' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_wait_command_removed(self, cli_runner):
        """Test: 'wait' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["wait"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'wait' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_clear_session_command_removed(self, cli_runner):
        """Test: 'clear-session' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["clear-session"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'clear-session' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_session_command_removed(self, cli_runner):
        """Test: 'session' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["session"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'session' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_start_debug_command_removed(self, cli_runner):
        """Test: 'start-debug' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["start-debug", "trans", "test query"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'start-debug' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_start_server_command_removed(self, cli_runner):
        """Test: 'start-server' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["start-server"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'start-server' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output

    def test_list_flows_command_removed(self, cli_runner):
        """Test: 'list-flows' command should NO LONGER exist."""
        result = cli_runner.invoke(cli, ["list-flows"])

        # Command should not exist - expect error
        assert result.exit_code != 0, "'list-flows' command should be removed but still exists"
        assert "No such command" in result.output or "Error" in result.output


class TestCoreClassesFunctional:
    """Test suite: Core classes must remain functional for puppet mode."""

    def test_noninteractive_client_class_exists(self):
        """Test: NonInteractiveDebugClient class should still exist."""
        # Class should be importable and instantiable
        client = NonInteractiveDebugClient(host="localhost", port=8000)

        assert client is not None
        assert hasattr(client, "host")
        assert hasattr(client, "port")
        assert client.host == "localhost"
        assert client.port == 8000

    def test_noninteractive_client_has_required_attributes(self):
        """Test: NonInteractiveDebugClient should have all required attributes."""
        client = NonInteractiveDebugClient()

        # Core attributes for puppet mode functionality
        assert hasattr(client, "base_url")
        assert hasattr(client, "ws_url")
        assert hasattr(client, "client")
        assert hasattr(client, "console")
        assert hasattr(client, "session_file")

    def test_noninteractive_client_has_session_management_methods(self):
        """Test: NonInteractiveDebugClient should have session management methods."""
        client = NonInteractiveDebugClient()

        # Session management methods (used by puppet mode)
        assert hasattr(client, "save_session")
        assert hasattr(client, "load_session")
        assert callable(client.save_session)
        assert callable(client.load_session)

    def test_noninteractive_client_has_connection_methods(self):
        """Test: NonInteractiveDebugClient should have connection methods."""
        client = NonInteractiveDebugClient()

        # Connection methods (used by puppet mode)
        assert hasattr(client, "connect")
        assert hasattr(client, "disconnect")
        assert callable(client.connect)
        assert callable(client.disconnect)

    def test_noninteractive_client_has_flow_control_methods(self):
        """Test: NonInteractiveDebugClient should have flow control methods."""
        client = NonInteractiveDebugClient()

        # Flow control methods (used by puppet mode)
        assert hasattr(client, "start_flow")
        assert hasattr(client, "send_message")
        assert hasattr(client, "wait_for_messages")
        assert callable(client.start_flow)
        assert callable(client.send_message)
        assert callable(client.wait_for_messages)

    def test_noninteractive_client_has_infrastructure_methods(self):
        """Test: NonInteractiveDebugClient should have infrastructure methods."""
        client = NonInteractiveDebugClient()

        # Infrastructure methods (used by CLI commands)
        assert hasattr(client, "get_logs")
        assert callable(client.get_logs)

    @pytest.mark.anyio
    async def test_client_can_connect(self, mock_flow_test_client):
        """Test: NonInteractiveDebugClient should be able to establish connections."""
        client = NonInteractiveDebugClient(host="localhost", port=8000)

        # Mock successful connection
        mock_flow_test_client.connect.return_value = None

        # Should be able to connect
        result = await client.connect()

        assert result is True or result is None  # Connection succeeded

    @pytest.mark.anyio
    async def test_client_can_start_flow(self, mock_flow_test_client):
        """Test: NonInteractiveDebugClient should be able to start flows (puppet mode)."""
        client = NonInteractiveDebugClient(host="localhost", port=8000)

        # Mock successful flow start
        mock_flow_test_client.connect.return_value = None
        mock_flow_test_client.start_flow.return_value = None
        mock_flow_test_client.collector.all_messages = []

        # Should be able to start a flow
        result = await client.start_flow(
            flow_name="test_flow",
            query="test query",
            wait_time=1,  # Short wait for testing
        )

        assert isinstance(result, dict)
        assert "error" not in result or result.get("session_id") is not None

    @pytest.mark.anyio
    async def test_client_can_send_messages(self, mock_flow_test_client):
        """Test: NonInteractiveDebugClient should be able to send messages (puppet mode)."""
        client = NonInteractiveDebugClient(host="localhost", port=8000)

        # Mock successful message sending
        mock_flow_test_client.connect.return_value = None
        mock_flow_test_client.send_manager_response.return_value = None
        mock_flow_test_client.collector.all_messages = []

        # Create a session file for testing
        client.save_session("test-session-123")

        # Should be able to send messages
        result = await client.send_message(
            message_type="response",
            content="test response",
            wait_time=1,  # Short wait for testing
        )

        assert isinstance(result, dict)
        assert "error" not in result or result.get("session_id") is not None

    @pytest.mark.anyio
    async def test_client_can_wait_for_messages(self, mock_flow_test_client):
        """Test: NonInteractiveDebugClient should be able to wait for messages (puppet mode)."""
        client = NonInteractiveDebugClient(host="localhost", port=8000)

        # Mock successful message waiting
        mock_flow_test_client.connect.return_value = None
        mock_flow_test_client.collector.all_messages = []

        # Create a session file for testing
        client.save_session("test-session-123")

        # Should be able to wait for messages
        result = await client.wait_for_messages(
            wait_time=1,  # Short wait for testing
        )

        assert isinstance(result, dict)
        assert "error" not in result or result.get("session_id") is not None

    @pytest.mark.anyio
    async def test_client_can_get_logs(self, tmp_path):
        """Test: NonInteractiveDebugClient should be able to retrieve logs."""
        client = NonInteractiveDebugClient()

        # Create a mock log file
        log_file = tmp_path / "bm_test_12345.jsonl"
        log_file.write_text('{"level": "INFO", "message": "Test log"}\n')

        # Should be able to get logs
        result = await client.get_logs(lines=10, log_file=str(log_file))

        assert isinstance(result, dict)
        assert "error" not in result or "lines" in result

    def test_session_file_location_is_accessible(self):
        """Test: Session file should be in an accessible temporary location."""
        client = NonInteractiveDebugClient()

        # Session file should be in temp directory
        assert client.session_file is not None
        assert isinstance(client.session_file, Path)
        assert "buttermilk_debug_session.json" in str(client.session_file)

    def test_client_can_save_and_load_session(self, tmp_path):
        """Test: Client should be able to save and load session data."""
        client = NonInteractiveDebugClient()

        # Override session file to use temp path
        client.session_file = tmp_path / "test_session.json"

        # Save a session
        client.save_session("test-session-456")

        # Should be able to load it back
        loaded_session = client.load_session()
        assert loaded_session == "test-session-456"

    def test_client_url_construction(self):
        """Test: Client should properly construct WebSocket URLs."""
        client = NonInteractiveDebugClient(host="example.com", port=9000)

        assert client.base_url == "http://example.com:9000"
        assert client.ws_url == "ws://example.com:9000/ws"


class TestCLIHelp:
    """Test suite: CLI help and structure validation."""

    def test_cli_help_shows_available_commands(self, cli_runner):
        """Test: CLI help should show only infrastructure commands."""
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0

        # Should show infrastructure commands
        assert "logs" in result.output
        assert "list-logs" in result.output
        assert "test-connection" in result.output

        # Should NOT show legacy commands
        assert "start " not in result.output or "Commands:" not in result.output
        assert "send" not in result.output or "Commands:" not in result.output
        assert "wait" not in result.output or "Commands:" not in result.output

    def test_cli_accepts_global_options(self, cli_runner):
        """Test: CLI should accept global options like --host, --port, --json-output."""
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--json-output" in result.output


class TestBackwardCompatibilityNotes:
    """Test suite: Document breaking changes and migration paths."""

    def test_puppet_mode_methods_available_programmatically(self):
        """
        Test: Puppet mode functionality should be available programmatically.

        NOTE: Users should migrate from CLI commands to DebugAgent puppet mode:
        - Old: `ws_debug_cli start flow_name query`
        - New: Use DebugAgent.start_puppet_mode() + puppet_start_flow()
        """
        client = NonInteractiveDebugClient()

        # All puppet mode methods should be available programmatically
        assert hasattr(client, "start_flow")
        assert hasattr(client, "send_message")
        assert hasattr(client, "wait_for_messages")
        assert hasattr(client, "connect")
        assert hasattr(client, "disconnect")

    def test_infrastructure_commands_remain_for_debugging(self, cli_runner, tmp_path):
        """
        Test: Infrastructure commands remain for operational debugging.

        NOTE: These commands are kept because they are useful for:
        - Viewing logs without complex setup (logs, list-logs)
        - Testing connectivity (test-connection)
        """
        # Create mock log
        log_file = tmp_path / "bm_test_12345.jsonl"
        log_file.write_text('{"level": "INFO", "message": "Test"}\n')

        with patch("buttermilk.debug.ws_debug_cli.glob.glob") as mock_glob:
            mock_glob.return_value = [str(log_file)]

            # All infrastructure commands should work
            result_logs = cli_runner.invoke(cli, ["logs"])
            result_list = cli_runner.invoke(cli, ["list-logs"])

            assert result_logs.exit_code == 0
            assert result_list.exit_code == 0


# Mark all tests with appropriate markers
pytestmark = pytest.mark.unit
