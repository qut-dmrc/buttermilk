"""Tests for ws_debug_cli command restoration (Issue #274 resolution).

This test suite validates that:
1. Infrastructure commands (logs, list-logs, test-connection) still work
2. Flow control commands (start, send, wait, session, clear-session) are restored
3. Commands default to JSON output (optimized for LLMs)
4. --pretty flag provides human-readable output
5. DebugAgent puppet mode is unaffected (regression test)
"""

import pytest
from click.testing import CliRunner

from buttermilk.debug.ws_debug_cli import cli
from buttermilk.debug.debug_agent import DebugAgent


class TestInfrastructureCommandsPreserved:
    """Test that infrastructure commands still work after deprecation."""

    def test_logs_command_exists(self):
        """Verify logs command is still available."""
        runner = CliRunner()
        # Use --help to check if command exists without executing it
        result = runner.invoke(cli, ['logs', '--help'])
        assert result.exit_code == 0, f"logs command should exist. Output: {result.output}"
        assert 'Show recent log lines' in result.output or 'log' in result.output.lower()

    def test_list_logs_command_exists(self):
        """Verify list-logs command is still available."""
        runner = CliRunner()
        result = runner.invoke(cli, ['list-logs', '--help'])
        assert result.exit_code == 0, f"list-logs command should exist. Output: {result.output}"
        assert 'List' in result.output or 'log' in result.output.lower()

    def test_test_connection_command_exists(self):
        """Verify test-connection command is still available."""
        runner = CliRunner()
        result = runner.invoke(cli, ['test-connection', '--help'])
        assert result.exit_code == 0, f"test-connection command should exist. Output: {result.output}"
        assert 'Test' in result.output or 'connection' in result.output.lower()


class TestFlowControlCommandsRestored:
    """Test that flow control commands are restored and work correctly."""

    def test_start_command_exists(self):
        """Verify start command is restored."""
        runner = CliRunner()
        result = runner.invoke(cli, ['start', '--help'])
        assert result.exit_code == 0, f"start command should exist. Output: {result.output}"
        assert 'Start a flow' in result.output or 'flow' in result.output.lower()

    def test_send_command_exists(self):
        """Verify send command is restored."""
        runner = CliRunner()
        result = runner.invoke(cli, ['send', '--help'])
        assert result.exit_code == 0, f"send command should exist. Output: {result.output}"
        assert 'Send a message' in result.output or 'message' in result.output.lower()

    def test_wait_command_exists(self):
        """Verify wait command is restored."""
        runner = CliRunner()
        result = runner.invoke(cli, ['wait', '--help'])
        assert result.exit_code == 0, f"wait command should exist. Output: {result.output}"
        assert 'Wait' in result.output or 'message' in result.output.lower()

    def test_session_command_exists(self):
        """Verify session command is restored."""
        runner = CliRunner()
        result = runner.invoke(cli, ['session', '--help'])
        assert result.exit_code == 0, f"session command should exist. Output: {result.output}"
        assert 'session' in result.output.lower()

    def test_clear_session_command_exists(self):
        """Verify clear-session command is restored."""
        runner = CliRunner()
        result = runner.invoke(cli, ['clear-session', '--help'])
        assert result.exit_code == 0, f"clear-session command should exist. Output: {result.output}"
        assert 'clear' in result.output.lower() or 'session' in result.output.lower()

    def test_start_command_has_record_option(self):
        """Verify start command has --record option."""
        runner = CliRunner()
        result = runner.invoke(cli, ['start', '--help'])
        assert '--record' in result.output, "start command should have --record option"

    def test_start_command_has_criteria_option(self):
        """Verify start command has --criteria option."""
        runner = CliRunner()
        result = runner.invoke(cli, ['start', '--help'])
        assert '--criteria' in result.output, "start command should have --criteria option"


class TestOutputModes:
    """Test JSON vs pretty output modes."""

    def test_pretty_flag_exists(self):
        """Verify --pretty flag is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert '--pretty' in result.output, "CLI should have --pretty flag"

    def test_json_output_flag_exists(self):
        """Verify --json-output flag is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert '--json-output' in result.output, "CLI should have --json-output flag"

    def test_session_command_outputs_json_by_default(self):
        """Verify commands output JSON by default."""
        runner = CliRunner()
        # Session command with no saved session should output JSON
        result = runner.invoke(cli, ['session'])
        assert result.exit_code == 0
        # Should be valid JSON (will have "error" key for no session)
        import json
        try:
            data = json.loads(result.output)
            assert "error" in data or "session_id" in data
        except json.JSONDecodeError:
            pytest.fail(f"Output should be JSON by default. Got: {result.output}")

    def test_clear_session_outputs_json_by_default(self):
        """Verify clear-session outputs JSON by default."""
        runner = CliRunner()
        result = runner.invoke(cli, ['clear-session'])
        assert result.exit_code == 0
        import json
        try:
            data = json.loads(result.output)
            assert "status" in data
        except json.JSONDecodeError:
            pytest.fail(f"Output should be JSON by default. Got: {result.output}")


class TestDebugAgentPuppetModePreserved:
    """Regression test: Ensure DebugAgent puppet mode is unaffected."""

    def test_debug_agent_has_puppet_methods(self):
        """Verify DebugAgent still has puppet mode methods."""
        agent = DebugAgent()
        
        # Check that puppet mode methods exist
        assert hasattr(agent, 'start_puppet_mode'), "DebugAgent should have start_puppet_mode"
        assert hasattr(agent, 'puppet_start_flow'), "DebugAgent should have puppet_start_flow"
        assert hasattr(agent, 'puppet_send_response'), "DebugAgent should have puppet_send_response"
        assert hasattr(agent, 'puppet_get_messages'), "DebugAgent should have puppet_get_messages"
        assert hasattr(agent, 'puppet_get_summary'), "DebugAgent should have puppet_get_summary"
        assert hasattr(agent, 'stop_puppet_mode'), "DebugAgent should have stop_puppet_mode"

    def test_debug_agent_puppet_methods_callable(self):
        """Verify puppet mode methods are callable (signature check)."""
        agent = DebugAgent()
        
        # Check methods are callable
        assert callable(agent.start_puppet_mode)
        assert callable(agent.puppet_start_flow)
        assert callable(agent.puppet_send_response)
        assert callable(agent.puppet_get_messages)
        assert callable(agent.puppet_get_summary)
        assert callable(agent.stop_puppet_mode)
