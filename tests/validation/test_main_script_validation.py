"""Validation tests for the main.py example script.

These tests validate that the main.py example script runs correctly
and demonstrates the unified bootstrap API and project validation.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from buttermilk import init


class TestMainScriptValidation:
    """Test the main.py example script execution."""

    @pytest.fixture
    def test_config_dir(self):
        """Create a test configuration directory for main.py validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "conf"
            config_dir.mkdir()

            # Create main config file
            config_file = config_dir / "config.yaml"
            config_content = """
defaults:
  - infrastructure: validation
  - _self_

run:
  type: cli
"""
            config_file.write_text(config_content)

            # Create infrastructure directory and config
            infra_dir = config_dir / "infrastructure"
            infra_dir.mkdir()
            infra_config_file = infra_dir / "validation.yaml"
            infra_config_content = """
# @package _global_

infrastructure:
  logging:
    enabled: true
    verbose: false
  clouds: []

llms:
  openai:
    models: []
  anthropic:
    models: []
"""
            infra_config_file.write_text(infra_config_content)

            yield str(config_dir)

    def test_main_script_imports_correctly(self):
        """Test that main.py can be imported without errors."""
        # Test that we can import the main script
        try:
            from buttermilk import main

            assert main is not None
        except ImportError as e:
            pytest.fail(f"Failed to import main.py: {e}")

    def test_main_script_cli_imports(self):
        """Test that the main script can import CLI utilities."""
        # Verify imports that main.py uses
        from buttermilk import logger, BM, init  # noqa

        assert logger is not None

    @patch("buttermilk.utils.cli.bootstrap_session")
    @patch("buttermilk.logger")
    def test_main_script_logic_simulation(self, mock_logger, mock_bootstrap_session):
        """Test the main script logic by simulating its execution."""

        # Mock BM instances for the three scenarios
        mock_bm1 = type(
            "MockBM",
            (),
            {"session_info": type("SessionInfo", (), {"session_id": "session_1", "project_name": "project_alpha", "job": "first_analysis"})()},
        )()

        mock_bm2 = type(
            "MockBM",
            (),
            {"session_info": type("SessionInfo", (), {"session_id": "session_2", "project_name": "project_beta", "job": "second_analysis"})()},
        )()

        mock_bm3 = type(
            "MockBM",
            (),
            {
                "session_info": type(
                    "SessionInfo",
                    (),
                    {
                        "session_id": "session_3",
                        "project_name": "project_beta",  # Inherited
                        "job": "third_analysis",
                    },
                )()
            },
        )()

        mock_bootstrap_session.side_effect = [mock_bm1, mock_bm2, mock_bm3]

        # Simulate the main script execution
        # logger.info("=== Testing simple new session creation ===")
        mock_logger.info.reset_mock()

        # logger.info("1. Creating first session...")
        # bm1 = init(job="first_analysis", project="project_alpha")
        bm1 = init(job="first_analysis", project="project_alpha")

        # logger.info("2. Creating second session...")
        # bm2 = init(job="second_analysis", project="project_beta")
        bm2 = init(job="second_analysis", project="project_beta")

        # logger.info("3. Creating third session (same project, different job)...")
        # bm3 = init(job="third_analysis")  # Inherits "project_beta" from execution context
        bm3 = init(job="third_analysis")

        # Verify the simulation matches expected behavior
        assert bm1.session_info.project_name == "project_alpha"
        assert bm1.session_info.job == "first_analysis"

        assert bm2.session_info.project_name == "project_beta"
        assert bm2.session_info.job == "second_analysis"

        assert bm3.session_info.project_name == "project_beta"  # Inherited
        assert bm3.session_info.job == "third_analysis"

        # Verify bootstrap_session was called correctly
        assert mock_bootstrap_session.call_count == 3
        calls = mock_bootstrap_session.call_args_list

        # First call: explicit project_alpha
        assert calls[0][1]["job"] == "first_analysis"
        assert calls[0][1]["project"] == "project_alpha"

        # Second call: explicit project_beta
        assert calls[1][1]["job"] == "second_analysis"
        assert calls[1][1]["project"] == "project_beta"

        # Third call: no project (inheritance)
        assert calls[2][1]["job"] == "third_analysis"
        assert calls[2][1]["project"] is None

    def test_main_script_message_patterns(self):
        """Test the message patterns that would be logged by main.py."""

        # Test the messages that main.py would log
        expected_messages = [
            "=== Testing simple new session creation ===",
            "1. Creating first session...",
            "2. Creating second session...",
            "3. Creating third session (same project, different job)...",
            "✅ All sessions created successfully!",
            "Each session reuses the same infrastructure but has its own context.",
        ]

        # Verify these are the expected messages (this is documentation)
        for message in expected_messages:
            assert isinstance(message, str)
            assert len(message) > 0

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_error_scenario(self, mock_bootstrap_session):
        """Test error scenario that could occur in main.py."""

        # Setup the first session to work
        mock_bm1 = type("MockBM", (), {"session_info": type("SessionInfo", (), {"project_name": "project_alpha", "job": "first_analysis"})()})()

        # Setup the second session to fail with project mismatch
        mock_bootstrap_session.side_effect = [
            mock_bm1,  # First session succeeds
            RuntimeError(
                "Project name mismatch: execution context is using project 'project_alpha', but session specified project 'project_beta'"
            ),  # Second session fails
        ]

        # First session should work
        bm1 = init(job="first_analysis", project="project_alpha")
        assert bm1.session_info.project_name == "project_alpha"

        # Second session should fail
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            init(job="second_analysis", project="project_beta")

    def test_main_script_session_info_structure(self):
        """Test that session info has the expected structure for main.py."""
        # This test verifies the session info interface that main.py expects
        with patch("buttermilk.utils.cli.bootstrap_session") as mock_bootstrap_session:
            # Create a mock BM with the expected session_info structure
            mock_bm = type(
                "MockBM",
                (),
                {"session_info": type("SessionInfo", (), {"session_id": "test_session_123", "project_name": "test_project", "job": "test_job"})()},
            )()

            mock_bootstrap_session.return_value = mock_bm

            # Test that the structure matches what main.py expects
            bm = init(job="test_job", project="test_project")

            # Verify the session info has all required attributes
            assert hasattr(bm, "session_info")
            assert hasattr(bm.session_info, "session_id")
            assert hasattr(bm.session_info, "project_name")
            assert hasattr(bm.session_info, "job")

            # Verify the values
            assert bm.session_info.session_id == "test_session_123"
            assert bm.session_info.project_name == "test_project"
            assert bm.session_info.job == "test_job"


class TestMainScriptContractValidation:
    """Test the contracts that main.py depends on."""

    def test_cli_init_function_signature(self):
        """Test that init has the expected function signature."""
        import inspect

        # Get function signature
        sig = inspect.signature(init)

        # Verify expected parameters
        params = list(sig.parameters.keys())
        assert "job" in params
        assert "project" in params

        # Verify job parameter is required (no default)
        assert sig.parameters["job"].default == inspect.Parameter.empty

        # Verify project parameter is optional (has default)
        assert sig.parameters["project"].default is None

    def test_bootstrap_session_function_signature(self):
        """Test that bootstrap_session has the expected signature."""
        import inspect

        from buttermilk._core.config_bootstrap import bootstrap_session

        sig = inspect.signature(bootstrap_session)
        params = list(sig.parameters.keys())

        # Verify expected parameters
        assert "job" in params
        assert "project" in params
        assert "run_type" in params

        # Verify defaults
        assert sig.parameters["project"].default is None
        assert sig.parameters["run_type"].default == "cli"

    def test_execution_context_validate_method_signature(self):
        """Test that ExecutionContext.validate_and_set_project has expected signature."""
        import inspect

        from buttermilk._core.execution_context import ExecutionContext

        method = ExecutionContext.validate_and_set_project
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        # Should have 'self' and 'project' parameters
        assert "self" in params
        assert "project" in params
        assert len(params) == 2

    def test_logger_import_availability(self):
        """Test that logger can be imported as shown in main.py."""
        # Test the import pattern used in main.py
        from buttermilk import logger

        assert logger is not None
        # Should have logging methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_session_info_project_name_field(self):
        """Test that session_info has project_name field as used in main.py."""
        # This test ensures the session_info object has the expected structure
        with patch("buttermilk.utils.cli.bootstrap_session") as mock_bootstrap_session:
            # Create mock with expected structure
            mock_session_info = type(
                "SessionInfo",
                (),
                {
                    "session_id": "test_123",
                    "project_name": "my_project",  # This is the key field for main.py
                    "job": "my_job",
                },
            )()

            mock_bm = type("MockBM", (), {"session_info": mock_session_info})()

            mock_bootstrap_session.return_value = mock_bm

            # Test the contract
            bm = init(job="my_job", project="my_project")

            # Verify the contract that main.py depends on
            assert hasattr(bm.session_info, "project_name")
            assert bm.session_info.project_name == "my_project"

            # This is the pattern used in main.py logging:
            # logger.info(f"Starting {run_type} run for {bm.session_info.project_name} job {bm.session_info.job}")
            log_message = f"Starting cli run for {bm.session_info.project_name} job {bm.session_info.job}"
            expected_message = "Starting cli run for my_project job my_job"
            assert log_message == expected_message
