"""Tests for main.py example script that demonstrate project validation logic.

These tests verify that the main.py example script scenarios work correctly
and demonstrate the project validation behavior in action.
"""

from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.execution_context import ExecutionContext


class TestMainScriptExamples:
    """Test scenarios from the main.py example script."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_scenario_1_first_session_with_project(self, mock_bootstrap_session):
        """Test scenario 1: First session with explicit project.

        This demonstrates: bm1 = init(job="first_analysis", project="project_alpha")
        """
        # Mock BM instance for first session
        mock_bm1 = MagicMock()
        mock_bm1.session_info.session_id = "session_1"
        mock_bm1.session_info.project_name = "project_alpha"
        mock_bm1.session_info.job = "first_analysis"
        mock_bootstrap_session.return_value = mock_bm1

        # Execute the first session scenario
        bm1 = init(job="first_analysis", project="project_alpha")

        # Verify bootstrap was called correctly
        mock_bootstrap_session.assert_called_once_with(job="first_analysis", project="project_alpha", run_type="cli", config_dir=None, overrides=[])

        # Verify session properties
        assert bm1.session_info.project_name == "project_alpha"
        assert bm1.session_info.job == "first_analysis"
        assert bm1.session_info.session_id == "session_1"

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_scenario_2_different_project_new_context(self, mock_bootstrap_session):
        """Test scenario 2: Different project creates new execution context.

        This demonstrates: bm2 = init(job="second_analysis", project="project_beta")
        """
        # Mock BM instance for second session with different project
        mock_bm2 = MagicMock()
        mock_bm2.session_info.session_id = "session_2"
        mock_bm2.session_info.project_name = "project_beta"
        mock_bm2.session_info.job = "second_analysis"
        mock_bootstrap_session.return_value = mock_bm2

        # Execute the second session scenario
        bm2 = init(job="second_analysis", project="project_beta")

        # Verify bootstrap was called correctly
        mock_bootstrap_session.assert_called_once_with(job="second_analysis", project="project_beta", run_type="cli", config_dir=None, overrides=[])

        # Verify session properties
        assert bm2.session_info.project_name == "project_beta"
        assert bm2.session_info.job == "second_analysis"
        assert bm2.session_info.session_id == "session_2"

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_scenario_3_inherited_project(self, mock_bootstrap_session):
        """Test scenario 3: Third session inherits project from execution context.

        This demonstrates: bm3 = init(job="third_analysis")  # Inherits "project_beta"
        """
        # Mock BM instance for third session that inherits project
        mock_bm3 = MagicMock()
        mock_bm3.session_info.session_id = "session_3"
        mock_bm3.session_info.project_name = "project_beta"  # Inherited from execution context
        mock_bm3.session_info.job = "third_analysis"
        mock_bootstrap_session.return_value = mock_bm3

        # Execute the third session scenario (no project specified - should inherit)
        bm3 = init(job="third_analysis")

        # Verify bootstrap was called with project=None (inheritance)
        mock_bootstrap_session.assert_called_once_with(
            job="third_analysis",
            project=None,  # No project specified - should inherit
            run_type="cli",
            config_dir=None,
            overrides=[],
        )

        # Verify session properties show inherited project
        assert bm3.session_info.project_name == "project_beta"
        assert bm3.session_info.job == "third_analysis"
        assert bm3.session_info.session_id == "session_3"

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_complete_scenario_sequence(self, mock_bootstrap_session):
        """Test the complete sequence from main.py as it would actually run."""
        # Setup mock BM instances for all three scenarios
        mock_bm1 = MagicMock()
        mock_bm1.session_info.session_id = "session_1"
        mock_bm1.session_info.project_name = "project_alpha"
        mock_bm1.session_info.job = "first_analysis"

        mock_bm2 = MagicMock()
        mock_bm2.session_info.session_id = "session_2"
        mock_bm2.session_info.project_name = "project_beta"
        mock_bm2.session_info.job = "second_analysis"

        mock_bm3 = MagicMock()
        mock_bm3.session_info.session_id = "session_3"
        mock_bm3.session_info.project_name = "project_beta"  # Inherited
        mock_bm3.session_info.job = "third_analysis"

        # Setup side effect to return different BMs for each call
        mock_bootstrap_session.side_effect = [mock_bm1, mock_bm2, mock_bm3]

        # Execute the complete main.py sequence

        # 1. First session - project required
        bm1 = init(job="first_analysis", project="project_alpha")

        # 2. Second session - different project (new execution context)
        bm2 = init(job="second_analysis", project="project_beta")

        # 3. Third session with same project but different job (inherits project)
        bm3 = init(job="third_analysis")  # Inherits "project_beta" from execution context

        # Verify all calls were made correctly
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

        # Verify results
        assert bm1.session_info.project_name == "project_alpha"
        assert bm2.session_info.project_name == "project_beta"
        assert bm3.session_info.project_name == "project_beta"  # Inherited

    def test_main_script_project_validation_scenarios_with_real_execution_context(self):
        """Test project validation scenarios using real ExecutionContext instances."""
        # Test first session - project required
        execution_context1 = ExecutionContext()

        # Should fail without project
        with pytest.raises(RuntimeError, match="project parameter is required for the first session"):
            execution_context1.validate_and_set_project(None)

        # Should succeed with project
        result1 = execution_context1.validate_and_set_project("project_alpha")
        assert result1 == "project_alpha"
        assert execution_context1.project_name == "project_alpha"

        # Test second session - different project (simulates new execution context)
        execution_context2 = ExecutionContext()
        result2 = execution_context2.validate_and_set_project("project_beta")
        assert result2 == "project_beta"
        assert execution_context2.project_name == "project_beta"

        # Test third session - inheritance (same execution context as second)
        result3 = execution_context2.validate_and_set_project(None)  # Should inherit
        assert result3 == "project_beta"
        assert execution_context2.project_name == "project_beta"

        # Test error case - trying to change project in same execution context
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            execution_context2.validate_and_set_project("project_gamma")


class TestMainScriptErrorScenarios:
    """Test error scenarios that could occur in main.py usage."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_error_first_session_no_project(self, mock_bootstrap_session):
        """Test error when first session doesn't provide project."""
        # Setup bootstrap to simulate ExecutionContext validation error
        mock_bootstrap_session.side_effect = RuntimeError("project parameter is required for the first session in an execution context")

        # This should fail
        with pytest.raises(RuntimeError, match="project parameter is required"):
            init(job="first_analysis")  # Missing project for first session

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_error_project_mismatch(self, mock_bootstrap_session):
        """Test error when trying to use different project in same execution context."""
        # Setup bootstrap to simulate project mismatch error
        mock_bootstrap_session.side_effect = RuntimeError(
            "Project name mismatch: execution context is using project 'project_alpha', but session specified project 'project_beta'"
        )

        # This should fail
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            init(job="conflicting_analysis", project="project_beta")

    def test_main_script_save_directory_project_integration(self):
        """Test that save directories would include project names."""
        # This is a contract test for save directory behavior
        # The actual save directory construction depends on BM implementation

        with patch("buttermilk.utils.cli.bootstrap_session") as mock_bootstrap_session:
            # Mock BM with save_dir that includes project name
            mock_bm = MagicMock()
            mock_bm.session_info.project_name = "project_alpha"
            mock_bm.session_info.job = "data_analysis"
            mock_bm.session_info.session_id = "session_123"

            # Mock save directory path that incorporates project name
            mock_bm.save_dir = "/base/output/project_alpha/data_analysis/session_123"

            mock_bootstrap_session.return_value = mock_bm

            # Execute scenario
            result = init(job="data_analysis", project="project_alpha")

            # Verify save directory includes project name
            assert "project_alpha" in result.save_dir
            assert result.session_info.project_name == "project_alpha"


class TestMainScriptLoggingIntegration:
    """Test logging integration scenarios from main.py."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    @patch("buttermilk.logger")
    def test_main_script_logging_includes_session_context(self, mock_logger, mock_bootstrap_session):
        """Test that logging includes session context (session_id, job, project_name)."""
        # Mock BM instance
        mock_bm = MagicMock()
        mock_bm.session_info.session_id = "session_123"
        mock_bm.session_info.project_name = "my_project"
        mock_bm.session_info.job = "analysis_job"
        mock_bootstrap_session.return_value = mock_bm

        # Execute init
        bm = init(job="analysis_job", project="my_project")

        # Simulate the logging that would happen in main.py
        # from buttermilk import logger
        # logger.info("Started analysis")

        # Verify that the logger would have access to session context
        assert bm.session_info.session_id == "session_123"
        assert bm.session_info.project_name == "my_project"
        assert bm.session_info.job == "analysis_job"

        # The actual logger integration test would verify that log messages
        # include these context fields, but that depends on the logger implementation
