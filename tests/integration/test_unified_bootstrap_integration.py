"""Integration tests for the unified bootstrap API with real configurations.

These tests verify that the unified bootstrap functions work with real configuration
files and demonstrate end-to-end functionality including project validation.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from buttermilk._core.config_bootstrap import bootstrap_session, bootstrap_session_with_config


class TestUnifiedBootstrapIntegration:
    """Integration tests for unified bootstrap API with real configurations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary configuration directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "conf"
            config_dir.mkdir()

            # Create main config file
            config_file = config_dir / "config.yaml"
            config_content = """
defaults:
  - infrastructure: minimal
  - _self_

run:
  type: test
"""
            config_file.write_text(config_content)

            # Create infrastructure directory and config
            infra_dir = config_dir / "infrastructure"
            infra_dir.mkdir()
            infra_config_file = infra_dir / "minimal.yaml"
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

    def test_cli_init_with_real_config_first_session(self, temp_config_dir):
        """Test CLI init with real config for first session requiring project."""
        # Mock logger to avoid actual logging infrastructure
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                bm = init(job="integration_test", project="test_project", config_dir=temp_config_dir)

                # Verify BM instance is created correctly
                assert bm is not None
                assert hasattr(bm, "session_info")
                assert bm.session_info.project_name == "test_project"
                assert bm.session_info.job == "integration_test"
                assert hasattr(bm.session_info, "session_id")

    def test_cli_init_with_real_config_subsequent_session_inheritance(self, temp_config_dir):
        """Test CLI init with real config for subsequent session that inherits project."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                # First session sets the project
                bm1 = init(job="first_job", project="shared_project", config_dir=temp_config_dir)

                # Second session should inherit the project
                bm2 = init(
                    job="second_job",
                    config_dir=temp_config_dir,
                    # Note: no project specified - should inherit "shared_project"
                )

                # Verify both sessions have the same project
                assert bm1.session_info.project_name == "shared_project"
                assert bm2.session_info.project_name == "shared_project"

                # But different jobs
                assert bm1.session_info.job == "first_job"
                assert bm2.session_info.job == "second_job"

                # And different session IDs
                assert bm1.session_info.session_id != bm2.session_info.session_id

    def test_cli_init_with_real_config_project_mismatch_error(self, temp_config_dir):
        """Test that project mismatch raises error with real config."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                # First session sets the project
                bm1 = init(job="first_job", project="original_project", config_dir=temp_config_dir)

                # Second session with different project should fail
                with pytest.raises(RuntimeError, match="Project name mismatch"):
                    init(job="conflicting_job", project="different_project", config_dir=temp_config_dir)

    def test_bootstrap_session_function_with_real_config(self, temp_config_dir):
        """Test the bootstrap_session function directly with real config."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                bm = bootstrap_session(job="direct_bootstrap_test", project="bootstrap_project", run_type="test", config_dir=temp_config_dir)

                # Verify BM instance
                assert bm is not None
                assert bm.session_info.project_name == "bootstrap_project"
                assert bm.session_info.job == "direct_bootstrap_test"

    def test_bootstrap_session_with_config_function_with_real_config(self, temp_config_dir):
        """Test the bootstrap_session_with_config function directly with real config."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                bm, config = bootstrap_session_with_config(
                    job="config_bootstrap_test", project="config_project", run_type="test", config_dir=temp_config_dir
                )

                # Verify BM instance
                assert bm is not None
                assert bm.session_info.project_name == "config_project"
                assert bm.session_info.job == "config_bootstrap_test"

                # Verify config object
                assert config is not None
                assert hasattr(config, "infrastructure")

    def test_cli_init_with_overrides_real_config(self, temp_config_dir):
        """Test CLI init with custom overrides and real config."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                custom_overrides = ["infrastructure.logging.verbose=true"]

                bm = init(job="override_test", project="override_project", config_dir=temp_config_dir, overrides=custom_overrides)

                assert bm.session_info.project_name == "override_project"
                assert bm.session_info.job == "override_test"

    def test_real_config_error_handling_missing_project(self, temp_config_dir):
        """Test error handling with real config when project is missing for first session."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                # First session without project should fail
                with pytest.raises(RuntimeError, match="project parameter is required"):
                    init(
                        job="missing_project_test",
                        config_dir=temp_config_dir,
                        # Note: no project specified for first session
                    )

    def test_session_isolation_with_real_config(self, temp_config_dir):
        """Test that sessions are properly isolated while sharing execution context."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                # Create multiple sessions with same project
                bm1 = init(job="isolation_test_1", project="isolation_project", config_dir=temp_config_dir)

                bm2 = init(job="isolation_test_2", config_dir=temp_config_dir)

                # Verify they share the same project (from execution context)
                assert bm1.session_info.project_name == "isolation_project"
                assert bm2.session_info.project_name == "isolation_project"

                # But have different session IDs and jobs
                assert bm1.session_info.session_id != bm2.session_info.session_id
                assert bm1.session_info.job != bm2.session_info.job

                # And are different BM instances
                assert bm1 is not bm2


class TestRealConfigurationValidation:
    """Test configuration validation with real config files."""

    @pytest.fixture
    def invalid_config_dir(self):
        """Create a temporary configuration directory with invalid config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "conf"
            config_dir.mkdir()

            # Create config file with missing required sections
            config_file = config_dir / "config.yaml"
            config_content = """
# Missing infrastructure section
run:
  type: test
"""
            config_file.write_text(config_content)

            yield str(config_dir)

    def test_error_handling_with_invalid_config(self, invalid_config_dir):
        """Test error handling when configuration is invalid."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            # Should fail gracefully with configuration error
            with pytest.raises(Exception):  # Could be RuntimeError or other config-related error
                init(job="invalid_config_test", project="test_project", config_dir=invalid_config_dir)

    def test_config_directory_path_handling(self):
        """Test that configuration directory paths are handled correctly."""
        # Test with non-existent directory
        with pytest.raises(Exception):  # Should fail with config loading error
            init(job="path_test", project="test_project", config_dir="/non/existent/path")


class TestMainScriptRealExecution:
    """Test the main.py scenarios with real (minimal) configuration."""

    @pytest.fixture
    def main_script_config_dir(self):
        """Create a configuration directory suitable for main.py scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "conf"
            config_dir.mkdir()

            # Create main config
            config_file = config_dir / "config.yaml"
            config_content = """
defaults:
  - infrastructure: testing
  - _self_

run:
  type: cli
"""
            config_file.write_text(config_content)

            # Create infrastructure config
            infra_dir = config_dir / "infrastructure"
            infra_dir.mkdir()
            infra_config_file = infra_dir / "testing.yaml"
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
"""
            infra_config_file.write_text(infra_config_content)

            yield str(config_dir)

    def test_main_script_scenario_with_real_config(self, main_script_config_dir):
        """Test the complete main.py scenario with real configuration."""
        with patch("buttermilk._core.config_bootstrap.logger"):
            with patch("buttermilk._core.execution_context.logger"):
                # Scenario 1: First session - project required
                bm1 = init(job="first_analysis", project="project_alpha", config_dir=main_script_config_dir)

                # Scenario 2: Second session - different project
                bm2 = init(job="second_analysis", project="project_beta", config_dir=main_script_config_dir)

                # Scenario 3: Third session - inherits project_beta
                bm3 = init(job="third_analysis", config_dir=main_script_config_dir)

                # Verify all scenarios work correctly
                assert bm1.session_info.project_name == "project_alpha"
                assert bm1.session_info.job == "first_analysis"

                assert bm2.session_info.project_name == "project_beta"
                assert bm2.session_info.job == "second_analysis"

                assert bm3.session_info.project_name == "project_beta"  # Inherited
                assert bm3.session_info.job == "third_analysis"

                # Verify all sessions have unique IDs
                session_ids = {bm1.session_info.session_id, bm2.session_info.session_id, bm3.session_info.session_id}
                assert len(session_ids) == 3  # All unique
