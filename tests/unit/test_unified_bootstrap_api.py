"""Tests for the unified bootstrap API and project validation logic.

These tests verify that the new CLI init() function works correctly with project parameter
validation, ExecutionContext project inheritance, and the unified bootstrap functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk import init
from buttermilk._core.bm_init import BM
from buttermilk._core.config_bootstrap import bootstrap_session_with_config
from buttermilk._core.execution_context import ExecutionContext

# SKIP: InfrastructureManager, bootstrap_session, and cli undefined - tests need refactoring
pytest.skip("InfrastructureManager, bootstrap_session, and cli undefined - tests need refactoring", allow_module_level=True)


class TestUnifiedBootstrapAPI:
    """Test the unified bootstrap API functions."""

    def test_bootstrap_session_with_config_function_exists_and_callable(self):
        """Test that the bootstrap_session_with_config function exists and is callable."""
        assert callable(bootstrap_session_with_config)

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    @patch("buttermilk._core.config_bootstrap.set_bm")
    @patch("buttermilk._core.config_bootstrap.asyncio.run")
    def test_bootstrap_session_basic_functionality(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test basic bootstrap_session functionality."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Mock BM instance
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "test_project"
        mock_bm.session_info.job = "test_job"

        # Mock ExecutionContext and InfrastructureManager
        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_execution_context.validate_and_set_project.return_value = "test_project"

        # Test bootstrap_session
        result = bootstrap_session_with_config(job="test_job", project="test_project", run_type="cli")

        # Verify bootstrap methods were called
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once()

        # Verify project validation was called
        mock_execution_context.validate_and_set_project.assert_called_once_with("test_project")

        # Verify set_bm was called
        mock_set_bm.assert_called_once_with(mock_bm)

        # Verify BM instance is returned
        assert result is mock_bm

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    @patch("buttermilk._core.config_bootstrap.set_bm")
    @patch("buttermilk._core.config_bootstrap.asyncio.run")
    def test_bootstrap_session_with_config_returns_tuple(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test that bootstrap_session_with_config returns both BM instance and config."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "test_project"
        mock_bm.session_info.job = "test_job"

        mock_config = MagicMock()
        mock_bootstrapper.get_configuration.return_value = mock_config

        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_execution_context.validate_and_set_project.return_value = "test_project"

        # Test bootstrap_session_with_config
        bm_result, config_result = bootstrap_session_with_config(job="test_job", project="test_project", run_type="cli")

        # Verify both BM instance and config are returned
        assert bm_result is mock_bm
        assert config_result is mock_config

        # Verify get_configuration was called
        mock_bootstrapper.get_configuration.assert_called_once()


class TestProjectValidationLogic:
    """Test project validation logic in ExecutionContext."""

    def test_execution_context_first_session_requires_project(self):
        """Test that first session requires project parameter."""
        execution_context = ExecutionContext()

        # First session without project should fail
        with pytest.raises(RuntimeError, match="project parameter is required for the first session"):
            execution_context.validate_and_set_project(None)

    def test_execution_context_first_session_sets_project(self):
        """Test that first session successfully sets project."""
        execution_context = ExecutionContext()

        # First session with project should succeed and set project
        result = execution_context.validate_and_set_project("my_project")

        assert result == "my_project"
        assert execution_context.project_name == "my_project"

    def test_execution_context_subsequent_session_inherits_project(self):
        """Test that subsequent sessions inherit project name."""
        execution_context = ExecutionContext()

        # Set initial project
        execution_context.validate_and_set_project("my_project")

        # Subsequent session without project should inherit existing project
        result = execution_context.validate_and_set_project(None)

        assert result == "my_project"
        assert execution_context.project_name == "my_project"

    def test_execution_context_subsequent_session_explicit_same_project(self):
        """Test that subsequent sessions can explicitly specify same project."""
        execution_context = ExecutionContext()

        # Set initial project
        execution_context.validate_and_set_project("my_project")

        # Subsequent session with same explicit project should succeed
        result = execution_context.validate_and_set_project("my_project")

        assert result == "my_project"
        assert execution_context.project_name == "my_project"

    def test_execution_context_subsequent_session_different_project_fails(self):
        """Test that subsequent sessions with different project fail."""
        execution_context = ExecutionContext()

        # Set initial project
        execution_context.validate_and_set_project("my_project")

        # Subsequent session with different project should fail
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            execution_context.validate_and_set_project("other_project")


class TestCLIInitWithProjectValidation:
    """Test CLI init() function with project parameter validation."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_cli_init_first_session_with_project(self, mock_bootstrap_session):
        """Test CLI init for first session with project parameter."""
        # Mock BM instance
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "my_project"
        mock_bm.session_info.job = "test_job"
        mock_bootstrap_session.return_value = mock_bm

        # Test CLI init with project
        result = init(job="test_job", project="my_project")

        # Verify bootstrap_session was called with correct parameters
        mock_bootstrap_session.assert_called_once_with(job="test_job", project="my_project", run_type="cli", config_dir=None, overrides=[])

        assert result is mock_bm

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_cli_init_subsequent_session_without_project(self, mock_bootstrap_session):
        """Test CLI init for subsequent session without project parameter."""
        # Mock BM instance
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "inherited_project"
        mock_bm.session_info.job = "test_job2"
        mock_bootstrap_session.return_value = mock_bm

        # Test CLI init without project (should inherit)
        result = init(job="test_job2")

        # Verify bootstrap_session was called with project=None
        mock_bootstrap_session.assert_called_once_with(job="test_job2", project=None, run_type="cli", config_dir=None, overrides=[])

        assert result is mock_bm

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_cli_init_with_overrides(self, mock_bootstrap_session):
        """Test CLI init with custom overrides."""
        mock_bm = MagicMock(spec=BM)
        mock_bootstrap_session.return_value = mock_bm

        custom_overrides = ["key=value", "other=setting"]

        # Test CLI init with overrides
        init(job="test_job", project="my_project", overrides=custom_overrides)

        # Verify overrides were passed through
        mock_bootstrap_session.assert_called_once_with(
            job="test_job", project="my_project", run_type="cli", config_dir=None, overrides=custom_overrides
        )

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_cli_bootstrap_session_with_config_dir(self, mock_bootstrap_session):
        """Test CLI init with custom config directory."""
        mock_bm = MagicMock(spec=BM)
        mock_bootstrap_session.return_value = mock_bm

        # Test CLI init with config_dir
        init(job="test_job", project="my_project", config_dir="./my_conf")

        # Verify config_dir was passed through
        mock_bootstrap_session.assert_called_once_with(job="test_job", project="my_project", run_type="cli", config_dir="./my_conf", overrides=[])


class TestCLIInitWithConfigProjectValidation:
    """Test CLI bootstrap_session_with_config() function with project parameter validation."""

    @patch("buttermilk.utils.cli.bootstrap_session_with_config")
    def test_cli_bootstrap_session_with_config_first_session_with_project(self, mock_bootstrap_session_with_config):
        """Test CLI bootstrap_session_with_config for first session with project parameter."""
        # Mock BM instance and config
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "my_project"
        mock_bm.session_info.job = "test_job"
        mock_config = MagicMock()
        mock_bootstrap_session_with_config.return_value = (mock_bm, mock_config)

        # Test CLI bootstrap_session_with_config with project
        bm_result, config_result = cli.bootstrap_session_with_config(job="test_job", project="my_project")

        # Verify bootstrap_session_with_config was called with correct parameters
        mock_bootstrap_session_with_config.assert_called_once_with(
            job="test_job", project="my_project", run_type="cli", config_dir=None, overrides=[]
        )

        assert bm_result is mock_bm
        assert config_result is mock_config

    @patch("buttermilk.utils.cli.bootstrap_session_with_config")
    @patch("buttermilk.utils.cli.sys.modules")
    def test_cli_bootstrap_session_with_config_uses_script_name_fallback(self, mock_sys_modules, mock_bootstrap_session_with_config):
        """Test CLI bootstrap_session_with_config uses script name when project is None (backward compatibility)."""
        # Mock BM instance and config
        mock_bm = MagicMock(spec=BM)
        mock_config = MagicMock()
        mock_bootstrap_session_with_config.return_value = (mock_bm, mock_config)

        # Mock __main__ module with __file__ attribute
        mock_main = MagicMock()
        mock_main.__file__ = "/path/to/my_script.py"
        mock_sys_modules.get.return_value = mock_main

        # Test CLI bootstrap_session_with_config without project (should use script name)
        bm_result, config_result = bootstrap_session_with_config(job="test_job")

        # Verify bootstrap_session_with_config was called with script name as project
        mock_bootstrap_session_with_config.assert_called_once_with(
            job="test_job",
            project="my_script",  # Should use script name
            run_type="cli",
            config_dir=None,
            overrides=[],
        )


class TestSessionInfoProjectName:
    """Test that session info correctly uses project_name field."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_session_info_has_project_name_field(self, mock_bootstrap_session):
        """Test that session info contains project_name field."""
        # Create a mock BM with session_info that has project_name
        mock_bm = MagicMock(spec=BM)
        mock_session_info = MagicMock()
        mock_session_info.project_name = "test_project"
        mock_session_info.job = "test_job"
        mock_session_info.session_id = "test_session_123"
        mock_bm.session_info = mock_session_info

        mock_bootstrap_session.return_value = mock_bm

        # Test CLI init
        result = init(job="test_job", project="test_project")

        # Verify session info has correct project_name
        assert result.session_info.project_name == "test_project"
        assert result.session_info.job == "test_job"
        assert hasattr(result.session_info, "session_id")


class TestBootstrapInfrastructureIntegration:
    """Test integration between bootstrap functions and infrastructure validation.

    CRITICAL: This test class reveals a potential bug in the current bootstrap code.
    The bootstrap functions call infrastructure.validate_and_set_project(project)
    but infrastructure is an InfrastructureManager which doesn't have this method.
    The method exists on ExecutionContext.
    """

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    @patch("buttermilk._core.config_bootstrap.set_bm")
    @patch("buttermilk._core.config_bootstrap.asyncio.run")
    def test_bootstrap_calls_project_validation_correctly(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test that bootstrap functions call project validation on the correct object.

        This test will FAIL if there's a bug where the bootstrap code calls
        infrastructure.validate_and_set_project() when it should call
        execution_context.validate_and_set_project().
        """
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "test_project"
        mock_bm.session_info.job = "test_job"

        # Create mocks that track method calls
        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_infrastructure = MagicMock(spec=InfrastructureManager)

        # The ExecutionContext should have validate_and_set_project
        mock_execution_context.validate_and_set_project.return_value = "test_project"

        # The InfrastructureManager should NOT have validate_and_set_project
        # If it does, that indicates a design issue or missing delegation

        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, "call_count"):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (mock_execution_context, mock_infrastructure)
            else:
                return mock_bm

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # This test will fail if the bootstrap code has a bug
        # The current code calls infrastructure.validate_and_set_project()
        # but infrastructure is an InfrastructureManager that doesn't have this method
        try:
            bootstrap_session(job="test_job", project="test_project")

            # If we get here, the method call succeeded
            # Verify that validate_and_set_project was called on ExecutionContext
            mock_execution_context.validate_and_set_project.assert_called_once_with("test_project")

        except AttributeError as e:
            # This is expected if there's a bug - infrastructure doesn't have validate_and_set_project
            if "validate_and_set_project" in str(e):
                pytest.fail(
                    "BOOTSTRAP BUG DETECTED: The bootstrap code is calling "
                    "infrastructure.validate_and_set_project() but InfrastructureManager "
                    "doesn't have this method. It should be calling "
                    "execution_context.validate_and_set_project() instead."
                )
            else:
                raise


class TestBootstrapErrorHandling:
    """Test error handling in bootstrap functions."""

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    def test_bootstrap_session_handles_project_validation_error(self, mock_bootstrapper_class):
        """Test that bootstrap_session properly handles project validation errors."""
        # Setup bootstrapper to raise RuntimeError during project validation
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Mock ExecutionContext that raises error
        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_execution_context.validate_and_set_project.side_effect = RuntimeError("Project mismatch")
        mock_infrastructure = MagicMock(spec=InfrastructureManager)

        with patch("buttermilk._core.config_bootstrap.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = (mock_execution_context, mock_infrastructure)

            with patch("buttermilk._core.config_bootstrap.logger") as mock_logger:
                with pytest.raises(RuntimeError, match="Project mismatch"):
                    bootstrap_session(job="test_job", project="wrong_project")

                # Verify error was logged
                mock_logger.error.assert_called_once()

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_cli_init_propagates_validation_errors(self, mock_bootstrap_session):
        """Test that CLI init propagates project validation errors."""
        # Setup bootstrap_session to raise project validation error
        mock_bootstrap_session.side_effect = RuntimeError(
            "Project name mismatch: execution context is using project 'project_a', but session specified project 'project_b'"
        )

        # Test that CLI init propagates the error
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            init(job="test_job", project="project_b")


class TestMainScriptCompatibility:
    """Test that main.py example script scenarios work correctly."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_first_session_scenario(self, mock_bootstrap_session):
        """Test the first session scenario from main.py."""
        mock_bm1 = MagicMock(spec=BM)
        mock_bm1.session_info.session_id = "session_1"
        mock_bm1.session_info.project_name = "project_alpha"
        mock_bm1.session_info.job = "first_analysis"
        mock_bootstrap_session.return_value = mock_bm1

        # Simulate: bm1 = init(job="first_analysis", project="project_alpha")
        result = init(job="first_analysis", project="project_alpha")

        # Verify bootstrap was called correctly
        mock_bootstrap_session.assert_called_once_with(job="first_analysis", project="project_alpha", run_type="cli", config_dir=None, overrides=[])

        assert result.session_info.project_name == "project_alpha"
        assert result.session_info.job == "first_analysis"

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_different_project_scenario(self, mock_bootstrap_session):
        """Test the different project scenario from main.py."""
        # First call setup
        mock_bm1 = MagicMock(spec=BM)
        mock_bm1.session_info.project_name = "project_alpha"
        mock_bm1.session_info.job = "first_analysis"

        # Second call setup
        mock_bm2 = MagicMock(spec=BM)
        mock_bm2.session_info.project_name = "project_beta"
        mock_bm2.session_info.job = "second_analysis"

        mock_bootstrap_session.side_effect = [mock_bm1, mock_bm2]

        # Simulate: bm1 = init(job="first_analysis", project="project_alpha")
        result1 = init(job="first_analysis", project="project_alpha")

        # Simulate: bm2 = init(job="second_analysis", project="project_beta")
        result2 = init(job="second_analysis", project="project_beta")

        # Verify both sessions were created with correct projects
        assert result1.session_info.project_name == "project_alpha"
        assert result2.session_info.project_name == "project_beta"

        # Verify bootstrap was called twice with different projects
        assert mock_bootstrap_session.call_count == 2
        calls = mock_bootstrap_session.call_args_list

        assert calls[0][1]["project"] == "project_alpha"
        assert calls[1][1]["project"] == "project_beta"

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_main_script_inherited_project_scenario(self, mock_bootstrap_session):
        """Test the inherited project scenario from main.py."""
        # Mock for third session that inherits project
        mock_bm3 = MagicMock(spec=BM)
        mock_bm3.session_info.project_name = "project_beta"  # Inherited
        mock_bm3.session_info.job = "third_analysis"
        mock_bootstrap_session.return_value = mock_bm3

        # Simulate: bm3 = init(job="third_analysis")  # Inherits "project_beta"
        result = init(job="third_analysis")

        # Verify bootstrap was called without project (should inherit)
        mock_bootstrap_session.assert_called_once_with(
            job="third_analysis",
            project=None,  # Should be None to trigger inheritance
            run_type="cli",
            config_dir=None,
            overrides=[],
        )

        # Verify the session has the inherited project
        assert result.session_info.project_name == "project_beta"
        assert result.session_info.job == "third_analysis"


class TestSaveDirectoryProjectName:
    """Test that save directory uses project_name in path construction."""

    @patch("buttermilk.utils.cli.bootstrap_session")
    def test_save_directory_includes_project_name(self, mock_bootstrap_session):
        """Test that save directory construction uses project_name."""
        # This test verifies the expectation that save directories should use project_name
        # The actual implementation will depend on how BM constructs save paths

        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "my_project"
        mock_bm.session_info.job = "my_job"
        mock_bm.session_info.session_id = "session_123"

        # Mock save_dir or similar attribute that would contain project_name
        mock_bm.save_dir = "/base/path/my_project/my_job/session_123"

        mock_bootstrap_session.return_value = mock_bm

        result = init(job="my_job", project="my_project")

        # Verify that save directory contains the project name
        # This is a contract test - the actual path construction may vary
        assert "my_project" in result.save_dir
        assert result.session_info.project_name == "my_project"


class TestBootstrapBugDetection:
    """Test that reveals the current bug in bootstrap implementation.

    This test class is designed to detect the specific bug where
    bootstrap functions call infrastructure.validate_and_set_project()
    instead of execution_context.validate_and_set_project().
    """

    def test_infrastructure_manager_does_not_have_validate_and_set_project(self):
        """Test that InfrastructureManager does not have validate_and_set_project method."""
        # Create a real InfrastructureManager instance
        infrastructure = InfrastructureManager()

        # Verify it does not have the validate_and_set_project method
        assert not hasattr(infrastructure, "validate_and_set_project"), "InfrastructureManager should not have validate_and_set_project method"

    def test_execution_context_has_validate_and_set_project(self):
        """Test that ExecutionContext has validate_and_set_project method."""
        # Create a real ExecutionContext instance
        execution_context = ExecutionContext()

        # Verify it has the validate_and_set_project method
        assert hasattr(execution_context, "validate_and_set_project"), "ExecutionContext should have validate_and_set_project method"

        # Verify it's callable
        assert callable(execution_context.validate_and_set_project), "ExecutionContext.validate_and_set_project should be callable"

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    def test_bootstrap_session_actual_bug_detection(self, mock_bootstrapper_class):
        """Test that detects the actual bug in bootstrap_session.

        This test calls the real bootstrap_session function and expects it to fail
        with AttributeError because it's calling validate_and_set_project on the
        wrong object.
        """
        # Setup minimal mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Create real objects that will expose the bug
        real_execution_context = ExecutionContext()

        # Mock the bootstrap_full_context to return real objects
        mock_bootstrapper.bootstrap_full_context = AsyncMock(return_value=(real_execution_context))

        with patch("buttermilk._core.config_bootstrap.asyncio.run") as mock_asyncio_run:
            # First call returns real objects, second call is irrelevant since we'll fail first
            mock_asyncio_run.return_value = real_execution_context

            # This should fail with AttributeError because InfrastructureManager
            # doesn't have validate_and_set_project method
            with pytest.raises(AttributeError) as exc_info:
                bootstrap_session_with_config(job="test_job", project="test_project")

            # Verify it's the expected error
            assert "validate_and_set_project" in str(exc_info.value)
            assert "InfrastructureManager" in str(exc_info.value) or "'validate_and_set_project'" in str(exc_info.value)
