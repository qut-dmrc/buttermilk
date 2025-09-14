"""Test to verify the bootstrap bug fix.

This test verifies that the bootstrap functions now correctly call
execution_context.validate_and_set_project() instead of the incorrect
infrastructure.validate_and_set_project().
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from buttermilk._core.config_bootstrap import bootstrap_session, bootstrap_session_with_config
from buttermilk._core.execution_context import ExecutionContext
from buttermilk._core.infrastructure import InfrastructureManager


class TestBootstrapBugFixed:
    """Test that verifies the bootstrap bug is fixed."""

    @patch('buttermilk._core.config_bootstrap.ConfigurationBootstrapper')
    @patch('buttermilk._core.config_bootstrap.set_bm')
    @patch('buttermilk._core.config_bootstrap.asyncio.run')
    def test_bootstrap_session_calls_execution_context_validation(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test that bootstrap_session now correctly calls execution_context.validate_and_set_project."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Create mocks that track method calls
        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_infrastructure = MagicMock(spec=InfrastructureManager)
        mock_bm = MagicMock()
        mock_bm.session_info.project_name = "test_project"
        mock_bm.session_info.job = "test_job"

        # Setup the validation method
        mock_execution_context.validate_and_set_project.return_value = "test_project"

        # Setup asyncio.run side effects
        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, 'call_count'):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (mock_execution_context, mock_infrastructure)  # bootstrap_full_context
            else:
                return mock_bm  # bootstrap_session_context

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test bootstrap_session
        result = bootstrap_session(job="test_job", project="test_project")

        # CRITICAL: Verify that validate_and_set_project was called on ExecutionContext
        mock_execution_context.validate_and_set_project.assert_called_once_with("test_project")

        # Verify infrastructure does NOT have validate_and_set_project called
        # (it shouldn't have this method and the code should not be calling it)
        assert not hasattr(mock_infrastructure, 'validate_and_set_project') or \
               not mock_infrastructure.validate_and_set_project.called

        assert result is mock_bm

    @patch('buttermilk._core.config_bootstrap.ConfigurationBootstrapper')
    @patch('buttermilk._core.config_bootstrap.set_bm')
    @patch('buttermilk._core.config_bootstrap.asyncio.run')
    def test_bootstrap_session_with_config_calls_execution_context_validation(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test that bootstrap_session_with_config now correctly calls execution_context.validate_and_set_project."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_infrastructure = MagicMock(spec=InfrastructureManager)
        mock_bm = MagicMock()
        mock_bm.session_info.project_name = "test_project"
        mock_bm.session_info.job = "test_job"
        mock_config = MagicMock()

        # Setup the validation method and config
        mock_execution_context.validate_and_set_project.return_value = "test_project"
        mock_bootstrapper.get_configuration.return_value = mock_config

        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, 'call_count'):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (mock_execution_context, mock_infrastructure)
            else:
                return mock_bm

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test bootstrap_session_with_config
        bm_result, config_result = bootstrap_session_with_config(job="test_job", project="test_project")

        # CRITICAL: Verify that validate_and_set_project was called on ExecutionContext
        mock_execution_context.validate_and_set_project.assert_called_once_with("test_project")

        # Verify results
        assert bm_result is mock_bm
        assert config_result is mock_config

    @patch('buttermilk._core.config_bootstrap.ConfigurationBootstrapper')
    @patch('buttermilk._core.config_bootstrap.logger')
    def test_bootstrap_session_with_real_objects_no_longer_fails(self, mock_logger, mock_bootstrapper_class):
        """Test that bootstrap_session with real objects no longer fails due to the bug."""
        # Setup minimal mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Create REAL objects - this should no longer cause an AttributeError
        real_execution_context = ExecutionContext()
        real_infrastructure = InfrastructureManager()
        mock_bm = MagicMock()

        # Verify our setup is correct
        assert hasattr(real_execution_context, 'validate_and_set_project')
        assert not hasattr(real_infrastructure, 'validate_and_set_project')

        with patch('buttermilk._core.config_bootstrap.asyncio.run') as mock_asyncio_run:
            def asyncio_run_side_effect(coro):
                if not hasattr(asyncio_run_side_effect, 'call_count'):
                    asyncio_run_side_effect.call_count = 0
                asyncio_run_side_effect.call_count += 1

                if asyncio_run_side_effect.call_count == 1:
                    return (real_execution_context, real_infrastructure)
                else:
                    return mock_bm

            mock_asyncio_run.side_effect = asyncio_run_side_effect

            with patch('buttermilk._core.config_bootstrap.set_bm'):
                # This should NOT raise AttributeError anymore
                result = bootstrap_session(job="test_job", project="test_project")

                # Verify that the real execution context now has the project set
                assert real_execution_context.project_name == "test_project"
                assert result is mock_bm

    def test_bug_fixed_verification_execution_context_method_works(self):
        """Verify that the ExecutionContext.validate_and_set_project method works correctly."""
        execution_context = ExecutionContext()

        # Test first session - should set project
        result = execution_context.validate_and_set_project("my_project")
        assert result == "my_project"
        assert execution_context.project_name == "my_project"

        # Test subsequent session - should inherit project
        result2 = execution_context.validate_and_set_project(None)
        assert result2 == "my_project"

        # Test subsequent session with explicit same project - should work
        result3 = execution_context.validate_and_set_project("my_project")
        assert result3 == "my_project"

        # Test subsequent session with different project - should fail
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            execution_context.validate_and_set_project("other_project")