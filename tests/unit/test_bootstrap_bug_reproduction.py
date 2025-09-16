"""Test to reproduce and fix the bootstrap bug.

This test reproduces the specific bug where bootstrap functions call
infrastructure.validate_and_set_project() instead of
execution_context.validate_and_set_project().
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from buttermilk._core.config_bootstrap import bootstrap_session
from buttermilk._core.execution_context import ExecutionContext
from buttermilk._core.infrastructure import InfrastructureManager


class TestBootstrapBugReproduction:
    """Test that reproduces the current bootstrap bug."""

    def test_infrastructure_manager_missing_validate_and_set_project(self):
        """Verify that InfrastructureManager doesn't have validate_and_set_project method."""
        infrastructure = InfrastructureManager()

        # This should be False - InfrastructureManager should not have this method
        assert not hasattr(infrastructure, 'validate_and_set_project')

    def test_execution_context_has_validate_and_set_project(self):
        """Verify that ExecutionContext has validate_and_set_project method."""
        execution_context = ExecutionContext()

        # This should be True - ExecutionContext should have this method
        assert hasattr(execution_context, 'validate_and_set_project')
        assert callable(execution_context.validate_and_set_project)

    @patch('buttermilk._core.config_bootstrap.ConfigurationBootstrapper')
    @patch('buttermilk._core.config_bootstrap.logger')
    def test_bootstrap_session_bug_reproduction(self, mock_logger, mock_bootstrapper_class):
        """Test that reproduces the actual bug in bootstrap_session.

        This test demonstrates that the current bootstrap_session code fails
        because it calls infrastructure.validate_and_set_project() but
        infrastructure is an InfrastructureManager which doesn't have this method.
        """
        # Setup minimal mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Create REAL objects to expose the bug
        real_execution_context = ExecutionContext()
        real_infrastructure = InfrastructureManager()

        # Verify our setup is correct
        assert hasattr(real_execution_context, 'validate_and_set_project')
        assert not hasattr(real_infrastructure, 'validate_and_set_project')

        with patch('buttermilk._core.config_bootstrap.asyncio.run') as mock_asyncio_run:
            # bootstrap_full_context returns (execution_context, infrastructure)
            mock_asyncio_run.return_value = (real_execution_context, real_infrastructure)

            # This should fail with AttributeError because InfrastructureManager
            # doesn't have validate_and_set_project method
            with pytest.raises(AttributeError) as exc_info:
                bootstrap_session(job="test_job", project="test_project")

            # Verify it's the expected error about the missing method
            error_message = str(exc_info.value)
            assert "validate_and_set_project" in error_message
            assert ("InfrastructureManager" in error_message or
                   "has no attribute" in error_message)

    def test_bug_reproduction_with_detailed_error_analysis(self):
        """Detailed analysis of what goes wrong in the current bootstrap code."""
        # The bootstrap_session function has this flow:
        # 1. _, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        # 2. validated_project = infrastructure.validate_and_set_project(project)

        # Let's simulate step 1:
        execution_context = ExecutionContext()
        infrastructure_manager = InfrastructureManager()

        # bootstrap_full_context returns (ExecutionContext, InfrastructureManager)
        bootstrap_result = (execution_context, infrastructure_manager)

        # Step 2 tries to do this:
        _, infrastructure = bootstrap_result

        # This line in the bootstrap code will fail:
        with pytest.raises(AttributeError):
            infrastructure.validate_and_set_project("test_project")

        # But this would work (the correct approach):
        execution_context.validate_and_set_project("test_project")

        # The fix is to change the bootstrap code to use execution_context instead of infrastructure