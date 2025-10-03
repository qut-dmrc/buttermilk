"""Tests for the unified bootstrap API and project validation logic.

These tests verify that the new CLI init() function works correctly with project parameter
validation, ExecutionContext project inheritance, and the unified bootstrap functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.bm_init import BM
from buttermilk._core.config_bootstrap import bootstrap_session_with_config
from buttermilk._core.execution_context import ExecutionContext


@pytest.mark.anyio
class TestUnifiedBootstrapAPI:
    """Test the unified bootstrap API functions."""

    async def test_bootstrap_session_with_config_function_exists_and_callable(self):
        """Test that the bootstrap_session_with_config function exists and is callable."""
        assert callable(bootstrap_session_with_config)

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    @patch("buttermilk._core.config_bootstrap.set_bm")
    async def test_bootstrap_session_basic_functionality(self, mock_set_bm, mock_bootstrapper_class):
        """Test basic bootstrap_session functionality."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.project_name = "test_project"
        mock_bm.session_info.job = "test_job"

        mock_execution_context = MagicMock(spec=ExecutionContext)
        mock_execution_context.validate_and_set_project.return_value = "test_project"

        mock_bootstrapper.bootstrap_full_context = AsyncMock(return_value=mock_execution_context)
        mock_bootstrapper.bootstrap_session_context = AsyncMock(return_value=mock_bm)

        # Test bootstrap_session
        bm_result, _ = await bootstrap_session_with_config(job="test_job", project="test_project", run_type="cli")

        # Verify bootstrap methods were called
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once()

        # Verify project validation was called
        mock_execution_context.validate_and_set_project.assert_called_once_with("test_project")

        # Verify set_bm was called
        mock_set_bm.assert_called_once_with(mock_bm)

        # Verify BM instance is returned
        assert bm_result is mock_bm

    @patch("buttermilk._core.config_bootstrap.ConfigurationBootstrapper")
    @patch("buttermilk._core.config_bootstrap.set_bm")
    async def test_bootstrap_session_with_config_returns_tuple(self, mock_set_bm, mock_bootstrapper_class):
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

        mock_bootstrapper.bootstrap_full_context = AsyncMock(return_value=mock_execution_context)
        mock_bootstrapper.bootstrap_session_context = AsyncMock(return_value=mock_bm)

        # Test bootstrap_session_with_config
        bm_result, config_result = await bootstrap_session_with_config(job="test_job", project="test_project", run_type="cli")

        # Verify both BM instance and config are returned
        assert bm_result is mock_bm
        assert config_result is mock_config

        # Verify get_configuration was called
        mock_bootstrapper.get_configuration.assert_called_once()
