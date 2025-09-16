"""Tests for initialization utilities integration with ConfigurationBootstrapper.

These tests verify that the new simplified initialization functions properly
integrate with the underlying ConfigurationBootstrapper architecture.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk import BM, init


class TestBootstrapperIntegration:
    """Test that utils properly use ConfigurationBootstrapper architecture."""

    @patch("buttermilk.utils.nb.ConfigurationBootstrapper")
    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_both_utils_use_same_bootstrapper_class(self, mock_cli_bootstrap_class, mock_nb_bootstrap_class):
        """Test that both nb and cli utils use the same ConfigurationBootstrapper class."""
        # Setup mocks to prevent actual execution
        mock_cli_bootstrap = MagicMock()
        mock_nb_bootstrap = MagicMock()
        mock_cli_bootstrap_class.return_value = mock_cli_bootstrap
        mock_nb_bootstrap_class.return_value = mock_nb_bootstrap

        mock_bm = MagicMock(spec=BM)
        mock_infrastructure = MagicMock()
        mock_config = MagicMock()

        # Setup async mocks
        mock_cli_bootstrap.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), mock_infrastructure))
        mock_cli_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)
        mock_nb_bootstrap.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), mock_infrastructure))
        mock_nb_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)
        mock_nb_bootstrap.get_configuration.return_value = mock_config

        with patch("buttermilk.utils.cli.set_bm"), patch("buttermilk.utils.nb.set_bm"):
            with patch("buttermilk.utils.cli.asyncio.run") as mock_cli_run:
                with patch("buttermilk.utils.nb.asyncio.run") as mock_nb_run:
                    # Setup asyncio.run mocks
                    def cli_run_side_effect(coro):
                        if not hasattr(cli_run_side_effect, "call_count"):
                            cli_run_side_effect.call_count = 0
                        cli_run_side_effect.call_count += 1
                        if cli_run_side_effect.call_count == 1:
                            return (MagicMock(), mock_infrastructure)
                        return mock_bm

                    def nb_run_side_effect(coro):
                        if not hasattr(nb_run_side_effect, "call_count"):
                            nb_run_side_effect.call_count = 0
                        nb_run_side_effect.call_count += 1
                        if nb_run_side_effect.call_count == 1:
                            return (MagicMock(), mock_infrastructure)
                        return mock_bm

                    mock_cli_run.side_effect = cli_run_side_effect
                    mock_nb_run.side_effect = nb_run_side_effect

                    # Test both initialization methods
                    init(job="test_job", path="/test/path")
                    nb.nb_init(job="test_job", path="/test/path")

                    # Verify both called ConfigurationBootstrapper
                    mock_cli_bootstrap_class.assert_called_once()
                    mock_nb_bootstrap_class.assert_called_once()

                    # Verify they both called the bootstrap methods
                    mock_cli_bootstrap.bootstrap_full_context.assert_called_once()
                    mock_cli_bootstrap.bootstrap_session_context.assert_called_once()
                    mock_nb_bootstrap.bootstrap_full_context.assert_called_once()
                    mock_nb_bootstrap.bootstrap_session_context.assert_called_once()

    @patch("buttermilk.utils.nb.ConfigurationBootstrapper")
    def test_nb_init_proper_bootstrap_sequence(self, mock_bootstrap_class):
        """Test that nb_init follows the proper bootstrap sequence."""
        mock_bootstrap = MagicMock()
        mock_bootstrap_class.return_value = mock_bootstrap
        
        mock_execution_context = MagicMock()
        mock_infrastructure = MagicMock()
        mock_bm = MagicMock(spec=BM)
        mock_config = MagicMock()
        
        # Setup method return values
        mock_bootstrap.bootstrap_full_context = AsyncMock(return_value=(mock_execution_context, mock_infrastructure))
        mock_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)
        mock_bootstrap.get_configuration.return_value = mock_config

        with patch("buttermilk.utils.nb.set_bm") as mock_set_bm:
            with patch("buttermilk.utils.nb.asyncio.run") as mock_run:

                def run_side_effect(coro):
                    if not hasattr(run_side_effect, "call_count"):
                        run_side_effect.call_count = 0
                    run_side_effect.call_count += 1
                    if run_side_effect.call_count == 1:
                        return (mock_execution_context, mock_infrastructure)
                    return mock_bm

                mock_run.side_effect = run_side_effect

                # Call nb_init
                result = nb.nb_init(job="test_job", name="test_session")

                # Verify bootstrap sequence
                assert mock_run.call_count == 2
                mock_bootstrap.bootstrap_full_context.assert_called_once()
                mock_bootstrap.bootstrap_session_context.assert_called_once()

                # Verify session_context was called with infrastructure from full_context
                session_call_kwargs = mock_bootstrap.bootstrap_session_context.call_args[1]
                assert session_call_kwargs["infrastructure"] is mock_infrastructure
                assert session_call_kwargs["name"] == "test_session"
                assert session_call_kwargs["job"] == "test_job"

                # Verify set_bm was called with the session BM
                mock_set_bm.assert_called_once_with(mock_bm)

                # Verify return structure
                assert hasattr(result, "bm")
                assert hasattr(result, "config")
                assert result.bm is mock_bm
                assert result.config is mock_config

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_cli_init_proper_bootstrap_sequence(self, mock_bootstrap_class):
        """Test that init follows the proper bootstrap sequence."""
        mock_bootstrap = MagicMock()
        mock_bootstrap_class.return_value = mock_bootstrap

        mock_execution_context = MagicMock()
        mock_infrastructure = MagicMock()
        mock_bm = MagicMock(spec=BM)

        # Setup method return values
        mock_bootstrap.bootstrap_full_context = AsyncMock(return_value=(mock_execution_context, mock_infrastructure))
        mock_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)

        with patch("buttermilk.utils.cli.set_bm") as mock_set_bm:
            with patch("buttermilk.utils.cli.asyncio.run") as mock_run:

                def run_side_effect(coro):
                    if not hasattr(run_side_effect, "call_count"):
                        run_side_effect.call_count = 0
                    run_side_effect.call_count += 1
                    if run_side_effect.call_count == 1:
                        return (mock_execution_context, mock_infrastructure)
                    return mock_bm

                mock_run.side_effect = run_side_effect

                # Call init
                result = init(job="test_job", name="test_session")

                # Verify bootstrap sequence
                assert mock_run.call_count == 2
                mock_bootstrap.bootstrap_full_context.assert_called_once()
                mock_bootstrap.bootstrap_session_context.assert_called_once()

                # Verify session_context was called with infrastructure from full_context
                session_call_kwargs = mock_bootstrap.bootstrap_session_context.call_args[1]
                assert session_call_kwargs["infrastructure"] is mock_infrastructure
                assert session_call_kwargs["name"] == "test_session"
                assert session_call_kwargs["job"] == "test_job"

                # Verify set_bm was called with the session BM
                mock_set_bm.assert_called_once_with(mock_bm)

                # Verify BM instance is returned directly
                assert result is mock_bm

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_cli_bootstrap_session_with_config_proper_bootstrap_sequence(self, mock_bootstrap_class):
        """Test that cli.bootstrap_session_with_config follows proper bootstrap sequence and returns both BM and config."""
        mock_bootstrap = MagicMock()
        mock_bootstrap_class.return_value = mock_bootstrap

        mock_execution_context = MagicMock()
        mock_infrastructure = MagicMock()
        mock_bm = MagicMock(spec=BM)
        mock_config = MagicMock()

        # Setup method return values
        mock_bootstrap.bootstrap_full_context = AsyncMock(return_value=(mock_execution_context, mock_infrastructure))
        mock_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)
        mock_bootstrap.get_configuration.return_value = mock_config

        with patch("buttermilk.utils.cli.set_bm") as mock_set_bm:
            with patch("buttermilk.utils.cli.asyncio.run") as mock_run:

                def run_side_effect(coro):
                    if not hasattr(run_side_effect, "call_count"):
                        run_side_effect.call_count = 0
                    run_side_effect.call_count += 1
                    if run_side_effect.call_count == 1:
                        return (mock_execution_context, mock_infrastructure)
                    return mock_bm

                mock_run.side_effect = run_side_effect

                # Call cli.bootstrap_session_with_config
                bm_result, config_result = cli.bootstrap_session_with_config(job="test_job", name="test_session")
                
                # Verify bootstrap sequence
                mock_bootstrap.bootstrap_full_context.assert_called_once()
                mock_bootstrap.bootstrap_session_context.assert_called_once()
                mock_bootstrap.get_configuration.assert_called_once()
                
                # Verify session_context was called with infrastructure from full_context
                session_call_kwargs = mock_bootstrap.bootstrap_session_context.call_args[1]
                assert session_call_kwargs["infrastructure"] is mock_infrastructure
                
                # Verify set_bm was called
                mock_set_bm.assert_called_once_with(mock_bm)
                
                # Verify both BM and config are returned
                assert bm_result is mock_bm
                assert config_result is mock_config
    
    def test_bootstrap_architecture_consistency(self):
        """Test that the utils maintain consistency with the bootstrap architecture."""
        # This test verifies that the utils follow the documented bootstrap pattern:
        # 1. Create ConfigurationBootstrapper
        # 2. Call bootstrap_full_context() to get ExecutionContext + Infrastructure
        # 3. Call bootstrap_session_context() with the Infrastructure to get BM instance
        # 4. Set the BM instance as global singleton

        with patch("buttermilk.utils.nb.ConfigurationBootstrapper") as mock_nb_bootstrap_class:
            with patch("buttermilk.utils.cli.ConfigurationBootstrapper") as mock_cli_bootstrap_class:
                # Setup mocks
                mock_nb_bootstrap = MagicMock()
                mock_cli_bootstrap = MagicMock()
                mock_nb_bootstrap_class.return_value = mock_nb_bootstrap
                mock_cli_bootstrap_class.return_value = mock_cli_bootstrap

                mock_infrastructure = MagicMock()
                mock_bm = MagicMock(spec=BM)

                # Setup async methods
                mock_nb_bootstrap.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), mock_infrastructure))
                mock_nb_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)
                mock_nb_bootstrap.get_configuration.return_value = MagicMock()

                mock_cli_bootstrap.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), mock_infrastructure))
                mock_cli_bootstrap.bootstrap_session_context = AsyncMock(return_value=mock_bm)

                with patch("buttermilk.utils.nb.set_bm"), patch("buttermilk.utils.cli.set_bm"):
                    with patch("buttermilk.utils.nb.asyncio.run") as mock_nb_run:
                        with patch("buttermilk.utils.cli.asyncio.run") as mock_cli_run:
                            # Setup asyncio.run behavior
                            def setup_run_mock(run_mock, bootstrap_mock):
                                def run_side_effect(coro):
                                    if not hasattr(run_side_effect, "call_count"):
                                        run_side_effect.call_count = 0
                                    run_side_effect.call_count += 1
                                    if run_side_effect.call_count == 1:
                                        return (MagicMock(), mock_infrastructure)
                                    return mock_bm

                                run_mock.side_effect = run_side_effect

                            setup_run_mock(mock_nb_run, mock_nb_bootstrap)
                            setup_run_mock(mock_cli_run, mock_cli_bootstrap)

                            # Test both initialization paths
                            nb.nb_init(job="test_job")
                            init(job="test_job")

                            # Verify that both follow the correct sequence:
                            # 1. ConfigurationBootstrapper creation
                            mock_nb_bootstrap_class.assert_called_once()
                            mock_cli_bootstrap_class.assert_called_once()

                            # 2. bootstrap_full_context called first
                            mock_nb_bootstrap.bootstrap_full_context.assert_called_once()
                            mock_cli_bootstrap.bootstrap_full_context.assert_called_once()

                            # 3. bootstrap_session_context called with infrastructure
                            mock_nb_bootstrap.bootstrap_session_context.assert_called_once()
                            mock_cli_bootstrap.bootstrap_session_context.assert_called_once()

                            # Verify infrastructure was passed to session context
                            nb_session_call = mock_nb_bootstrap.bootstrap_session_context.call_args[1]
                            cli_session_call = mock_cli_bootstrap.bootstrap_session_context.call_args[1]

                            assert nb_session_call["infrastructure"] is mock_infrastructure
                            assert cli_session_call["infrastructure"] is mock_infrastructure


class TestBootstrapperErrorPropagation:
    """Test that errors from ConfigurationBootstrapper are properly propagated."""

    @patch("buttermilk.utils.nb.ConfigurationBootstrapper")
    def test_nb_init_propagates_bootstrap_errors(self, mock_bootstrap_class):
        """Test that nb_init properly propagates bootstrap errors."""
        mock_bootstrap = MagicMock()
        mock_bootstrap_class.return_value = mock_bootstrap

        # Setup bootstrap to raise exception
        mock_bootstrap.bootstrap_full_context = AsyncMock(side_effect=Exception("Bootstrap failed"))

        with patch("buttermilk.utils.nb.asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Bootstrap failed")

            with patch("buttermilk.utils.nb.logger") as mock_logger:
                with pytest.raises(Exception, match="Bootstrap failed"):
                    nb.nb_init(job="test_job")

                # Verify error was logged
                mock_logger.error.assert_called_once()

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_cli_init_propagates_bootstrap_errors(self, mock_bootstrap_class):
        """Test that init properly propagates bootstrap errors."""
        mock_bootstrap = MagicMock()
        mock_bootstrap_class.return_value = mock_bootstrap

        # Setup bootstrap to raise exception
        mock_bootstrap.bootstrap_session_context = AsyncMock(side_effect=Exception("Session bootstrap failed"))

        with patch("buttermilk.utils.cli.asyncio.run") as mock_run:

            def run_side_effect(coro):
                if not hasattr(run_side_effect, "call_count"):
                    run_side_effect.call_count = 0
                run_side_effect.call_count += 1
                if run_side_effect.call_count == 1:
                    return (MagicMock(), MagicMock())  # First call succeeds
                else:
                    raise Exception("Session bootstrap failed")  # Second call fails

            mock_run.side_effect = run_side_effect

            with patch("buttermilk.utils.cli.logger") as mock_logger:
                with pytest.raises(Exception, match="Session bootstrap failed"):
                    init(job="test_job")
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
