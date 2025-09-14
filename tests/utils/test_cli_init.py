"""Tests for CLI initialization utility functions.

These tests verify the CLI initialization functions in buttermilk.utils.cli
including the new simplified interfaces for CLI scripts.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.bm_init import BM, init


class TestCliInit:
    """Test the CLI init() function."""

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    @patch("buttermilk.utils.cli.set_bm")
    @patch("buttermilk.utils.cli.asyncio.run")
    def test_cli_init_basic_functionality(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test basic CLI init functionality with mocked dependencies."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Mock BM instance
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.name = "test_session"
        mock_bm.session_info.job = "test_job"

        # Mock infrastructure
        mock_infrastructure = MagicMock()

        # Setup asyncio.run side effects
        def asyncio_run_side_effect(coro):
            # First call returns infrastructure, second returns bm
            if not hasattr(asyncio_run_side_effect, "call_count"):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)  # bootstrap_full_context
            else:
                return mock_bm  # bootstrap_session_context

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test basic call
        result = init(job="test_job", name="test_session")

        # Verify bootstrapper was created with correct parameters
        mock_bootstrapper_class.assert_called_once()
        call_kwargs = mock_bootstrapper_class.call_args[1]
        assert "config_path" in call_kwargs
        assert "overrides" in call_kwargs
        assert "run=cli" in call_kwargs["overrides"]

        # Verify bootstrap methods were called
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once()

        # Verify set_bm was called
        mock_set_bm.assert_called_once_with(mock_bm)

        # Verify BM instance is returned
        assert result is mock_bm

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    @patch("buttermilk.utils.cli.set_bm")
    @patch("buttermilk.utils.cli.asyncio.run")
    @patch("buttermilk.utils.cli.sys.modules")
    def test_cli_init_name_from_script_filename(self, mock_sys_modules, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test automatic name extraction from script filename."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        mock_bm = MagicMock(spec=BM)
        mock_infrastructure = MagicMock()

        # Mock __main__ module with __file__ attribute
        mock_main = MagicMock()
        mock_main.__file__ = "/path/to/my_script.py"
        mock_sys_modules.get.return_value = mock_main

        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, "call_count"):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test without explicit name
        result = init(job="test_job")

        # Verify that session was created with script name
        session_call = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call[1]["name"] == "my_script"

        assert result is mock_bm

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    @patch("buttermilk.utils.cli.set_bm")
    @patch("buttermilk.utils.cli.asyncio.run")
    @patch("buttermilk.utils.cli.sys.modules")
    def test_cli_init_fallback_name_when_no_script(self, mock_sys_modules, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test fallback name when script filename cannot be determined."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        mock_bm = MagicMock(spec=BM)
        mock_infrastructure = MagicMock()

        # Mock __main__ module without __file__ attribute
        mock_main = MagicMock()
        mock_main.__file__ = None
        mock_sys_modules.get.return_value = mock_main

        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, "call_count"):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test without explicit name and no script file
        result = init(job="test_job")

        # Verify fallback name was used
        session_call = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call[1]["name"] == "cli_script"

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_cli_init_path_handling(self, mock_bootstrapper_class):
        """Test path handling for configuration directory."""
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Mock the asyncio.run calls to prevent actual execution
        with patch("buttermilk.utils.cli.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = [
                (MagicMock(), MagicMock()),  # bootstrap_full_context
                MagicMock(),  # bootstrap_session_context
            ]
            with patch("buttermilk.utils.cli.set_bm"):
                # Test default path (./conf from current working directory)
                with patch("buttermilk.utils.cli.Path.cwd") as mock_cwd:
                    mock_cwd.return_value = Path("/current/working/dir")
                    init(job="test_job")

                    # Verify default path is used
                    call_kwargs = mock_bootstrapper_class.call_args[1]
                    config_path = call_kwargs["config_path"]
                    assert config_path == "/current/working/dir/conf"

                    # Test custom path
                    mock_bootstrapper_class.reset_mock()
                    custom_path = "/custom/config/path"
                    init(job="test_job", path=custom_path)

                    call_kwargs = mock_bootstrapper_class.call_args[1]
                    assert call_kwargs["config_path"] == custom_path

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_cli_init_overrides_handling(self, mock_bootstrapper_class):
        """Test that overrides are properly handled and CLI-specific overrides added."""
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        with patch("buttermilk.utils.cli.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = [(MagicMock(), MagicMock()), MagicMock()]
            with patch("buttermilk.utils.cli.set_bm"):
                # Test with custom overrides
                custom_overrides = ["key=value", "other=setting"]
                init(job="test_job", overrides=custom_overrides)

                # Verify overrides include both custom and CLI-specific
                call_kwargs = mock_bootstrapper_class.call_args[1]
                final_overrides = call_kwargs["overrides"]
                assert "key=value" in final_overrides
                assert "other=setting" in final_overrides
                assert "run=cli" in final_overrides

                # Verify original list was not modified
                assert custom_overrides == ["key=value", "other=setting"]

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_cli_init_error_handling(self, mock_bootstrapper_class):
        """Test error handling and propagation."""
        # Setup bootstrapper to raise exception
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        with patch("buttermilk.utils.cli.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = Exception("Bootstrap failed")

            with patch("buttermilk.utils.cli.logger") as mock_logger:
                with pytest.raises(Exception, match="Bootstrap failed"):
                    init(job="test_job")

                # Verify error was logged
                mock_logger.error.assert_called_once()
                assert "Failed to initialize Buttermilk" in str(mock_logger.error.call_args)


class TestCliInitWithConfig:
    """Test the CLI bootstrap_session_with_config() function."""

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    @patch("buttermilk.utils.cli.set_bm")
    @patch("buttermilk.utils.cli.asyncio.run")
    def test_bootstrap_session_with_config_basic_functionality(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test basic bootstrap_session_with_config functionality."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        # Mock BM instance and config
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.name = "test_session"
        mock_bm.session_info.job = "test_job"
        mock_config = MagicMock()
        mock_bootstrapper.get_configuration.return_value = mock_config

        # Mock infrastructure
        mock_infrastructure = MagicMock()

        # Setup asyncio.run side effects
        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, "call_count"):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test basic call
        bm_result, config_result = cli.bootstrap_session_with_config(job="test_job", name="test_session")

        # Verify bootstrapper was created and called
        mock_bootstrapper_class.assert_called_once()
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once()
        mock_bootstrapper.get_configuration.assert_called_once()

        # Verify set_bm was called
        mock_set_bm.assert_called_once_with(mock_bm)

        # Verify both BM instance and config are returned
        assert bm_result is mock_bm
        assert config_result is mock_config

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    @patch("buttermilk.utils.cli.set_bm")
    @patch("buttermilk.utils.cli.asyncio.run")
    @patch("buttermilk.utils.cli.sys.modules")
    def test_bootstrap_session_with_config_name_from_script(self, mock_sys_modules, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test bootstrap_session_with_config with automatic name extraction."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        mock_bm = MagicMock(spec=BM)
        mock_config = MagicMock()
        mock_bootstrapper.get_configuration.return_value = mock_config
        mock_infrastructure = MagicMock()

        # Mock __main__ module with __file__ attribute
        mock_main = MagicMock()
        mock_main.__file__ = "/path/to/data_processor.py"
        mock_sys_modules.get.return_value = mock_main

        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, "call_count"):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1

            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm

        mock_asyncio_run.side_effect = asyncio_run_side_effect

        # Test without explicit name
        bm_result, config_result = cli.bootstrap_session_with_config(job="test_job")

        # Verify that session was created with script name
        session_call = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call[1]["name"] == "data_processor"

        assert bm_result is mock_bm
        assert config_result is mock_config

    @patch("buttermilk.utils.cli.ConfigurationBootstrapper")
    def test_bootstrap_session_with_config_error_handling(self, mock_bootstrapper_class):
        """Test error handling in bootstrap_session_with_config."""
        # Setup bootstrapper to raise exception
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper

        with patch("buttermilk.utils.cli.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = Exception("Configuration error")

            with patch("buttermilk.utils.cli.logger") as mock_logger:
                with pytest.raises(Exception, match="Configuration error"):
                    cli.bootstrap_session_with_config(job="test_job")

                # Verify error was logged
                mock_logger.error.assert_called_once()
                assert "Failed to initialize Buttermilk" in str(mock_logger.error.call_args)


class TestCliInitIntegration:
    """Integration tests using real configuration."""

    def test_cli_init_integration_with_real_config(self, real_conf, tmp_path):
        """Test CLI init with real configuration to verify end-to-end functionality."""
        # Create a temporary config directory structure for the test
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        # Write a minimal test config file
        config_file = config_dir / "config.yaml"
        config_content = """
defaults:
  - bm: testing
  - _self_

infrastructure:
  logging:
    verbose: false
  
run:
  name: cli_integration_test
  job: test_cli_integration
"""
        config_file.write_text(config_content)

        # Create the bm config directory and file
        bm_dir = config_dir / "bm"
        bm_dir.mkdir()
        bm_config_file = bm_dir / "testing.yaml"
        bm_config_content = """
# @package _global_

llms:
  openai:
    models: []
  anthropic:
    models: []

infrastructure:
  cloud_manager:
    enabled: false
  secret_manager:
    enabled: false
  logging:
    enabled: true
    verbose: false
  tracing:
    enabled: false
"""
        bm_config_file.write_text(bm_config_content)

        # Test CLI init with the test config
        with patch("buttermilk.utils.cli.logger"):
            bm_instance = init(job="cli_integration_test", name="cli_test_session", path=str(config_dir))

        # Verify BM instance
        assert bm_instance is not None
        assert hasattr(bm_instance, "session_info")
        assert bm_instance.session_info.name == "cli_test_session"
        assert bm_instance.session_info.job == "cli_integration_test"

    def test_bootstrap_session_with_config_integration_with_real_config(self, real_conf, tmp_path):
        """Test bootstrap_session_with_config with real configuration."""
        # Create a temporary config directory structure
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        # Write minimal config files
        config_file = config_dir / "config.yaml"
        config_content = """
defaults:
  - bm: testing
  - _self_

infrastructure:
  logging:
    verbose: false

run:
  name: cli_config_test
  job: test_cli_config
"""
        config_file.write_text(config_content)

        bm_dir = config_dir / "bm"
        bm_dir.mkdir()
        bm_config_file = bm_dir / "testing.yaml"
        bm_config_content = """
# @package _global_

llms:
  openai:
    models: []
  anthropic:
    models: []

infrastructure:
  cloud_manager:
    enabled: false
  secret_manager:
    enabled: false
  logging:
    enabled: true
    verbose: false
  tracing:
    enabled: false
"""
        bm_config_file.write_text(bm_config_content)

        # Test bootstrap_session_with_config
        with patch("buttermilk.utils.cli.logger"):
            bm_instance, config = cli.bootstrap_session_with_config(job="cli_config_test", name="cli_config_session", path=str(config_dir))

        # Verify BM instance
        assert bm_instance is not None
        assert hasattr(bm_instance, "session_info")
        assert bm_instance.session_info.name == "cli_config_session"
        assert bm_instance.session_info.job == "cli_config_test"

        # Verify config object
        assert config is not None
        # Config should have infrastructure settings
        assert hasattr(config, "infrastructure")


class TestCliUtilsComparison:
    """Tests comparing CLI utils behavior with notebook utils."""

    def test_path_defaults_difference(self):
        """Test that CLI and notebook utils have different default paths."""
        with patch("buttermilk.utils.cli.ConfigurationBootstrapper") as mock_cli_bootstrap:
            with patch("buttermilk.utils.nb.ConfigurationBootstrapper") as mock_nb_bootstrap:
                with patch("buttermilk.utils.cli.asyncio.run"), patch("buttermilk.utils.nb.asyncio.run"):
                    with patch("buttermilk.utils.cli.set_bm"), patch("buttermilk.utils.nb.set_bm"):
                        with patch("buttermilk.utils.cli.Path.cwd") as mock_cwd:
                            mock_cwd.return_value = Path("/current/dir")

                            # Mock successful execution for both
                            mock_cli_bootstrap.return_value.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), MagicMock()))
                            mock_cli_bootstrap.return_value.bootstrap_session_context = AsyncMock(return_value=MagicMock())
                            mock_nb_bootstrap.return_value.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), MagicMock()))
                            mock_nb_bootstrap.return_value.bootstrap_session_context = AsyncMock(return_value=MagicMock())
                            mock_nb_bootstrap.return_value.get_configuration.return_value = MagicMock()

                            # Test CLI init
                            init(job="test_job")

                            # Import and test notebook init separately to avoid circular import
                            from buttermilk.utils import nb

                            nb.nb_init(job="test_job")

                            # Verify different default paths
                            cli_call = mock_cli_bootstrap.call_args[1]
                            nb_call = mock_nb_bootstrap.call_args[1]

                            # CLI should use ./conf from current working directory
                            assert cli_call["config_path"] == "/current/dir/conf"

                            # Notebook should use ../../conf from nb.py location
                            nb_path = nb_call["config_path"]
                            assert nb_path.endswith("/conf")
                            assert nb_path != "/current/dir/conf"

    def test_run_override_difference(self):
        """Test that CLI and notebook utils add different run overrides."""
        with patch("buttermilk.utils.cli.ConfigurationBootstrapper") as mock_cli_bootstrap:
            with patch("buttermilk.utils.nb.ConfigurationBootstrapper") as mock_nb_bootstrap:
                with patch("buttermilk.utils.cli.asyncio.run"), patch("buttermilk.utils.nb.asyncio.run"):
                    with patch("buttermilk.utils.cli.set_bm"), patch("buttermilk.utils.nb.set_bm"):
                        # Mock successful execution
                        mock_cli_bootstrap.return_value.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), MagicMock()))
                        mock_cli_bootstrap.return_value.bootstrap_session_context = AsyncMock(return_value=MagicMock())
                        mock_nb_bootstrap.return_value.bootstrap_full_context = AsyncMock(return_value=(MagicMock(), MagicMock()))
                        mock_nb_bootstrap.return_value.bootstrap_session_context = AsyncMock(return_value=MagicMock())
                        mock_nb_bootstrap.return_value.get_configuration.return_value = MagicMock()

                        # Test CLI init
                        init(job="test_job", path="/test/path")

                        # Import and test notebook init separately
                        from buttermilk.utils import nb

                        nb.nb_init(job="test_job", path="/test/path")

                        # Verify different run overrides
                        cli_overrides = mock_cli_bootstrap.call_args[1]["overrides"]
                        nb_overrides = mock_nb_bootstrap.call_args[1]["overrides"]

                        assert "run=cli" in cli_overrides
                        assert "run=notebook" in nb_overrides
                        assert "run=cli" not in nb_overrides
                        assert "run=notebook" not in cli_overrides
