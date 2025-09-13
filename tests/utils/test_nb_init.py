"""Tests for notebook initialization utility functions.

These tests verify the notebook initialization functions in buttermilk.utils.nb
including backwards compatibility and new simplified interfaces.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from buttermilk.utils import nb
import buttermilk.utils.nb as nb_module
from buttermilk._core.bm_init import BM
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper


class TestNbInit:
    """Test the backwards-compatible nb_init() function."""
    
    @patch('buttermilk.utils.nb.ConfigurationBootstrapper')
    @patch('buttermilk.utils.nb.set_bm')
    @patch('buttermilk.utils.nb.asyncio.run')
    def test_nb_init_basic_functionality(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test basic nb_init functionality with mocked dependencies."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        
        # Mock BM instance
        mock_bm = MagicMock(spec=BM)
        mock_bm.session_info.name = "test_session"
        mock_bm.session_info.job = "test_job"
        
        # Mock infrastructure
        mock_infrastructure = MagicMock()
        
        # Mock config
        mock_config = MagicMock()
        mock_bootstrapper.get_configuration.return_value = mock_config
        
        # Setup asyncio.run side effects
        def asyncio_run_side_effect(coro):
            # First call returns infrastructure, second returns bm
            if not hasattr(asyncio_run_side_effect, 'call_count'):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1
            
            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)  # bootstrap_full_context
            else:
                return mock_bm  # bootstrap_session_context
        
        mock_asyncio_run.side_effect = asyncio_run_side_effect
        
        # Test basic call
        result = nb.nb_init(job="test_job", name="test_session")
        
        # Verify bootstrapper was created with correct parameters
        mock_bootstrapper_class.assert_called_once()
        call_kwargs = mock_bootstrapper_class.call_args[1]
        assert 'config_path' in call_kwargs
        assert 'overrides' in call_kwargs
        assert '+run=notebook' in call_kwargs['overrides']
        
        # Verify bootstrap methods were called
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once()
        
        # Verify set_bm was called
        mock_set_bm.assert_called_once_with(mock_bm)
        
        # Verify return structure
        assert hasattr(result, 'bm')
        assert hasattr(result, 'config')
        assert result.bm is mock_bm
        assert result.config is mock_config
    
    @patch('buttermilk.utils.nb.ConfigurationBootstrapper')
    @patch('buttermilk.utils.nb.set_bm')
    @patch('buttermilk.utils.nb.asyncio.run')
    def test_nb_init_backwards_compatibility_name_in_overrides(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test backwards compatibility for name extraction from overrides."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        mock_bm = MagicMock(spec=BM)
        mock_infrastructure = MagicMock()
        
        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, 'call_count'):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1
            
            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm
        
        mock_asyncio_run.side_effect = asyncio_run_side_effect
        
        # Test with name in overrides (old style)
        overrides = ["name=extracted_name", "other=value"]
        result = nb.nb_init(job="test_job", overrides=overrides)
        
        # Verify that session was created with extracted name
        session_call = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call[1]['name'] == "extracted_name"
        
        # Verify overrides were modified (name removed)
        bootstrap_call = mock_bootstrapper_class.call_args[1]
        final_overrides = bootstrap_call['overrides']
        assert "name=extracted_name" not in final_overrides
        assert "other=value" in final_overrides
        assert "+run=notebook" in final_overrides
    
    @patch('buttermilk.utils.nb.ConfigurationBootstrapper')
    @patch('buttermilk.utils.nb.set_bm')
    @patch('buttermilk.utils.nb.asyncio.run')
    def test_nb_init_backwards_compatibility_bm_session_info_name(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test backwards compatibility for bm.session_info.name= format."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        mock_bm = MagicMock(spec=BM)
        mock_infrastructure = MagicMock()
        
        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, 'call_count'):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1
            
            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm
        
        mock_asyncio_run.side_effect = asyncio_run_side_effect
        
        # Test with bm.session_info.name format
        overrides = ["bm.session_info.name=extracted_name", "other=value"]
        result = nb.nb_init(job="test_job", overrides=overrides)
        
        # Verify that session was created with extracted name
        session_call = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call[1]['name'] == "extracted_name"
        
        # Verify overrides were modified
        bootstrap_call = mock_bootstrapper_class.call_args[1]
        final_overrides = bootstrap_call['overrides']
        assert "bm.session_info.name=extracted_name" not in final_overrides
        assert "other=value" in final_overrides
    
    @patch('buttermilk.utils.nb.ConfigurationBootstrapper')
    @patch('buttermilk.utils.nb.set_bm')
    @patch('buttermilk.utils.nb.asyncio.run')
    def test_nb_init_default_name_fallback(self, mock_asyncio_run, mock_set_bm, mock_bootstrapper_class):
        """Test default name fallback when no name provided."""
        # Setup mocks
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        mock_bm = MagicMock(spec=BM)
        mock_infrastructure = MagicMock()
        
        def asyncio_run_side_effect(coro):
            if not hasattr(asyncio_run_side_effect, 'call_count'):
                asyncio_run_side_effect.call_count = 0
            asyncio_run_side_effect.call_count += 1
            
            if asyncio_run_side_effect.call_count == 1:
                return (MagicMock(), mock_infrastructure)
            else:
                return mock_bm
        
        mock_asyncio_run.side_effect = asyncio_run_side_effect
        
        # Test without name parameter
        result = nb.nb_init(job="test_job")
        
        # Verify default name was used
        session_call = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call[1]['name'] == "notebook_session"
    
    @patch('buttermilk.utils.nb.ConfigurationBootstrapper')
    def test_nb_init_path_handling(self, mock_bootstrapper_class):
        """Test path handling for configuration directory."""
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        
        # Mock the asyncio.run calls to prevent actual execution
        with patch('buttermilk.utils.nb.asyncio.run') as mock_asyncio_run:
            mock_asyncio_run.side_effect = [  
                (MagicMock(), MagicMock()),  # bootstrap_full_context
                MagicMock()  # bootstrap_session_context
            ]
            with patch('buttermilk.utils.nb.set_bm'):
                # Test default path
                nb.nb_init(job="test_job")
                
                # Verify default path is used (../../conf from nb.py location)
                call_kwargs = mock_bootstrapper_class.call_args[1]
                config_path = call_kwargs['config_path']
                assert config_path.endswith('/conf')
                
                # Test custom path
                mock_bootstrapper_class.reset_mock()
                custom_path = "/custom/config/path"
                nb.nb_init(job="test_job", path=custom_path)
                
                call_kwargs = mock_bootstrapper_class.call_args[1]
                assert call_kwargs['config_path'] == custom_path
    
    @patch('buttermilk.utils.nb.ConfigurationBootstrapper')
    def test_nb_init_error_handling(self, mock_bootstrapper_class):
        """Test error handling and propagation."""
        # Setup bootstrapper to raise exception
        mock_bootstrapper = MagicMock()
        mock_bootstrapper_class.return_value = mock_bootstrapper
        
        with patch('buttermilk.utils.nb.asyncio.run') as mock_asyncio_run:
            mock_asyncio_run.side_effect = Exception("Bootstrap failed")
            
            with patch('buttermilk.utils.nb.logger') as mock_logger:
                with pytest.raises(Exception, match="Bootstrap failed"):
                    nb.nb_init(job="test_job")
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                assert "Failed to initialize Buttermilk" in str(mock_logger.error.call_args)


class TestNbInitSimple:
    """Test the new simplified init() function."""
    
    @patch('buttermilk.utils.nb.nb_init')
    def test_init_basic_functionality(self, mock_nb_init):
        """Test that init() properly calls nb_init() and returns BM instance."""
        # Setup mock return value
        mock_objs = MagicMock()
        mock_bm = MagicMock(spec=BM)
        mock_objs.bm = mock_bm
        mock_nb_init.return_value = mock_objs
        
        # Test basic call
        result = nb.init(job="test_job", name="test_session")
        
        # Verify nb_init was called with correct parameters
        mock_nb_init.assert_called_once_with(job="test_job", name="test_session")
        
        # Verify BM instance is returned
        assert result is mock_bm
    
    @patch('buttermilk.utils.nb.nb_init')
    def test_init_default_name(self, mock_nb_init):
        """Test that init() uses default name correctly."""
        mock_objs = MagicMock()
        mock_bm = MagicMock(spec=BM)
        mock_objs.bm = mock_bm
        mock_nb_init.return_value = mock_objs
        
        # Test with default name
        result = nb.init(job="test_job")
        
        # Verify default name was passed
        mock_nb_init.assert_called_once_with(job="test_job", name="notebook_session")
        
        assert result is mock_bm
    
    @patch('buttermilk.utils.nb.nb_init')
    def test_init_kwargs_passthrough(self, mock_nb_init):
        """Test that init() passes through additional kwargs to nb_init()."""
        mock_objs = MagicMock()
        mock_bm = MagicMock(spec=BM)
        mock_objs.bm = mock_bm
        mock_nb_init.return_value = mock_objs
        
        # Test with additional kwargs
        overrides = ["key=value"]
        path = "/custom/path"
        result = nb.init(job="test_job", name="custom_name", overrides=overrides, path=path)
        
        # Verify all parameters were passed through
        mock_nb_init.assert_called_once_with(
            job="test_job", 
            name="custom_name", 
            overrides=overrides, 
            path=path
        )
        
        assert result is mock_bm


class TestNbInitIntegration:
    """Integration tests using real configuration."""
    
    def test_nb_init_integration_with_real_config(self, real_conf, tmp_path):
        """Test nb_init with real configuration to verify end-to-end functionality."""
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
  name: integration_test
  job: test_integration
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
        
        # Test nb_init with the test config
        with patch('buttermilk.utils.nb.logger'):
            result = nb.nb_init(
                job="integration_test",
                name="test_session", 
                path=str(config_dir)
            )
        
        # Verify the result structure
        assert hasattr(result, 'bm')
        assert hasattr(result, 'config')
        assert result.bm is not None
        assert result.config is not None
        
        # Verify BM instance properties
        bm_instance = result.bm
        assert hasattr(bm_instance, 'session_info')
        assert bm_instance.session_info.name == "test_session"
        assert bm_instance.session_info.job == "integration_test"
    
    def test_init_integration_with_real_config(self, real_conf, tmp_path):
        """Test simplified init() with real configuration."""
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
  name: simple_test
  job: test_simple
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
        
        # Test simplified init
        with patch('buttermilk.utils.nb.logger'):
            bm_instance = nb.init(
                job="simple_test",
                name="simple_session",
                path=str(config_dir)
            )
        
        # Verify BM instance
        assert bm_instance is not None
        assert hasattr(bm_instance, 'session_info')
        assert bm_instance.session_info.name == "simple_session"
        assert bm_instance.session_info.job == "simple_test"
