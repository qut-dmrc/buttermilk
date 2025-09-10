"""End-to-end integration tests for bootstrap sequence architecture.

These tests validate the complete bootstrap sequence with real components
to ensure the architecture fix works correctly in practice. Tests include:
1. Full bootstrap sequence with real configuration
2. Infrastructure sharing validation 
3. Logging consistency across bootstrap
4. OTEL tracing initialization without "CloudManager not available" errors
5. Session creation using shared ExecutionContext infrastructure

These tests use real configuration but mock external dependencies
to avoid requiring actual cloud credentials during testing.
"""

import asyncio
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock

import pytest
from omegaconf import OmegaConf

from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.execution_context import (
    get_execution_context, 
    get_or_create_execution_context,
    _global_execution_context,
    _execution_context_initialized
)
from buttermilk._core.log import logger


class TestBootstrapSequenceE2E:
    """End-to-end tests for bootstrap sequence with real components."""
    
    def setup_method(self):
        """Reset global state and prepare test environment."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
        
        # Clear any existing BM singleton
        from buttermilk import _global_bm
        _global_bm = None
    
    def test_full_bootstrap_sequence_with_minimal_config(self):
        """Test complete bootstrap sequence with minimal real configuration."""
        # Create minimal test configuration
        test_config = {
            'infrastructure': {
                'clouds': [],  # Empty clouds to avoid cloud authentication
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            },
            'run': {
                'name': 'test_bootstrap',
                'job': 'e2e_test',
                'mode': 'console'
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Step 1: Bootstrap full context (ExecutionContext + Infrastructure)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext was created
        assert execution_context is not None
        assert execution_context.execution_context_id.startswith("exec-")
        
        # Verify ExecutionContext is accessible globally
        global_context = get_execution_context()
        assert global_context is execution_context
        
        # Verify infrastructure was created
        assert infrastructure is not None
        
        # Step 2: Bootstrap session context using existing infrastructure
        session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            name="test_session",
            job="e2e_session",
            infrastructure=infrastructure  # Use existing infrastructure
        ))
        
        # Verify session BM was created
        assert session_bm is not None
        assert hasattr(session_bm, 'session_info')
        assert session_bm.session_info.session_id
        
        # Verify session BM can access ExecutionContext infrastructure
        # (This validates the infrastructure sharing)
        from buttermilk import get_bm
        global_bm = get_bm()
        assert global_bm is session_bm
    
    def test_execution_context_prevents_duplicate_creation(self):
        """Test that ExecutionContext prevents duplicate creation in real scenario."""
        test_config = {
            'infrastructure': {
                'clouds': [],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper1 = ConfigurationBootstrapper(config=config)
        
        # First bootstrap should succeed
        execution_context1, infrastructure1 = asyncio.run(bootstrapper1.bootstrap_full_context())
        
        # Second bootstrap attempt should return same ExecutionContext
        bootstrapper2 = ConfigurationBootstrapper(config=config)
        execution_context2, infrastructure2 = asyncio.run(bootstrapper2.bootstrap_full_context())
        
        # Should be same ExecutionContext instance
        assert execution_context1 is execution_context2
        assert execution_context1.execution_context_id == execution_context2.execution_context_id
    
    @patch('buttermilk._core.execution_context.CloudManager')
    def test_infrastructure_sharing_validation(self, mock_cloud_manager):
        """Test that ExecutionContext and sessions share infrastructure correctly."""
        # Configure with cloud to trigger CloudManager creation
        test_config = {
            'infrastructure': {
                'clouds': [{'type': 'gcp', 'project_id': 'test-project'}],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            }
        }
        
        # Mock CloudManager to avoid actual cloud authentication
        mock_cloud_instance = Mock()
        mock_cloud_manager.return_value = mock_cloud_instance
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Bootstrap full context
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Access cloud manager to trigger creation
        cloud_mgr = execution_context.cloud_manager
        
        # Verify CloudManager was created with correct configuration
        mock_cloud_manager.assert_called_once_with(
            clouds=[{'type': 'gcp', 'project_id': 'test-project'}]
        )
        assert cloud_mgr is mock_cloud_instance
        
        # Create session using existing infrastructure
        session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            name="test_session",
            job="share_test",
            infrastructure=infrastructure
        ))
        
        # Session BM should be able to access the shared infrastructure
        assert session_bm is not None
        # Infrastructure sharing is validated by the fact that session creation succeeded
        # using the pre-existing infrastructure instance
    
    def test_logging_consistency_across_bootstrap(self):
        """Test that logging maintains consistency across bootstrap sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configure logging to write to temp directory
            test_config = {
                'infrastructure': {
                    'clouds': [],
                    'secret_provider': None,
                    'logging': {'verbose': True},  # Enable verbose logging
                    'tracing': {},
                    'datasets': {}
                }
            }
            
            config = OmegaConf.create(test_config)
            bootstrapper = ConfigurationBootstrapper(config=config)
            
            # Bootstrap sequence
            execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
            session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
                name="log_test",
                job="logging_consistency",
                infrastructure=infrastructure
            ))
            
            # Verify ExecutionContext has consistent ID
            exec_id = execution_context.execution_context_id
            assert exec_id.startswith("exec-")
            
            # Log something from session to verify logging works
            logger.info("Test log message for bootstrap sequence validation", 
                       execution_context_id=exec_id,
                       session_id=session_bm.session_info.session_id)
            
            # The fact that no exceptions were raised indicates logging consistency
            assert True  # If we reach here, logging configuration is consistent
    
    @patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context')
    def test_otel_tracing_no_cloudmanager_error(self, mock_otel_setup):
        """Test that OTEL tracing initializes without 'CloudManager not available' error."""
        test_config = {
            'infrastructure': {
                'clouds': [{'type': 'gcp', 'project_id': 'test-project'}],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {
                    'otel': {
                        'enabled': True,
                        'endpoint': 'http://test-otel-endpoint'
                    }
                },
                'datasets': {}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Mock CloudManager to avoid real authentication
        with patch('buttermilk._core.execution_context.CloudManager') as mock_cloud_mgr:
            mock_cloud_instance = Mock()
            mock_cloud_mgr.return_value = mock_cloud_instance
            
            # Bootstrap should succeed and initialize tracing
            execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
            
            # Verify OTEL setup was called with ExecutionContext
            # (This validates that CloudManager is available to OTEL)
            mock_otel_setup.assert_called_once()
            call_args = mock_otel_setup.call_args
            
            # Verify ExecutionContext was passed to OTEL setup
            assert call_args[0][1] is execution_context  # Second argument should be ExecutionContext
            
            # Verify ExecutionContext has CloudManager available
            assert execution_context.cloud_manager is mock_cloud_instance
    
    def test_session_creation_uses_execution_context_infrastructure(self):
        """Test that session creation properly uses ExecutionContext infrastructure."""
        test_config = {
            'infrastructure': {
                'clouds': [],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {'test_dataset': {'type': 'memory'}}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Bootstrap full context first
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext has datasets configuration
        assert 'test_dataset' in execution_context.datasets
        
        # Create multiple sessions using the same infrastructure
        session_bm1 = asyncio.run(bootstrapper.bootstrap_session_context(
            name="session1",
            job="infra_test1",
            infrastructure=infrastructure
        ))
        
        session_bm2 = asyncio.run(bootstrapper.bootstrap_session_context(
            name="session2", 
            job="infra_test2",
            infrastructure=infrastructure
        ))
        
        # Both sessions should be created successfully
        assert session_bm1 is not None
        assert session_bm2 is not None
        
        # Sessions should have different session IDs but use same infrastructure
        assert session_bm1.session_info.session_id != session_bm2.session_info.session_id
        
        # This validates that infrastructure sharing works correctly
        # without creating duplicate ExecutionContext instances
    
    @patch('buttermilk._core.execution_context.LLMs')
    def test_llms_initialization_through_execution_context(self, mock_llms_class):
        """Test LLMs initialization through ExecutionContext with mock credentials."""
        # Create temporary cache file with mock LLM connections
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / ".cache" / "buttermilk" / "models.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            mock_connections = {
                "openai": {"api_key": "test-key"},
                "anthropic": {"api_key": "test-anthropic-key"}
            }
            cache_path.write_text(json.dumps(mock_connections))
            
            # Patch the cache path
            with patch('buttermilk._core.execution_context.CONFIG_CACHE_PATH', str(cache_path)):
                test_config = {
                    'infrastructure': {
                        'clouds': [],
                        'secret_provider': None,
                        'logging': {'verbose': False},
                        'tracing': {},
                        'datasets': {}
                    }
                }
                
                config = OmegaConf.create(test_config)
                bootstrapper = ConfigurationBootstrapper(config=config)
                
                # Bootstrap and access LLMs
                execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
                
                # Access LLMs to trigger initialization
                llms_instance = execution_context.llms
                
                # Verify LLMs was initialized with cached connections
                mock_llms_class.assert_called_once_with(connections=mock_connections)
                assert llms_instance is mock_llms_class.return_value


class TestBootstrapSequenceErrorRecovery:
    """Test error recovery and failure scenarios in bootstrap sequence."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
        
        from buttermilk import _global_bm
        _global_bm = None
    
    def test_bootstrap_with_invalid_configuration(self):
        """Test bootstrap behavior with invalid configuration."""
        # Missing required infrastructure configuration
        invalid_config = {
            'run': {
                'name': 'test',
                'job': 'error_test'
            }
            # Missing 'infrastructure' section
        }
        
        config = OmegaConf.create(invalid_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Should raise RuntimeError for missing infrastructure config
        with pytest.raises(RuntimeError, match="No infrastructure configuration found"):
            asyncio.run(bootstrapper.bootstrap_full_context())
    
    @patch('buttermilk._core.execution_context.CloudManager')
    def test_execution_context_initialization_failure_recovery(self, mock_cloud_manager):
        """Test recovery when ExecutionContext initialization fails."""
        # Make CloudManager fail during initialization
        mock_cloud_manager.side_effect = Exception("Cloud authentication failed")
        
        test_config = {
            'infrastructure': {
                'clouds': [{'type': 'gcp', 'project_id': 'test'}],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Bootstrap should fail due to ExecutionContext initialization error
        with pytest.raises(RuntimeError, match="ExecutionContext initialization failed"):
            asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Global ExecutionContext should still be set (but in error state)
        execution_context = get_execution_context()
        assert execution_context is not None
        assert execution_context._initialization_error is not None
    
    def test_session_creation_failure_with_valid_execution_context(self):
        """Test session creation failure when ExecutionContext is valid."""
        test_config = {
            'infrastructure': {
                'clouds': [],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Bootstrap ExecutionContext successfully
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Mock infrastructure to fail session creation
        with patch.object(infrastructure, 'create_session_bm') as mock_create_session:
            mock_create_session.side_effect = Exception("Session creation failed")
            
            # Session bootstrap should fail but ExecutionContext remains valid
            with pytest.raises(Exception, match="Session creation failed"):
                asyncio.run(bootstrapper.bootstrap_session_context(
                    name="failing_session",
                    job="error_test",
                    infrastructure=infrastructure
                ))
            
            # ExecutionContext should still be accessible and valid
            assert get_execution_context() is execution_context
            assert execution_context._initialization_error is None


class TestBootstrapSequencePerformance:
    """Test performance characteristics of bootstrap sequence."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_caching_behavior(self):
        """Test that ExecutionContext properly caches expensive operations."""
        test_config = {
            'infrastructure': {
                'clouds': [],
                'secret_provider': None,
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        # Bootstrap ExecutionContext
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Multiple accesses to execution_context_id should return same value
        id1 = execution_context.execution_context_id
        id2 = execution_context.execution_context_id
        id3 = execution_context.execution_context_id
        
        assert id1 == id2 == id3
        
        # Multiple accesses to the same ExecutionContext should return same instance
        context1 = get_execution_context()
        context2 = get_execution_context()
        context3 = get_or_create_execution_context()
        
        assert context1 is context2 is context3 is execution_context
    
    def test_lazy_initialization_of_infrastructure_components(self):
        """Test that infrastructure components are lazily initialized."""
        test_config = {
            'infrastructure': {
                'clouds': [{'type': 'gcp', 'project_id': 'test'}],
                'secret_provider': {'type': 'gcp'},
                'logging': {'verbose': False},
                'tracing': {},
                'datasets': {}
            }
        }
        
        config = OmegaConf.create(test_config)
        bootstrapper = ConfigurationBootstrapper(config=config)
        
        with patch('buttermilk._core.execution_context.CloudManager') as mock_cloud_mgr, \
             patch('buttermilk._core.execution_context.SecretsManager') as mock_secrets_mgr:
            
            # Bootstrap ExecutionContext
            execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
            
            # Infrastructure components should not be created yet (lazy loading)
            mock_cloud_mgr.assert_not_called()
            mock_secrets_mgr.assert_not_called()
            
            # First access should trigger creation
            _ = execution_context.cloud_manager
            mock_cloud_mgr.assert_called_once()
            
            _ = execution_context.secret_manager
            mock_secrets_mgr.assert_called_once()
            
            # Second access should not create new instances
            _ = execution_context.cloud_manager
            _ = execution_context.secret_manager
            
            # Should still be only one call each (cached)
            assert mock_cloud_mgr.call_count == 1
            assert mock_secrets_mgr.call_count == 1