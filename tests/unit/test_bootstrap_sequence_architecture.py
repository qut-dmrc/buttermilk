"""Unit tests for bootstrap sequence architecture validation.

These tests validate the bootstrap sequence architecture fix that ensures:
1. Single ExecutionContext creation per process
2. ExecutionContext created before sessions  
3. Sessions use existing ExecutionContext infrastructure
4. Proper infrastructure sharing without duplication
5. OTEL tracing can access CloudManager from ExecutionContext

This addresses the fix for the bootstrap order issue where ExecutionContext
and sessions were creating separate infrastructure instances.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from omegaconf import DictConfig, OmegaConf

from buttermilk._core.execution_context import (
    ExecutionContext, 
    get_execution_context,
    set_execution_context, 
    create_execution_context,
    get_or_create_execution_context,
    _global_execution_context,
    _execution_context_initialized
)
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper


class TestExecutionContextCreation:
    """Test single ExecutionContext creation per process."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_singleton_behavior(self):
        """Test that ExecutionContext maintains singleton-like behavior."""
        # Create first ExecutionContext
        context1 = create_execution_context(
            clouds=[],
            secret_provider=None,
            logging=None,
            tracing={}
        )
        
        # Verify it's been set as global
        assert get_execution_context() is context1
        
        # Attempting to create another should fail
        with pytest.raises(RuntimeError, match="ExecutionContext has already been initialized"):
            create_execution_context()
    
    def test_get_or_create_safe_initialization(self):
        """Test safe initialization pattern with get_or_create_execution_context."""
        # First call creates new context
        context1 = get_or_create_execution_context(
            clouds=[],
            secret_provider=None,
            logging=None,
            tracing={}
        )
        
        # Second call returns existing context
        context2 = get_or_create_execution_context()
        assert context1 is context2
        
        # Third call with different args still returns existing context
        context3 = get_or_create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}],
            secret_provider={"type": "gcp"},
        )
        assert context1 is context3
    
    def test_execution_context_id_consistency(self):
        """Test that ExecutionContext has consistent ID across access."""
        context = create_execution_context()
        
        # ID should be consistent
        id1 = context.execution_context_id
        id2 = context.execution_context_id
        assert id1 == id2
        assert id1.startswith("exec-")
    
    def test_execution_context_initialization_error_prevention(self):
        """Test that ExecutionContext prevents accidental reinitialization."""
        # Create first context
        context1 = create_execution_context()
        
        # Get existing context should work
        context2 = get_execution_context()
        assert context1 is context2
        
        # But creating new one should fail
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
        
        error_msg = str(exc_info.value)
        assert "ExecutionContext has already been initialized" in error_msg
        assert "break logging configuration" in error_msg
        assert "reset verbose logging settings" in error_msg


class TestInfrastructureSharing:
    """Test infrastructure sharing between ExecutionContext and sessions."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    @patch('buttermilk._core.execution_context.CloudManager')
    @patch('buttermilk._core.execution_context.SecretsManager')
    def test_execution_context_has_own_infrastructure(self, mock_secrets_mgr, mock_cloud_mgr):
        """Test that ExecutionContext creates its own infrastructure."""
        # Mock cloud and secret configurations
        cloud_config = {"type": "gcp", "project_id": "test-project"}
        secret_config = {"type": "gcp"}
        
        context = create_execution_context(
            clouds=[cloud_config],
            secret_provider=secret_config,
            logging=None,
            tracing={}
        )
        
        # Access infrastructure properties to trigger lazy loading
        _ = context.cloud_manager
        _ = context.secret_manager
        
        # Verify infrastructure was created
        mock_cloud_mgr.assert_called_once_with(clouds=[cloud_config])
        mock_secrets_mgr.assert_called_once_with(**secret_config)
    
    @patch('buttermilk._core.execution_context.CloudManager')
    def test_infrastructure_lazy_loading(self, mock_cloud_mgr):
        """Test that infrastructure is lazily loaded when accessed."""
        cloud_config = {"type": "gcp", "project_id": "test-project"}
        context = create_execution_context(clouds=[cloud_config])
        
        # CloudManager should not be created yet
        mock_cloud_mgr.assert_not_called()
        
        # Access cloud_manager property to trigger creation
        _ = context.cloud_manager
        
        # Now CloudManager should be created
        mock_cloud_mgr.assert_called_once_with(clouds=[cloud_config])
        
        # Second access should return same instance (no additional creation)
        _ = context.cloud_manager
        mock_cloud_mgr.assert_called_once()  # Still only one call
    
    @patch('buttermilk._core.execution_context.LLMs')
    def test_llms_initialization_with_execution_context(self, mock_llms):
        """Test that LLMs can be initialized through ExecutionContext."""
        context = create_execution_context()
        
        # Mock secret manager to return LLM connections
        mock_secret_manager = Mock()
        mock_secret_manager.get_secret.return_value = {"openai": {"api_key": "test-key"}}
        context._secret_manager = mock_secret_manager
        
        # Access LLMs to trigger initialization
        _ = context.llms
        
        # Verify LLMs was initialized with connections from secret manager
        mock_llms.assert_called_once_with(connections={"openai": {"api_key": "test-key"}})


class TestBootstrapOrderValidation:
    """Test that bootstrap order is correct: ExecutionContext first, sessions second."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    @patch('buttermilk._core.config_bootstrap.get_or_create_execution_context')
    def test_bootstrap_full_context_creates_execution_context_first(self, mock_get_or_create):
        """Test that bootstrap_full_context creates ExecutionContext before infrastructure."""
        # Mock configuration
        mock_config = {
            'infrastructure': {
                'clouds': [{'type': 'gcp', 'project_id': 'test'}],
                'secret_provider': {'type': 'gcp'},
                'logging': {'verbose': True},
                'tracing': {},
                'datasets': {}
            }
        }
        
        # Mock ExecutionContext
        mock_execution_context = Mock()
        mock_execution_context.ensure_initialized = AsyncMock()
        mock_get_or_create.return_value = mock_execution_context
        
        # Create bootstrapper
        bootstrapper = ConfigurationBootstrapper(config=OmegaConf.create(mock_config))
        
        # Bootstrap should create ExecutionContext first
        with patch.object(bootstrapper, '_create_infrastructure_manager') as mock_create_infra:
            mock_infrastructure = Mock()
            mock_create_infra.return_value = mock_infrastructure
            
            # Run bootstrap
            result_context, result_infra = asyncio.run(bootstrapper.bootstrap_full_context())
            
            # Verify ExecutionContext was created with full infrastructure config
            mock_get_or_create.assert_called_once_with(
                clouds=[{'type': 'gcp', 'project_id': 'test'}],
                secret_provider={'type': 'gcp'},
                logging={'verbose': True},
                pubsub=None,
                tracing={},
                datasets={}
            )
            
            # Verify ExecutionContext initialization was awaited
            mock_execution_context.ensure_initialized.assert_called_once()
            
            # Verify infrastructure was created after ExecutionContext
            mock_create_infra.assert_called_once()
            
            assert result_context is mock_execution_context
            assert result_infra is mock_infrastructure
    
    @patch('buttermilk._core.config_bootstrap.get_or_create_execution_context')
    def test_bootstrap_session_context_uses_existing_infrastructure(self, mock_get_or_create):
        """Test that session context uses existing infrastructure from ExecutionContext."""
        # Mock existing ExecutionContext
        mock_execution_context = Mock()
        mock_get_or_create.return_value = mock_execution_context
        
        # Mock infrastructure manager
        mock_infrastructure = Mock()
        mock_session_bm = Mock()
        mock_session_bm.ensure_initialized = AsyncMock()
        mock_infrastructure.create_session_bm.return_value = mock_session_bm
        
        # Create bootstrapper
        bootstrapper = ConfigurationBootstrapper()
        
        # Bootstrap session with existing infrastructure
        result_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            name="test-session",
            job="test-job",
            infrastructure=mock_infrastructure  # Pass existing infrastructure
        ))
        
        # Verify session BM was created using existing infrastructure
        mock_infrastructure.create_session_bm.assert_called_once_with(
            name="test-session",
            job="test-job",
            platform="local"
        )
        
        # Verify session BM initialization was awaited
        mock_session_bm.ensure_initialized.assert_called_once()
        
        assert result_bm is mock_session_bm


class TestOTELTracingIntegration:
    """Test OTEL tracing integration with ExecutionContext infrastructure."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    @patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context')
    def test_otel_tracing_setup_with_execution_context(self, mock_setup_otel):
        """Test that OTEL tracing setup receives ExecutionContext."""
        otel_config = {"enabled": True, "endpoint": "http://test-endpoint"}
        
        context = create_execution_context(
            tracing={"otel": otel_config}
        )
        
        # Trigger OTEL initialization
        asyncio.run(context._initialize_otel())
        
        # Verify OTEL setup was called with ExecutionContext
        mock_setup_otel.assert_called_once_with(otel_config, context)
    
    @patch('buttermilk._core.execution_context.weave')
    def test_weave_tracing_initialization(self, mock_weave):
        """Test Weave tracing initialization through ExecutionContext."""
        weave_config = {
            "enabled": True,
            "project_id": "test-entity",
            "api_key": "test-api-key"
        }
        
        context = create_execution_context(
            tracing={"weave": weave_config}
        )
        
        # Trigger Weave initialization
        asyncio.run(context._initialize_weave())
        
        # Verify Weave was initialized with correct parameters
        mock_weave.init.assert_called_once()
        call_args = mock_weave.init.call_args
        assert "test-entity/" in call_args.kwargs["project_name"]
        assert call_args.kwargs["autopatch_settings"] == {"autogen": {"enabled": False}}
    
    def test_tracing_deferred_initialization(self):
        """Test that tracing initialization is deferred until first access."""
        otel_config = {"enabled": True}
        weave_config = {"enabled": True, "project_id": "test", "api_key": "test"}
        
        context = create_execution_context(
            tracing={"otel": otel_config, "weave": weave_config}
        )
        
        # Tracing should be marked as instrumented but not initialized
        assert context._tracing_instrumented.is_set()
        assert not context._tracing_providers_initialized
        
        # First access should trigger initialization
        with patch.object(context, '_initialize_otel') as mock_otel, \
             patch.object(context, '_initialize_weave') as mock_weave:
            
            asyncio.run(context._ensure_tracing_initialized())
            
            # Both providers should be initialized
            mock_otel.assert_called_once()
            mock_weave.assert_called_once()
            assert context._tracing_providers_initialized


class TestArchitectureCompliance:
    """Test compliance with the architecture patterns."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_process_stable_execution_context(self):
        """Test that ExecutionContext maintains process-stable behavior."""
        # Create ExecutionContext
        context1 = create_execution_context()
        context1_id = context1.execution_context_id
        
        # Get same context should return same instance and ID
        context2 = get_execution_context()
        assert context1 is context2
        assert context2.execution_context_id == context1_id
        
        # Safe get_or_create should return same instance
        context3 = get_or_create_execution_context()
        assert context1 is context3
        assert context3.execution_context_id == context1_id
    
    @patch('buttermilk._core.execution_context.setup_console_logging')
    @patch('buttermilk._core.execution_context.setup_file_logging')
    def test_logging_state_preservation(self, mock_file_logging, mock_console_logging):
        """Test that logging state is preserved across bootstrap."""
        logging_config = {"verbose": True}
        
        # Create ExecutionContext with logging
        context = create_execution_context(logging=logging_config)
        
        # Verify logging was set up during initialization
        mock_console_logging.assert_called_once_with(verbose=True)
        mock_file_logging.assert_called_once_with(
            execution_context_id=context.execution_context_id,
            verbose=True
        )
        
        # Multiple access should not re-setup logging
        _ = context.execution_context_id
        _ = context.execution_context_id
        
        # Logging setup methods should still be called only once
        assert mock_console_logging.call_count == 1
        assert mock_file_logging.call_count == 1
    
    def test_session_ephemeral_pattern(self):
        """Test that session-scoped instances reference shared infrastructure."""
        # This test verifies the conceptual pattern that ExecutionContext
        # provides stable infrastructure while session instances are ephemeral
        
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}],
            secret_provider={"type": "gcp"}
        )
        
        # Mock infrastructure components
        with patch.object(context, 'cloud_manager') as mock_cloud_mgr, \
             patch.object(context, 'secret_manager') as mock_secret_mgr:
            
            mock_cloud_mgr.return_value = Mock()
            mock_secret_mgr.return_value = Mock()
            
            # Session instances should reference the same infrastructure
            cloud_mgr_1 = context.cloud_manager
            secret_mgr_1 = context.secret_manager
            
            cloud_mgr_2 = context.cloud_manager
            secret_mgr_2 = context.secret_manager
            
            # Should be same instances (lazy-loaded singletons within ExecutionContext)
            assert cloud_mgr_1 is cloud_mgr_2
            assert secret_mgr_1 is secret_mgr_2


class TestErrorConditions:
    """Test error conditions and failure scenarios."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_get_execution_context_before_initialization(self):
        """Test error when trying to get ExecutionContext before initialization."""
        with pytest.raises(RuntimeError, match="ExecutionContext not initialized"):
            get_execution_context()
    
    @patch('buttermilk._core.execution_context.CloudManager')
    def test_execution_context_initialization_failure(self, mock_cloud_mgr):
        """Test handling of ExecutionContext initialization failure."""
        # Make CloudManager initialization fail
        mock_cloud_mgr.side_effect = Exception("Cloud authentication failed")
        
        # ExecutionContext creation should capture the error
        context = create_execution_context(clouds=[{"type": "gcp"}])
        
        # Error should be captured during sync initialization
        assert context._initialization_error is not None
        assert "Cloud authentication failed" in str(context._initialization_error)
        
        # ensure_initialized should raise the captured error
        with pytest.raises(RuntimeError, match="ExecutionContext initialization failed"):
            asyncio.run(context.ensure_initialized())
    
    def test_secret_manager_missing_configuration(self):
        """Test error when secret manager is accessed without configuration."""
        context = create_execution_context(secret_provider=None)
        
        with pytest.raises(RuntimeError, match="Secret provider configuration is missing"):
            _ = context.secret_manager
    
    @patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context')
    def test_tracing_initialization_failure_handling(self, mock_setup_otel):
        """Test handling of tracing initialization failure."""
        mock_setup_otel.side_effect = Exception("OTEL setup failed")
        
        context = create_execution_context(
            tracing={"otel": {"enabled": True}}
        )
        
        # Tracing failure should raise RuntimeError
        with pytest.raises(RuntimeError, match="OTEL tracing initialization failed"):
            asyncio.run(context._initialize_otel())


class TestBootstrapSequenceIntegration:
    """Integration test for complete bootstrap sequence."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    @patch('buttermilk._core.config_bootstrap.get_or_create_execution_context')
    @patch('buttermilk._core.infrastructure.InfrastructureManager')
    def test_complete_bootstrap_sequence(self, mock_infra_mgr, mock_get_or_create):
        """Test complete bootstrap sequence coordination."""
        # Mock configuration
        mock_config = {
            'infrastructure': {
                'clouds': [{'type': 'gcp', 'project_id': 'test'}],
                'secret_provider': {'type': 'gcp'},
                'logging': {'verbose': True},
                'tracing': {'otel': {'enabled': True}},
                'datasets': {}
            }
        }
        
        # Mock ExecutionContext and infrastructure
        mock_execution_context = Mock()
        mock_execution_context.ensure_initialized = AsyncMock()
        mock_execution_context._initialize_all_tracing_providers = AsyncMock()
        mock_get_or_create.return_value = mock_execution_context
        
        mock_infrastructure = Mock()
        mock_infra_mgr.return_value = mock_infrastructure
        
        # Mock session BM
        mock_session_bm = Mock()
        mock_session_bm.ensure_initialized = AsyncMock()
        mock_infrastructure.create_session_bm.return_value = mock_session_bm
        
        # Create bootstrapper and run complete sequence
        bootstrapper = ConfigurationBootstrapper(config=OmegaConf.create(mock_config))
        
        with patch.object(bootstrapper, '_create_infrastructure_manager', return_value=mock_infrastructure):
            # Step 1: Bootstrap full context
            exec_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
            
            # Step 2: Bootstrap session context using existing infrastructure
            session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
                name="test-session",
                job="test-job", 
                infrastructure=infrastructure
            ))
            
            # Verify sequence:
            # 1. ExecutionContext created with full infrastructure config
            mock_get_or_create.assert_called_once_with(
                clouds=[{'type': 'gcp', 'project_id': 'test'}],
                secret_provider={'type': 'gcp'},
                logging={'verbose': True},
                pubsub=None,
                tracing={'otel': {'enabled': True}},
                datasets={}
            )
            
            # 2. ExecutionContext initialization awaited
            mock_execution_context.ensure_initialized.assert_called_once()
            
            # 3. Tracing providers initialized
            mock_execution_context._initialize_all_tracing_providers.assert_called_once()
            
            # 4. Session BM created using existing infrastructure
            mock_infrastructure.create_session_bm.assert_called_once_with(
                name="test-session",
                job="test-job",
                platform="local"
            )
            
            # 5. Session BM initialization awaited
            mock_session_bm.ensure_initialized.assert_called_once()
            
            # Verify returns
            assert exec_context is mock_execution_context
            assert infrastructure is mock_infrastructure
            assert session_bm is mock_session_bm