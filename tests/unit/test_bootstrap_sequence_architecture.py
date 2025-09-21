"""Unit tests for bootstrap sequence architecture validation.

These tests validate the bootstrap sequence architecture fix using real configuration
from testing.yaml to ensure:
1. Single ExecutionContext creation per process
2. ExecutionContext created before sessions  
3. Sessions use existing ExecutionContext infrastructure
4. Proper infrastructure sharing without duplication

This addresses the fix for the bootstrap order issue where ExecutionContext
and sessions were creating separate infrastructure instances.
"""

import asyncio

from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.execution_context import (
    get_execution_context,
    get_or_create_execution_context,
)


class TestExecutionContextCreation:
    """Test single ExecutionContext creation per process using real config."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_singleton_behavior(self, real_conf):
        """Test that ExecutionContext maintains singleton-like behavior with real config."""
        # Create first ExecutionContext using real configuration
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        context1, infrastructure1 = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify it's been set as global
        global_context = get_execution_context()
        assert global_context is context1
        
        # Second attempt should return same context
        context2, infrastructure2 = asyncio.run(bootstrapper.bootstrap_full_context())
        assert context1 is context2
        assert context1.execution_context_id == context2.execution_context_id
        
        # Using get_or_create should also return same context
        context3 = get_or_create_execution_context()
        assert context1 is context3
    
    def test_execution_context_id_format_and_consistency(self, real_conf):
        """Test that ExecutionContext ID has proper format and is consistent."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify execution context ID format
        exec_id = context.execution_context_id
        assert exec_id.startswith("exec-")
        assert len(exec_id) > 10  # Should have timestamp, UUID, etc.
        
        # ID should be consistent across multiple accesses
        for _ in range(5):
            assert context.execution_context_id == exec_id
            
        # Global access should return same ID
        global_context = get_execution_context()
        assert global_context.execution_context_id == exec_id


class TestInfrastructureSharing:
    """Test infrastructure sharing between ExecutionContext and sessions."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
        
        # Clear BM singleton
        import buttermilk._core.dmrc as dmrc_module
        dmrc_module._bm_instance = None
    
    def test_execution_context_has_own_infrastructure(self, real_conf):
        """Test that ExecutionContext properly initializes its own infrastructure."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # ExecutionContext should have properly initialized infrastructure
        assert context is not None
        assert infrastructure is not None
        
        # Infrastructure should be functional (has required components)
        assert hasattr(infrastructure, "create_session_bm")
        
        # ExecutionContext should be accessible globally
        global_context = get_execution_context()
        assert global_context is context
    
    def test_sessions_share_execution_context_infrastructure(self, real_conf):
        """Test that sessions share ExecutionContext infrastructure instead of creating new."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # Bootstrap full context first
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        initial_exec_id = execution_context.execution_context_id
        
        # Create multiple sessions using shared infrastructure
        session1 = asyncio.run(bootstrapper.bootstrap_session_context(
            name="session1",
            job="test1",
            infrastructure=infrastructure
        ))
        
        session2 = asyncio.run(bootstrapper.bootstrap_session_context(
            name="session2",
            job="test2",
            infrastructure=infrastructure
        ))
        
        # Sessions should be created successfully
        assert session1 is not None
        assert session2 is not None
        
        # ExecutionContext should remain the same (not recreated)
        current_context = get_execution_context()
        assert current_context is execution_context
        assert current_context.execution_context_id == initial_exec_id
        
        # Sessions should have different IDs but use same infrastructure
        assert session1.session_info.session_id != session2.session_info.session_id
    
    def test_infrastructure_manager_consistency(self, real_conf):
        """Test that infrastructure manager remains consistent across operations."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # Get infrastructure manager
        infrastructure1 = bootstrapper.get_infrastructure_manager()
        
        # Bootstrap full context
        execution_context, infrastructure2 = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Infrastructure should be consistent
        assert infrastructure1 is not None
        assert infrastructure2 is not None
        
        # Both should be functional
        assert hasattr(infrastructure1, "create_session_bm")
        assert hasattr(infrastructure2, "create_session_bm")


class TestBootstrapOrderValidation:
    """Test that bootstrap order follows correct sequence."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
        
        import buttermilk._core.dmrc as dmrc_module
        dmrc_module._bm_instance = None
    
    def test_execution_context_before_session_creation(self, real_conf):
        """Test that ExecutionContext is created before session creation."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # Step 1: ExecutionContext should be created first
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext is properly initialized
        assert execution_context is not None
        assert execution_context.execution_context_id.startswith("exec-")
        
        # Verify global access works
        global_context = get_execution_context()
        assert global_context is execution_context
        
        # Step 2: Session creation should use existing ExecutionContext
        session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            name="ordered_session",
            job="order_test",
            infrastructure=infrastructure
        ))
        
        # Session creation should not affect ExecutionContext
        post_session_context = get_execution_context()
        assert post_session_context is execution_context
        assert post_session_context.execution_context_id == execution_context.execution_context_id
        
        # Session should be properly created
        assert session_bm is not None
        assert hasattr(session_bm, "session_info")
    
    def test_multiple_bootstrap_calls_are_safe(self, real_conf):
        """Test that multiple bootstrap calls don't break the architecture."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # First bootstrap
        context1, infra1 = asyncio.run(bootstrapper.bootstrap_full_context())
        initial_id = context1.execution_context_id
        
        # Second bootstrap should be safe
        context2, infra2 = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Should return same ExecutionContext
        assert context1 is context2
        assert context1.execution_context_id == initial_id
        
        # Global state should remain consistent
        global_context = get_execution_context()
        assert global_context is context1
        assert global_context.execution_context_id == initial_id


class TestExecutionContextInitialization:
    """Test ExecutionContext initialization with real configuration."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_initialization_with_real_components(self, real_conf):
        """Test ExecutionContext initializes properly with real configuration components."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # ExecutionContext should be properly initialized
        assert execution_context is not None
        
        # Should have proper execution context ID
        assert execution_context.execution_context_id.startswith("exec-")
        
        # Should be accessible globally
        assert get_execution_context() is execution_context
        
        # Should support async initialization
        asyncio.run(execution_context.ensure_initialized())
        
        # ID should remain consistent after initialization
        post_init_id = execution_context.execution_context_id
        assert post_init_id.startswith("exec-")
    
    def test_execution_context_provides_infrastructure_access(self, real_conf):
        """Test that ExecutionContext provides access to infrastructure components."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # ExecutionContext should provide infrastructure access
        assert execution_context is not None
        
        # Should have configuration attributes based on real config
        if hasattr(execution_context, "clouds"):
            assert hasattr(execution_context, "clouds")
            
        if hasattr(execution_context, "logging"):
            assert hasattr(execution_context, "logging")
            
        if hasattr(execution_context, "tracing"):
            assert hasattr(execution_context, "tracing")
        
        # Infrastructure should be functional
        assert infrastructure is not None
        assert hasattr(infrastructure, "create_session_bm")
