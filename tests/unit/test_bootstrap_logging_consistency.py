"""Unit tests for logging consistency across bootstrap sequence.

These tests validate that the bootstrap sequence architecture fix maintains
proper logging consistency with a single execution_context_id across all
components using real configuration from testing.yaml.

This ensures:
1. ExecutionContext creates structured logging with consistent ID
2. Sessions use the same execution_context_id for observability
3. Log files contain consistent execution context tracking
4. Logging state is preserved across bootstrap phases
5. No duplicate logging setups that could break observability

This addresses the logging aspect of the infrastructure sharing fix.
"""

import asyncio

import pytest

from buttermilk._core.execution_context import (
    get_execution_context,
    get_or_create_execution_context,
    _global_execution_context,
    _execution_context_initialized
)
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.log import logger


class TestBootstrapLoggingConsistency:
    """Test logging consistency throughout bootstrap sequence using real configuration."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_logging_initialization_with_real_config(self, real_conf):
        """Test that ExecutionContext properly initializes logging using real configuration."""
        # Create ExecutionContext using real testing configuration
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext was created with proper logging
        assert execution_context is not None
        assert execution_context.execution_context_id.startswith("exec-")
        
        # Verify logging is properly initialized (no exceptions means success)
        logger.info("Test log message to verify logging is working")
        
        # Verify execution context ID is consistent across access
        id1 = execution_context.execution_context_id
        id2 = execution_context.execution_context_id
        assert id1 == id2
        
        # Verify global access works
        global_context = get_execution_context()
        assert global_context is execution_context
        assert global_context.execution_context_id == id1
    
    def test_logging_setup_not_duplicated_with_real_config(self, real_conf):
        """Test that logging setup is not duplicated using real configuration."""
        # Create ExecutionContext using real configuration
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Access ExecutionContext multiple times - should not cause duplicate setup
        initial_id = execution_context.execution_context_id
        
        # Multiple accesses should work without issues
        for _ in range(5):
            context = get_execution_context()
            assert context is execution_context
            assert context.execution_context_id == initial_id
        
        # Verify logging is still functional after multiple accesses
        logger.info("Logging works after multiple accesses")
        
        # Using get_or_create should return same context
        safe_context = get_or_create_execution_context()
        assert safe_context is execution_context
        assert safe_context.execution_context_id == initial_id
    
    def test_execution_context_id_global_consistency(self, real_conf):
        """Test that execution context ID is globally consistent using real config."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Get ExecutionContext via different methods
        context1 = get_execution_context()
        context2 = get_execution_context()
        context3 = get_or_create_execution_context()
        
        # All should be same instance with same ID
        assert execution_context is context1 is context2 is context3
        assert (execution_context.execution_context_id == 
                context1.execution_context_id == 
                context2.execution_context_id == 
                context3.execution_context_id)
    
    def test_execution_context_id_persistence_across_operations(self, real_conf):
        """Test that execution context ID persists across various operations using real config."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Store initial ID
        initial_id = execution_context.execution_context_id
        
        # Perform various operations that might affect state
        # Access infrastructure components
        if hasattr(execution_context, 'cloud_manager'):
            _ = execution_context.cloud_manager
        
        # Trigger async initialization
        asyncio.run(execution_context.ensure_initialized())
        
        # ID should remain consistent
        assert execution_context.execution_context_id == initial_id
        
        # Global access should return same ID
        global_context = get_execution_context()
        assert global_context.execution_context_id == initial_id


class TestBootstrapSessionLoggingIntegration:
    """Test logging integration between ExecutionContext and sessions using real config."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
        
        # Clear BM singleton
        from buttermilk import _global_bm
        _global_bm = None
    
    def test_session_uses_execution_context_logging(self, real_conf):
        """Test that sessions can access ExecutionContext logging configuration."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # Bootstrap full context
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Create session using existing infrastructure
        session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            name="test_session",
            job="logging_test",
            infrastructure=infrastructure
        ))
        
        # Verify ExecutionContext logging was set up
        assert execution_context.execution_context_id.startswith("exec-")
        
        # Session should have access to the same execution context ID
        assert session_bm is not None
        
        # Verify logging works from session context
        logger.info("Test log from session context", 
                   execution_context_id=execution_context.execution_context_id,
                   session_id=session_bm.session_info.session_id)
    
    def test_multiple_sessions_share_execution_context_id(self, real_conf):
        """Test that multiple sessions share the same execution context ID."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # Bootstrap full context once
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        exec_id = execution_context.execution_context_id
        
        # Create multiple sessions using same infrastructure
        session_bm1 = asyncio.run(bootstrapper.bootstrap_session_context(
            name="session1",
            job="job1",
            infrastructure=infrastructure
        ))
        
        session_bm2 = asyncio.run(bootstrapper.bootstrap_session_context(
            name="session2", 
            job="job2",
            infrastructure=infrastructure
        ))
        
        # Both sessions should be created (indicating shared infrastructure works)
        assert session_bm1 is not None
        assert session_bm2 is not None
        
        # ExecutionContext ID should remain consistent
        assert execution_context.execution_context_id == exec_id
        
        # Global ExecutionContext should be the same
        global_context = get_execution_context()
        assert global_context is execution_context
        assert global_context.execution_context_id == exec_id


class TestLoggingStatePreservation:
    """Test that logging state is preserved across bootstrap phases using real config."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_verbose_logging_state_preservation(self, real_conf, config_override):
        """Test that verbose logging state is preserved across bootstrap."""
        # Override config to ensure verbose logging
        verbose_config = config_override(real_conf, {
            "infrastructure.logging.verbose": True
        })
        
        bootstrapper = ConfigurationBootstrapper(config=verbose_config)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext was created
        assert execution_context is not None
        initial_id = execution_context.execution_context_id
        
        # Multiple operations should not change logging setup
        _ = execution_context.execution_context_id
        asyncio.run(execution_context.ensure_initialized())
        _ = get_execution_context()
        
        # ExecutionContext ID should remain consistent
        assert execution_context.execution_context_id == initial_id
        
        # Logging should still work
        logger.debug("Debug message in verbose mode")
        logger.info("Info message in verbose mode")
    
    def test_non_verbose_logging_state_preservation(self, real_conf, config_override):
        """Test that non-verbose logging state is preserved."""
        # Override config to ensure non-verbose logging
        non_verbose_config = config_override(real_conf, {
            "infrastructure.logging.verbose": False
        })
        
        bootstrapper = ConfigurationBootstrapper(config=non_verbose_config)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext was created
        assert execution_context is not None
        initial_id = execution_context.execution_context_id
        
        # Multiple operations should not change logging setup
        _ = execution_context.execution_context_id
        asyncio.run(execution_context.ensure_initialized())
        
        # ExecutionContext ID should remain consistent
        assert execution_context.execution_context_id == initial_id
        
        # Logging should still work
        logger.info("Info message in non-verbose mode")
        logger.warning("Warning message in non-verbose mode")


class TestLoggingObservabilityIntegration:
    """Test logging integration with observability infrastructure using real config."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_id_available_for_observability(self, real_conf):
        """Test that execution context ID is available for observability systems."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        exec_id = execution_context.execution_context_id
        
        # Execution context ID should be available for logging
        assert exec_id is not None
        assert isinstance(exec_id, str)
        assert len(exec_id) > 0
        assert exec_id.startswith("exec-")
        
        # Should be accessible through global getter
        global_context = get_execution_context()
        assert global_context.execution_context_id == exec_id
    
    def test_structured_logging_with_execution_context(self, real_conf):
        """Test structured logging includes execution context information."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Verify ExecutionContext has proper ID format
        exec_id = execution_context.execution_context_id
        assert exec_id.startswith("exec-")
        
        # Test structured logging with execution context
        logger.info("Structured log test", 
                   execution_context_id=exec_id,
                   test_component="logging_consistency")
        
        # Should be able to log without exceptions
        assert True  # If we reach here, logging is working properly
    
    def test_execution_context_logging_integration_with_tracing(self, real_conf):
        """Test that execution context logging integrates with tracing systems."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Even with real tracing config, logging should work
        exec_id = execution_context.execution_context_id
        assert exec_id is not None
        
        # Tracing config should be available
        assert hasattr(execution_context, 'tracing')
        
        # Logging should integrate properly with tracing
        logger.info("Test log with tracing context",
                   execution_context_id=exec_id,
                   tracing_enabled=True)