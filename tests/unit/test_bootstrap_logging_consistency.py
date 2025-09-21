"""Unit tests for logging consistency across bootstrap sequence.

Simple tests using real configuration from testing.yaml.
"""

import asyncio

from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.execution_context import get_execution_context
from buttermilk._core.log import logger


class TestBootstrapLoggingConsistency:
    """Test logging consistency using real configuration."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    def test_execution_context_logging_works(self, real_conf):
        """Test that ExecutionContext properly initializes logging."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Basic checks
        assert execution_context is not None
        assert execution_context.execution_context_id.startswith("exec-")
        
        # Logging should work without exceptions
        logger.info("Test log message")
    
    def test_execution_context_id_consistent(self, real_conf):
        """Test that execution context ID is consistent."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # ID should be consistent across access
        id1 = execution_context.execution_context_id
        id2 = execution_context.execution_context_id
        assert id1 == id2
        
        # Global access should work
        global_context = get_execution_context()
        assert global_context is execution_context
    
    def test_session_creation_works(self, real_conf):
        """Test that sessions can be created using ExecutionContext."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        
        # Bootstrap full context
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Create session - should not raise exceptions
        session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            name="test_session",
            job="test_job",
            infrastructure=infrastructure
        ))
        
        assert session_bm is not None
        assert execution_context.execution_context_id.startswith("exec-")
