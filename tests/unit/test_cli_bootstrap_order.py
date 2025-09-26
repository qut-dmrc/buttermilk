"""Unit tests for CLI bootstrap order validation.

These tests validate the CLI bootstrap sequence using real configuration
to ensure proper order and infrastructure sharing.
"""

import asyncio

from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.execution_context import get_execution_context


class TestCLIBootstrapOrder:
    """Test CLI bootstrap order with real configuration."""

    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False

        # Clear BM singleton
        import buttermilk._core.dmrc as dmrc_module

        dmrc_module._bm_instance = None

    def test_bootstrap_order_execution_context_first(self, real_conf):
        """Test that ExecutionContext is created before session creation."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)

        # Step 1: Bootstrap ExecutionContext first (as CLI should do)
        execution_context = asyncio.run(bootstrapper.bootstrap_full_context())

        # Verify ExecutionContext is properly set up
        assert execution_context is not None
        assert execution_context.execution_context_id.startswith("exec-")

        # Verify global access works
        global_context = get_execution_context()
        assert global_context is execution_context

        # Step 2: Create session using existing infrastructure (as CLI should do)
        session_bm = asyncio.run(
            bootstrapper.bootstrap_session_context(
                name="cli_session",
                job="cli_operation",
            )
        )

        # Verify session creation succeeded and used existing ExecutionContext
        assert session_bm is not None

        # ExecutionContext should remain unchanged
        post_session_context = get_execution_context()
        assert post_session_context is execution_context
        assert post_session_context.execution_context_id == execution_context.execution_context_id

    def test_infrastructure_consistency_in_cli_pattern(self, real_conf):
        """Test that CLI pattern maintains infrastructure consistency."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)

        # Simulate CLI bootstrap pattern
        execution_context = asyncio.run(bootstrapper.bootstrap_full_context())
        initial_exec_id = execution_context.execution_context_id

        # Create session BM (as CLI does)
        session_bm = asyncio.run(bootstrapper.bootstrap_session_context(name="cli_test", job="consistency_test"))

        # Set BM as singleton (as CLI does)
        from buttermilk._core.dmrc import get_bm, set_bm

        set_bm(session_bm)

        # Verify everything is consistent
        assert get_bm() is session_bm
        assert get_execution_context() is execution_context
        assert execution_context.execution_context_id == initial_exec_id

        # Session should have proper info
        assert session_bm.session_info.session_id is not None
        assert session_bm.session_info.job == "consistency_test"

    def test_multiple_cli_operations_share_infrastructure(self, real_conf):
        """Test that multiple CLI operations can share the same infrastructure."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)

        # First CLI operation
        execution_context = asyncio.run(bootstrapper.bootstrap_full_context())

        session1 = asyncio.run(bootstrapper.bootstrap_session_context(name="cli_op1", job="operation1"))

        # Second CLI operation (should reuse infrastructure)
        session2 = asyncio.run(bootstrapper.bootstrap_session_context(name="cli_op2", job="operation2"))

        # Both operations should succeed
        assert session1 is not None
        assert session2 is not None

        # Should have different session IDs
        assert session1.session_info.session_id != session2.session_info.session_id

        # But should share the same ExecutionContext
        assert get_execution_context() is execution_context

        # ExecutionContext should remain consistent
        assert execution_context.execution_context_id.startswith("exec-")


class TestCLIConfigurationBootstrap:
    """Test CLI configuration bootstrap patterns."""

    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False

    def test_cli_session_creation_pattern(self, real_conf):
        """Test CLI session creation pattern with real configuration."""
        bootstrapper = ConfigurationBootstrapper(config=real_conf)

        # Create session with CLI-typical parameters
        session_bm = asyncio.run(
            bootstrapper.bootstrap_session_context(
                name="buttermilk",  # Default CLI name
                job="testing",  # From config
            )
        )

        # Verify session has proper CLI characteristics
        assert session_bm is not None
        assert session_bm.session_info.session_id is not None
        assert session_bm.session_info.job == "testing"

        # Should be able to set as global BM
        from buttermilk._core.dmrc import get_bm, set_bm  # Local import to avoid circular dependency

        set_bm(session_bm)
        assert get_bm() is session_bm
