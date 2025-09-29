"""Unit tests for BM.get_weave_client() delegation behavior.

Tests the delegation mechanism that attempts to use ExecutionContext.get_weave_client()
when available, falling back to direct weave.get_client() when ExecutionContext is not available.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import weave

from buttermilk._core.bm_init import BM, SessionInfo


class TestBMWeaveClientDelegation:
    """Test BM.get_weave_client() delegation and fallback behavior."""
    
    @pytest.mark.anyio
    async def test_delegation_to_execution_context_success(self):
        """Test successful delegation to ExecutionContext.get_weave_client()."""
        # Create a mock weave client that ExecutionContext will return
        mock_weave_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        mock_weave_client.project_name = "test-project"
        
        # Create mock ExecutionContext with get_weave_client method
        mock_execution_context = Mock()
        mock_execution_context.get_weave_client = AsyncMock(return_value=mock_weave_client)
        
        # Create minimal BM instance for testing (mock initialization)
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # Mock the get_execution_context function to return our mock
        with patch("buttermilk._core.execution_context.get_execution_context", return_value=mock_execution_context):
            # Call get_weave_client
            result = await bm.get_weave_client()
            
            # Verify delegation occurred
            mock_execution_context.get_weave_client.assert_called_once()
            assert result is mock_weave_client
            assert result.project_name == "test-project"

    @pytest.mark.anyio
    async def test_execution_context_get_weave_client_failure_propagates(self):
        """Test that if ExecutionContext.get_weave_client() fails, the error propagates."""
        # Create mock ExecutionContext that raises when get_weave_client is called
        mock_execution_context = Mock()
        mock_execution_context.get_weave_client = AsyncMock(side_effect=Exception("Weave initialization failed"))
        
        # Create minimal BM instance for testing (mock initialization)
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # Mock get_execution_context to return our mock that will fail
        with patch("buttermilk._core.execution_context.get_execution_context", return_value=mock_execution_context):
            # get_weave_client should propagate the exception from ExecutionContext
            with pytest.raises(Exception, match="Weave initialization failed"):
                await bm.get_weave_client()
    
    @pytest.mark.anyio
    async def test_both_paths_return_valid_weave_client_interface(self):
        """Test that both delegation and fallback paths return objects with WeaveClient interface."""
        # Test delegation path
        mock_delegated_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        mock_execution_context = Mock()
        mock_execution_context.get_weave_client = AsyncMock(return_value=mock_delegated_client)
        
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # Test delegation path
        with patch("buttermilk._core.execution_context.get_execution_context", return_value=mock_execution_context):
            delegated_result = await bm.get_weave_client()
            # Verify it's a WeaveClient-like object
            assert hasattr(delegated_result, "__class__")
            assert delegated_result is mock_delegated_client
        
        # Test fallback path
        mock_fallback_client = Mock(spec=weave.trace.weave_client.WeaveClient)

        with patch("buttermilk._core.execution_context.get_execution_context", side_effect=RuntimeError("Not initialized")):
            with patch("weave.get_client", return_value=mock_fallback_client):
                fallback_result = await bm.get_weave_client()
                # Verify it's a WeaveClient-like object
                assert hasattr(fallback_result, "__class__")
                assert fallback_result is mock_fallback_client
        
        # Both should return some form of WeaveClient-compatible objects
        # (The important thing is they both return valid clients, not the exact implementation details)
        assert delegated_result is not None
        assert fallback_result is not None
        assert hasattr(delegated_result, "__class__")
        assert hasattr(fallback_result, "__class__")
    
    @pytest.mark.anyio
    async def test_import_isolation_in_fallback(self):
        """Test that weave import in fallback doesn't interfere with module imports."""
        # Create a mock weave client for fallback
        mock_fallback_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # Mock get_execution_context to trigger fallback
        with patch("buttermilk._core.execution_context.get_execution_context", side_effect=RuntimeError("Not initialized")):
            # Mock weave.get_client() in the fallback path
            with patch("weave.get_client", return_value=mock_fallback_client) as mock_get_client:
                
                # Call get_weave_client
                result = await bm.get_weave_client()
                
                # Verify fallback weave.get_client was used
                mock_get_client.assert_called_once()
                assert result is mock_fallback_client


class TestBMWeaveClientBehaviorDocumentation:
    """Test documented behavior from the implementation docstring."""
    
    @pytest.mark.anyio
    async def test_documented_delegation_behavior(self):
        """Test the documented behavior: prefer ExecutionContext, fallback for compatibility."""
        # This test serves as living documentation of the expected behavior
        
        # Mock a properly initialized ExecutionContext
        mock_execution_context = Mock()
        mock_initialized_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        mock_initialized_client.project_name = "properly-initialized"
        mock_execution_context.get_weave_client = AsyncMock(return_value=mock_initialized_client)
        
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # When ExecutionContext is available, it should be used (preferred path)
        with patch("buttermilk._core.execution_context.get_execution_context", return_value=mock_execution_context):
            result = await bm.get_weave_client()
            assert result.project_name == "properly-initialized"
            mock_execution_context.get_weave_client.assert_called_once()
    
    @pytest.mark.anyio
    async def test_documented_fallback_behavior(self):
        """Test the documented fallback behavior for backward compatibility."""
        # Mock a direct weave client for backward compatibility
        mock_direct_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        mock_direct_client.project_name = "backward-compatible"
        
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # When ExecutionContext is not available, fallback to direct weave.get_client()
        with patch(
            "buttermilk._core.execution_context.get_execution_context",
            side_effect=RuntimeError("ExecutionContext not initialized. Call set_execution_context() first."),
        ):
            with patch("weave.get_client", return_value=mock_direct_client):
                result = await bm.get_weave_client()
                assert result.project_name == "backward-compatible"
    
    @pytest.mark.anyio
    async def test_proper_weave_init_vs_fallback_documentation(self):
        """Document the difference between proper weave.init() and fallback behavior."""
        # The implementation comments note that the fallback may not have proper
        # initialization if weave.init() wasn't called. This test documents that behavior.
        
        # Properly initialized client (via ExecutionContext)
        mock_execution_context = Mock()
        mock_proper_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        mock_proper_client.initialized = True  # Represents proper initialization
        mock_execution_context.get_weave_client = AsyncMock(return_value=mock_proper_client)
        
        # Fallback client (may not be properly initialized)
        mock_fallback_client = Mock(spec=weave.trace.weave_client.WeaveClient)
        mock_fallback_client.initialized = False  # Represents potential lack of initialization
        
        session_info = SessionInfo(project_name="test-project", job="weave-test")
        with patch.object(BM, "_post_init_setup"):
            bm = BM(session_info=session_info)
        
        # Test proper initialization path
        with patch("buttermilk._core.execution_context.get_execution_context", return_value=mock_execution_context):
            proper_result = await bm.get_weave_client()
            assert proper_result.initialized is True
        
        # Test fallback path
        with patch("buttermilk._core.execution_context.get_execution_context", side_effect=RuntimeError("Not initialized")):
            with patch("weave.get_client", return_value=mock_fallback_client):
                fallback_result = await bm.get_weave_client()
                assert fallback_result.initialized is False
        
        # This documents the key difference: ExecutionContext provides properly initialized clients
        assert proper_result.initialized != fallback_result.initialized
