"""
Integration test for weave client initialization fallback logic.

This test demonstrates that the enhanced fallback logic in BM.get_weave_client()
successfully handles scenarios where ExecutionContext is not available, which
was causing RuntimeError exceptions in API flows when orchestrator.run() tried
to get a weave client.

The test verifies that:
1. BM instances created without ExecutionContext can still get weave clients
2. The fallback logic properly initializes weave using secret manager credentials
3. The specific RuntimeError scenario that was breaking flows is resolved
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
from hydra import compose, initialize
import sys
sys.path.insert(0, '/src/buttermilk')

from buttermilk._core.bm_init import BM
from buttermilk._core.infrastructure import InfrastructureManager
from buttermilk._core.keys import SecretsManager


class TestWeaveClientFallback:
    """Test the enhanced weave client fallback logic in BM.get_weave_client()."""

    @pytest.fixture
    def mock_secret_manager(self):
        """Create a mock secret manager with WANDB credentials."""
        mock_sm = Mock(spec=SecretsManager)
        # Configure mock to return credentials when called with cfg_key="credentials_secret"
        def mock_get_secret(cfg_key=None, **kwargs):
            if cfg_key == "credentials_secret":
                return {
                    "WANDB_API_KEY": "test_wandb_key_12345",
                    "WANDB_ENTITY": "test_entity"
                }
            return {}
        mock_sm.get_secret.side_effect = mock_get_secret
        return mock_sm

    @pytest.fixture
    def mock_session_info(self):
        """Create mock session info."""
        mock_session = Mock()
        mock_session.session_id = "test_session_12345678"
        return mock_session

    @pytest.fixture
    def bm_without_execution_context(self, mock_secret_manager, mock_session_info):
        """
        Create a BM instance similar to how it's created in API flows.
        
        This simulates the exact scenario where ExecutionContext is not available,
        which was causing the RuntimeError that broke orchestrator.run() calls.
        """
        # Create minimal infrastructure without ExecutionContext
        infrastructure = Mock(spec=InfrastructureManager)
        
        # Create BM instance directly (bypassing normal initialization that sets up ExecutionContext)
        bm = BM(
            name="test_api_flow",
            job="api_test",
            platform="test",
            infrastructure=infrastructure
        )
        
        # Set up the secret manager and session info that would normally be set
        bm._secret_manager = mock_secret_manager
        bm.session_info = mock_session_info
        
        return bm

    @pytest.mark.anyio
    async def test_weave_client_fallback_without_execution_context(
        self, 
        bm_without_execution_context, 
        mock_secret_manager
    ):
        """
        Test that BM.get_weave_client() successfully falls back when ExecutionContext is not available.
        
        This is the exact scenario that was failing: orchestrator.run() would call
        `weave_client = await bm.get_weave_client()` and get a RuntimeError about
        ExecutionContext not being initialized, causing flows to sit there doing nothing.
        """
        bm = bm_without_execution_context
        
        # Mock the ExecutionContext to raise RuntimeError (simulating the failing scenario)
        with patch('buttermilk._core.bm_init.get_execution_context') as mock_get_context:
            mock_get_context.side_effect = RuntimeError("ExecutionContext not initialized")
            
            # Mock weave to verify initialization is called properly
            with patch('buttermilk._core.bm_init.weave') as mock_weave:
                mock_client = Mock()
                mock_weave.get_client.return_value = mock_client
                
                # This call would previously fail with RuntimeError
                weave_client = await bm.get_weave_client()
                
                # Verify that we got a weave client (not a RuntimeError)
                assert weave_client is not None
                assert weave_client == mock_client
                
                # Verify that weave.init() was called with proper credentials from secret manager
                mock_weave.init.assert_called_once_with(
                    project_name="test_entity/session-test_ses",  # session_id[:8]
                    autopatch_settings={"autogen": {"enabled": False}}
                )
                
                # Verify that weave.get_client() was called to get the final client
                mock_weave.get_client.assert_called_once()

    @pytest.mark.anyio
    async def test_weave_client_fallback_sets_environment_variables(
        self, 
        bm_without_execution_context
    ):
        """
        Test that the fallback logic properly sets WANDB environment variables.
        
        This ensures that when ExecutionContext is not available, the fallback
        still configures weave properly using credentials from the secret manager.
        """
        bm = bm_without_execution_context
        
        # Clear any existing WANDB environment variables
        original_api_key = os.environ.get("WANDB_API_KEY")
        original_entity = os.environ.get("WANDB_ENTITY")
        
        try:
            if "WANDB_API_KEY" in os.environ:
                del os.environ["WANDB_API_KEY"]
            if "WANDB_ENTITY" in os.environ:
                del os.environ["WANDB_ENTITY"]
            
            # Mock ExecutionContext failure
            with patch('buttermilk._core.bm_init.get_execution_context') as mock_get_context:
                mock_get_context.side_effect = RuntimeError("ExecutionContext not initialized")
                
                with patch('buttermilk._core.bm_init.weave') as mock_weave:
                    mock_weave.get_client.return_value = Mock()
                    
                    # Call get_weave_client which should set environment variables
                    await bm.get_weave_client()
                    
                    # Verify environment variables were set from secret manager
                    assert os.environ["WANDB_API_KEY"] == "test_wandb_key_12345"
                    assert os.environ["WANDB_ENTITY"] == "test_entity"
                    
        finally:
            # Restore original environment variables
            if original_api_key is not None:
                os.environ["WANDB_API_KEY"] = original_api_key
            elif "WANDB_API_KEY" in os.environ:
                del os.environ["WANDB_API_KEY"]
                
            if original_entity is not None:
                os.environ["WANDB_ENTITY"] = original_entity
            elif "WANDB_ENTITY" in os.environ:
                del os.environ["WANDB_ENTITY"]

    @pytest.mark.anyio
    async def test_weave_client_fallback_without_credentials(self, mock_session_info):
        """
        Test that fallback gracefully handles missing credentials.
        
        Even without WANDB credentials, the method should still return a weave client
        rather than failing completely.
        """
        # Create BM without secret manager or with empty credentials
        infrastructure = Mock(spec=InfrastructureManager)
        bm = BM(
            name="test_no_creds",
            job="api_test", 
            platform="test",
            infrastructure=infrastructure
        )
        bm.session_info = mock_session_info
        bm._secret_manager = None  # No secret manager
        
        with patch('buttermilk._core.bm_init.get_execution_context') as mock_get_context:
            mock_get_context.side_effect = RuntimeError("ExecutionContext not initialized")
            
            with patch('buttermilk._core.bm_init.weave') as mock_weave:
                mock_client = Mock()
                mock_weave.get_client.return_value = mock_client
                
                # Should still work, just without weave.init() call
                weave_client = await bm.get_weave_client()
                
                assert weave_client is not None
                assert weave_client == mock_client
                
                # weave.init() should not have been called without credentials
                mock_weave.init.assert_not_called()
                
                # But weave.get_client() should still be called
                mock_weave.get_client.assert_called_once()

    @pytest.mark.anyio
    async def test_weave_client_fallback_handles_init_failure(
        self, 
        bm_without_execution_context
    ):
        """
        Test that fallback gracefully handles weave.init() failures.
        
        If weave.init() fails for any reason, the method should still return
        a basic weave client rather than propagating the exception.
        """
        bm = bm_without_execution_context
        
        with patch('buttermilk._core.bm_init.get_execution_context') as mock_get_context:
            mock_get_context.side_effect = RuntimeError("ExecutionContext not initialized")
            
            with patch('buttermilk._core.bm_init.weave') as mock_weave:
                # Make weave.init() fail
                mock_weave.init.side_effect = Exception("Weave init failed")
                mock_client = Mock()
                mock_weave.get_client.return_value = mock_client
                
                # Should still return a client despite init failure
                weave_client = await bm.get_weave_client()
                
                assert weave_client is not None
                assert weave_client == mock_client
                
                # Verify that despite init failure, we still get a client
                mock_weave.init.assert_called_once()
                mock_weave.get_client.assert_called_once()

    @pytest.mark.anyio 
    async def test_demonstrates_original_error_scenario_resolution(self):
        """
        Integration test demonstrating the exact error scenario that was resolved.
        
        This test recreates the original failing scenario:
        1. API flow creates BM instance without ExecutionContext
        2. orchestrator.run() calls bm.get_weave_client()
        3. Previously: RuntimeError about ExecutionContext, flow hangs
        4. Now: Successful fallback to direct weave initialization
        """
        # Simulate how BM is created in API flows (minimal setup)
        with initialize(version_base=None, config_path="../../buttermilk/conf"):
            cfg = compose(config_name="testing")
        
        # Create minimal infrastructure (without ExecutionContext setup)
        infrastructure = Mock(spec=InfrastructureManager)
        
        # Create BM similar to API flow creation
        bm = BM(
            name="api_flow_test",
            job="orchestrator_run",
            platform="api",
            infrastructure=infrastructure
        )
        
        # Set up minimal components that would exist in API context
        mock_secret_manager = Mock(spec=SecretsManager)
        # Configure mock to return credentials when called with cfg_key="credentials_secret"
        def mock_get_secret(cfg_key=None, **kwargs):
            if cfg_key == "credentials_secret":
                return {
                    "WANDB_API_KEY": "api_test_key",
                    "WANDB_ENTITY": "api_test_entity"
                }
            return {}
        mock_secret_manager.get_secret.side_effect = mock_get_secret
        bm._secret_manager = mock_secret_manager
        
        mock_session = Mock()
        mock_session.session_id = "api_session_12345678"
        bm.session_info = mock_session
        
        # This is the call that was failing in orchestrator.run()
        with patch('buttermilk._core.bm_init.weave') as mock_weave:
            mock_weave.get_client.return_value = Mock()
            
            # This call would previously raise RuntimeError and cause flows to hang
            # Now it should succeed with fallback logic
            weave_client = await bm.get_weave_client()
            
            # Verify successful resolution of the original error
            assert weave_client is not None
            
            # Verify the fallback initialization was used
            mock_weave.init.assert_called_once_with(
                project_name="api_test_entity/session-api_sess",
                autopatch_settings={"autogen": {"enabled": False}}
            )
            
            print("✅ Original RuntimeError scenario successfully resolved!")
            print("✅ orchestrator.run() can now get weave clients without ExecutionContext!")