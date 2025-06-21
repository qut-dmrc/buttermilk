"""Test BM async initialization behavior.

This test verifies that:
1. Logger is configured early and available
2. Google Cloud authentication happens before first use
3. Secrets are fetched and cached asynchronously
4. GCS save_dir is properly set from config
"""

import asyncio
import tempfile
from unittest.mock import Mock, patch, AsyncMock
import pytest

from buttermilk._core.bm_init import BM
from buttermilk._core.config import CloudProviderCfg


@pytest.fixture
def mock_cloud_config():
    """Create mock cloud configuration."""
    return CloudProviderCfg(
        type="gcp",
        project_id="test-project",
        quota_project_id="test-project",
    )


@pytest.fixture
def mock_logger_config():
    """Create mock logger configuration."""
    return {
        "type": "gcp",
        "project": "test-project",
        "location": "us-central1",
        "verbose": True,
    }


@pytest.fixture 
def mock_secret_config():
    """Create mock secret provider configuration."""
    return CloudProviderCfg(
        type="gcp",
        project="test-project",
        models_secret="test_models",
        credentials_secret="test_credentials",
    )


class TestBMAsyncInitialization:
    """Test BM async initialization behavior."""
    
    @pytest.mark.asyncio
    async def test_initialization_completes_before_use(self, mock_cloud_config, mock_logger_config, mock_secret_config):
        """Test that ensure_initialized waits for background tasks."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('buttermilk._core.bm_init.logger') as mock_logger:
            
            # Create BM instance with GCS save_dir
            bm = BM(
                platform="test",
                name="test",
                job="test-job",
                save_dir_base="gs://test-bucket/runs",
                clouds=[mock_cloud_config],
                logger_cfg=mock_logger_config,
                secret_provider=mock_secret_config,
            )
            
            # Verify initialization event is created
            assert hasattr(bm, '_initialization_complete')
            assert hasattr(bm, '_initialization_error')
            
            # Wait for initialization
            await bm.ensure_initialized()
            
            # Verify logger was called during initialization
            assert mock_logger.debug.called
            assert mock_logger.info.called
            
            # Verify save_dir includes GCS path
            assert bm.save_dir.startswith("gs://test-bucket/runs")
            assert "test/test-job" in bm.save_dir
    
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, mock_cloud_config):
        """Test that initialization errors are properly propagated."""
        with patch('buttermilk._core.bm_init.CloudManager') as mock_cloud_manager:
            # Make cloud manager raise an error
            mock_cloud_manager.side_effect = Exception("Cloud auth failed")
            
            bm = BM(
                platform="test",
                name="test", 
                job="test-job",
                clouds=[mock_cloud_config],
            )
            
            # Should raise error when waiting for initialization
            with pytest.raises(RuntimeError, match="BM initialization failed"):
                await bm.ensure_initialized()
    
    def test_sync_initialization_fallback(self, mock_cloud_config, mock_secret_config):
        """Test synchronous initialization when no event loop is running."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('buttermilk._core.bm_init.logger') as mock_logger, \
             patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop")):
            
            # Create BM instance - should fall back to sync init
            bm = BM(
                platform="test",
                name="test",
                job="test-job", 
                save_dir_base="gs://test-bucket/runs",
                clouds=[mock_cloud_config],
                secret_provider=mock_secret_config,
            )
            
            # Verify sync initialization was called
            mock_logger.debug.assert_any_call("No event loop available, performing synchronous initialization")
            
            # Initialization should be marked complete immediately
            assert bm._initialization_complete.is_set()
    
    @pytest.mark.asyncio
    async def test_cloud_manager_lazy_initialization(self, mock_cloud_config):
        """Test that cloud manager is initialized on first access."""
        with patch('buttermilk._core.bm_init.CloudManager') as mock_cloud_manager_class:
            mock_instance = Mock()
            mock_cloud_manager_class.return_value = mock_instance
            
            bm = BM(
                platform="test",
                name="test",
                job="test-job",
                clouds=[mock_cloud_config],
            )
            
            # Cloud manager should not be created yet
            mock_cloud_manager_class.assert_not_called()
            
            # Access cloud manager property
            _ = bm.cloud_manager
            
            # Now it should be created
            mock_cloud_manager_class.assert_called_once()
            mock_instance.login_clouds.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_secret_manager_early_initialization(self, mock_secret_config):
        """Test that secret manager is initialized during background init."""
        with patch('buttermilk._core.bm_init.SecretsManager') as mock_secret_manager_class, \
             patch('buttermilk._core.bm_init.CloudManager'):
            
            mock_instance = Mock()
            mock_secret_manager_class.return_value = mock_instance
            
            bm = BM(
                platform="test",
                name="test",
                job="test-job",
                secret_provider=mock_secret_config,
            )
            
            # Wait for background initialization
            await bm.ensure_initialized()
            
            # Secret manager should have been created during init
            mock_secret_manager_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gcp_environment_variables_set_early(self, mock_cloud_config):
        """Test that GCP environment variables are set immediately."""
        import os
        original_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        original_quota = os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT")
        
        try:
            bm = BM(
                platform="test",
                name="test",
                job="test-job",
                clouds=[mock_cloud_config],
            )
            
            # Environment variables should be set immediately
            assert os.environ.get("GOOGLE_CLOUD_PROJECT") == "test-project"
            assert os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT") == "test-project"
            
        finally:
            # Restore original values
            if original_project:
                os.environ["GOOGLE_CLOUD_PROJECT"] = original_project
            else:
                os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            
            if original_quota:
                os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = original_quota  
            else:
                os.environ.pop("GOOGLE_CLOUD_QUOTA_PROJECT", None)
    
    @pytest.mark.asyncio
    async def test_multiple_ensure_initialized_calls(self):
        """Test that ensure_initialized can be called multiple times safely."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'):
            
            bm = BM(platform="test", name="test", job="test-job")
            
            # Call ensure_initialized multiple times
            await bm.ensure_initialized()
            await bm.ensure_initialized()
            await bm.ensure_initialized()
            
            # Should not raise any errors
            assert bm._initialization_complete.is_set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])