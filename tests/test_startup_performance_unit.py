"""Unit tests for startup performance optimizations."""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

pytestmark = pytest.mark.anyio


class TestBMInitialization:
    """Test BM class initialization performance and lazy loading."""

    def test_bm_creation_is_fast(self):
        """Test that BM instance creation doesn't do heavy work immediately."""
        start_time = time.time()
        
        with patch('buttermilk._core.bm_init.CloudManager') as mock_cloud:
            from buttermilk._core import BM
            
            # Create BM instance with minimal config
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
        
        creation_time = time.time() - start_time
        
        # BM creation should be very fast (under 100ms)
        assert creation_time < 0.1, f"BM creation took {creation_time:.3f}s, expected <0.1s"
        assert bm.name == "test"
        assert bm.job == "test"

    def test_llm_property_is_lazy(self):
        """Test that LLMs are not loaded until first access."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager') as mock_secrets:
            
            from buttermilk._core import BM
            
            bm = BM(
                name="test",
                job="test", 
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # SecretsManager should not be called during BM creation
            mock_secrets.assert_not_called()
            
            # Private LLM instance should not be set yet
            assert bm._llms_instance is None

    def test_secret_manager_is_lazy(self):
        """Test that secret manager client is not created until first access."""
        with patch('buttermilk._core.bm_init.CloudManager'):
            from buttermilk._core import BM
            
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # Secret manager should not be initialized yet
            assert bm._secret_manager is None

    def test_cloud_manager_is_lazy(self):
        """Test that cloud manager doesn't immediately authenticate."""
        with patch('google.auth.default') as mock_auth:
            from buttermilk._core import BM
            
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # Cloud authentication should not happen during BM creation
            mock_auth.assert_not_called()

    def test_weave_import_is_cached(self):
        """Test that weave import is cached after first access."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('weave.init') as mock_weave_init:
            
            from buttermilk._core import BM
            
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # First access should import and initialize weave
            weave1 = bm.weave
            mock_weave_init.assert_called_once()
            
            # Second access should use cached value
            weave2 = bm.weave
            assert weave1 is weave2
            # Should still only be called once
            assert mock_weave_init.call_count == 1


class TestLazyRouteManager:
    """Test lazy route loading functionality."""

    def test_lazy_route_manager_creation(self):
        """Test LazyRouteManager can be created."""
        from fastapi import FastAPI
        from buttermilk.api.lazy_routes import LazyRouteManager
        
        app = FastAPI()
        lazy_manager = LazyRouteManager(app)
        
        assert lazy_manager.app is app
        assert lazy_manager._deferred_routers == []
        assert not lazy_manager._core_routes_registered
        assert not lazy_manager._heavy_routes_registered

    def test_core_routes_registration(self):
        """Test that core routes can be registered immediately."""
        from fastapi import FastAPI
        from buttermilk.api.lazy_routes import LazyRouteManager
        
        app = FastAPI()
        lazy_manager = LazyRouteManager(app)
        
        lazy_manager.register_core_routes()
        
        assert lazy_manager._core_routes_registered
        # Should have health check route
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths

    def test_router_deferral(self):
        """Test that routers can be deferred for lazy loading."""
        from fastapi import FastAPI, APIRouter
        from buttermilk.api.lazy_routes import LazyRouteManager
        
        app = FastAPI()
        lazy_manager = LazyRouteManager(app)
        
        test_router = APIRouter()
        test_router.get("/test")(lambda: {"test": "response"})
        
        lazy_manager.defer_router(test_router, prefix="/api")
        
        assert len(lazy_manager._deferred_routers) == 1
        assert lazy_manager._deferred_routers[0]["prefix"] == "/api"
        assert lazy_manager._deferred_routers[0]["router"] is test_router

    async def test_heavy_routes_loaded_on_demand(self):
        """Test that heavy routes are loaded when needed."""
        from fastapi import FastAPI, APIRouter
        from buttermilk.api.lazy_routes import LazyRouteManager
        
        app = FastAPI()
        lazy_manager = LazyRouteManager(app)
        
        test_router = APIRouter()
        test_router.get("/heavy")(lambda: {"heavy": "route"})
        
        lazy_manager.defer_router(test_router, prefix="/api")
        
        # Initially should not be loaded
        assert not lazy_manager._heavy_routes_registered
        
        # Load on demand
        await lazy_manager.load_heavy_routes_on_demand()
        
        assert lazy_manager._heavy_routes_registered
        # Should now have the heavy route
        route_paths = [route.path for route in app.routes]
        assert "/api/heavy" in route_paths

    def test_needs_heavy_routes_detection(self):
        """Test detection of paths that need heavy routes."""
        from fastapi import FastAPI
        from buttermilk.api.lazy_routes import LazyRouteManager
        
        app = FastAPI()
        lazy_manager = LazyRouteManager(app)
        
        # These should trigger heavy route loading
        assert lazy_manager._needs_heavy_routes("/api/flows/trans")
        assert lazy_manager._needs_heavy_routes("/api/records/123")
        assert lazy_manager._needs_heavy_routes("/api/session/abc")
        assert lazy_manager._needs_heavy_routes("/tools/judge")
        assert lazy_manager._needs_heavy_routes("/ws/session123")
        
        # These should not
        assert not lazy_manager._needs_heavy_routes("/health")
        assert not lazy_manager._needs_heavy_routes("/flow/trans")
        assert not lazy_manager._needs_heavy_routes("/docs")


class TestSecretsManagerOptimizations:
    """Test SecretsManager lazy loading optimizations."""

    def test_secrets_manager_client_is_lazy(self):
        """Test that SecretManager client is not created until first access."""
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            from buttermilk._core.keys import SecretsManager
            
            # Create SecretsManager
            sm = SecretsManager(type="gcp", project="test-project")
            
            # Client should not be created yet
            mock_client.assert_not_called()
            
            # Access client property to trigger lazy loading
            client = sm.client
            
            # Now client should be created
            mock_client.assert_called_once()

    def test_secrets_manager_client_is_cached(self):
        """Test that SecretManager client is cached after first access."""
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            from buttermilk._core.keys import SecretsManager
            
            sm = SecretsManager(type="gcp", project="test-project")
            
            # Access client multiple times
            client1 = sm.client
            client2 = sm.client
            
            # Should only create client once
            assert mock_client.call_count == 1
            assert client1 is client2


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_bm_requires_secret_provider(self):
        """Test that BM raises error without secret provider."""
        from buttermilk._core import BM
        
        with pytest.raises(Exception):  # Should raise validation error
            BM(
                name="test",
                job="test",
                clouds=[{"type": "gcp", "project": "test-project"}]
                # Missing secret_provider
            )

    def test_storage_config_validation(self):
        """Test StorageConfig validation and computed properties."""
        from buttermilk._core.storage_config import StorageConfig
        
        # Valid config
        config = StorageConfig(
            type="bigquery",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table"
        )
        
        assert config.full_table_id == "test-project.test_dataset.test_table"
        
        # Incomplete config
        incomplete_config = StorageConfig(
            type="bigquery",
            project_id="test-project"
            # Missing dataset_id and table_id
        )
        
        assert incomplete_config.full_table_id is None

    def test_bigquery_defaults_no_hardcoded_values(self):
        """Test that BigQueryDefaults has no hardcoded values after optimization."""
        from buttermilk._core.storage_config import BigQueryDefaults
        
        defaults = BigQueryDefaults()
        
        # After Phase 1 optimizations, these should be None
        assert defaults.dataset_id is None
        assert defaults.table_id is None


class TestAsyncCacheOperations:
    """Test async cache operations don't block startup."""

    async def test_llm_cache_writing_is_async(self):
        """Test that LLM cache writing happens asynchronously."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch.object(Path, 'write_text') as mock_write:
            
            from buttermilk._core import BM
            
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # Create mock data
            test_data = {"test": "data"}
            test_path = Path("/tmp/test_cache.json")
            
            # Test the async cache method
            await bm._cache_llm_connections_async(test_data, test_path)
            
            # Should eventually call write_text
            await asyncio.sleep(0.1)  # Give time for async operation
            mock_write.assert_called()


class TestStartupTiming:
    """Test startup timing benchmarks."""

    def test_core_imports_are_fast(self):
        """Test that core imports don't take too long."""
        start_time = time.time()
        
        from buttermilk._core import BM
        from buttermilk.api.flow import create_app
        from buttermilk.api.lazy_routes import LazyRouteManager
        
        import_time = time.time() - start_time
        
        # Core imports should be under 1 second
        assert import_time < 1.0, f"Core imports took {import_time:.3f}s, expected <1.0s"

    def test_fastapi_app_creation_is_fast(self):
        """Test that FastAPI app creation is reasonably fast."""
        from fastapi import FastAPI
        
        start_time = time.time()
        
        app = FastAPI()
        
        creation_time = time.time() - start_time
        
        # FastAPI creation should be very fast
        assert creation_time < 0.05, f"FastAPI creation took {creation_time:.3f}s, expected <0.05s"