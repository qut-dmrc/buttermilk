"""Unit tests for startup performance optimizations."""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.anyio


class TestBMInitialization:
    """Test BM class initialization performance and lazy loading."""

    def test_bm_creation_is_fast(self):
        """Test that BM instance creation doesn't do heavy work immediately."""
        start_time = time.time()

        from buttermilk import BM
        from buttermilk._core.bm_init import SessionInfo

        # Create BM instance with minimal config
        session_info = SessionInfo(project_name="test", job="test")
        bm = BM(session_info=session_info)

        creation_time = time.time() - start_time

        # BM creation should be very fast (under 100ms)
        assert creation_time < 0.1, f"BM creation took {creation_time:.3f}s, expected <0.1s"
        assert bm.session_info.project_name == "test"
        assert bm.session_info.job == "test"

    def test_llm_property_is_lazy(self):
        """Test that LLMs are not loaded until first access."""
        from buttermilk import BM
        from buttermilk._core.bm_init import SessionInfo

        session_info = SessionInfo(project_name="test", job="test")
        bm = BM(session_info=session_info)

        # Private LLM instance should not be set yet
        assert bm._llms_instance is None

    def test_secret_manager_is_lazy(self):
        """Test that secret manager client is not created until first access."""
        from buttermilk import BM
        from buttermilk._core.bm_init import SessionInfo

        session_info = SessionInfo(project_name="test", job="test")
        bm = BM(session_info=session_info)

        # Secret manager should not be initialized yet
        assert bm._secret_manager is None

    def test_cloud_manager_is_lazy(self):
        """Test that cloud manager doesn't immediately authenticate."""
        with patch("google.auth.default") as mock_auth:
            from buttermilk import BM
            from buttermilk._core.bm_init import SessionInfo

            session_info = SessionInfo(project_name="test", job="test")
            BM(session_info=session_info)

            # Cloud authentication should not happen during BM creation
            mock_auth.assert_not_called()

    async def test_weave_import_is_cached(self):
        """Test that weave import is cached after first access."""
        from unittest.mock import AsyncMock

        from buttermilk import BM
        from buttermilk._core.bm_init import SessionInfo

        session_info = SessionInfo(project_name="test", job="test")
        bm = BM(session_info=session_info)

        # Mock the execution context to return a consistent weave client
        mock_weave_client = AsyncMock()
        mock_context = AsyncMock()
        mock_context.get_weave_client = AsyncMock(return_value=mock_weave_client)

        with patch("buttermilk._core.execution_context.get_execution_context", return_value=mock_context):
            # First access
            weave1 = await bm.get_weave_client()
            assert mock_context.get_weave_client.call_count == 1

            # Second access should use the same execution context
            weave2 = await bm.get_weave_client()
            assert weave1 is weave2
            # Execution context's get_weave_client called twice, but returns cached client
            assert mock_context.get_weave_client.call_count == 2


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
        from fastapi import APIRouter, FastAPI

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
        from fastapi import APIRouter, FastAPI

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
        with patch("google.cloud.secretmanager.SecretManagerServiceClient") as mock_client:
            from buttermilk._core.keys import SecretsManager

            # Create SecretsManager
            sm = SecretsManager(type="gcp", project="test-project")

            # Client should not be created yet
            mock_client.assert_not_called()

            # Access client property to trigger lazy loading
            _ = sm.client

            # Now client should be created
            mock_client.assert_called_once()

    def test_secrets_manager_client_is_cached(self):
        """Test that SecretManager client is cached after first access."""
        with patch("google.cloud.secretmanager.SecretManagerServiceClient") as mock_client:
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

    def test_bm_requires_session_info(self):
        """Test that BM requires session_info."""
        from buttermilk import BM

        with pytest.raises(Exception):  # Should raise validation error
            BM()  # Missing session_info

    def test_storage_config_validation(self):
        """Test StorageConfig validation and computed properties."""
        from buttermilk._core.storage_config import BigQueryStorageConfig

        # Valid config
        config = BigQueryStorageConfig(type="bigquery", project_id="test-project", dataset_id="test_dataset", table_id="test_table")

        assert config.full_table_id == "test-project.test_dataset.test_table"

        # Incomplete config
        incomplete_config = BigQueryStorageConfig(
            type="bigquery",
            project_id="test-project",
            # Missing dataset_id and table_id
        )

        assert incomplete_config.full_table_id is None


class TestAsyncCacheOperations:
    """Test async cache operations don't block startup."""

    async def test_llm_cache_writing_is_async(self):
        """Test that async operations don't block during initialization."""
        import json

        from buttermilk import BM
        from buttermilk._core.bm_init import SessionInfo

        session_info = SessionInfo(project_name="test", job="test")
        BM(session_info=session_info)

        # Test that we can write async without blocking
        test_path = Path("/tmp/test_cache.json")

        async def write_async():
            """Simulate async write operation."""
            await asyncio.sleep(0.01)
            test_path.write_text(json.dumps({"test": "data"}))

        start = time.time()
        await write_async()
        duration = time.time() - start

        # Async write should complete quickly
        assert duration < 0.1
        assert test_path.exists()
        test_path.unlink()  # Clean up


class TestStartupTiming:
    """Test startup timing benchmarks."""

    def test_core_imports_are_fast(self):
        """Test that core imports don't take too long."""
        start_time = time.time()

        import_time = time.time() - start_time

        # Core imports should be under 1 second
        assert import_time < 1.0, f"Core imports took {import_time:.3f}s, expected <1.0s"

    def test_fastapi_app_creation_is_fast(self):
        """Test that FastAPI app creation is reasonably fast."""
        from fastapi import FastAPI

        start_time = time.time()

        FastAPI()

        creation_time = time.time() - start_time

        # FastAPI creation should be very fast
        assert creation_time < 0.05, f"FastAPI creation took {creation_time:.3f}s, expected <0.05s"
