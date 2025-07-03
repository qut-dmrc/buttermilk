"""Unit tests for lazy loading utilities and patterns."""

import asyncio
import pytest
from unittest.mock import Mock, patch, call
import time

pytestmark = pytest.mark.anyio


class TestCachedProperty:
    """Test the cached_property decorator functionality."""

    def test_cached_property_basic_functionality(self):
        """Test that cached_property works like a property but caches results."""
        from buttermilk._core.utils.lazy_loading import cached_property
        
        call_count = 0
        
        class TestClass:
            @cached_property
            def expensive_operation(self):
                nonlocal call_count
                call_count += 1
                return f"result_{call_count}"
        
        obj = TestClass()
        
        # First access should compute value
        result1 = obj.expensive_operation
        assert result1 == "result_1"
        assert call_count == 1
        
        # Second access should return cached value
        result2 = obj.expensive_operation
        assert result2 == "result_1"  # Same result
        assert call_count == 1  # Not computed again
        
        # Different instance should compute separately
        obj2 = TestClass()
        result3 = obj2.expensive_operation
        assert result3 == "result_2"
        assert call_count == 2

    def test_cached_property_with_exceptions(self):
        """Test cached_property behavior when the method raises exceptions."""
        from buttermilk._core.utils.lazy_loading import cached_property
        
        call_count = 0
        
        class TestClass:
            @cached_property
            def failing_operation(self):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ValueError("First call fails")
                return "success"
        
        obj = TestClass()
        
        # First call should raise exception
        with pytest.raises(ValueError, match="First call fails"):
            obj.failing_operation
        
        # Exception should not be cached - second call should succeed
        result = obj.failing_operation
        assert result == "success"
        assert call_count == 2

    def test_cached_property_performance(self):
        """Test that cached_property provides performance benefits."""
        from buttermilk._core.utils.lazy_loading import cached_property
        
        class TestClass:
            @cached_property
            def slow_operation(self):
                time.sleep(0.01)  # Simulate slow operation
                return "computed"
        
        obj = TestClass()
        
        # First access - should be slow
        start_time = time.time()
        result1 = obj.slow_operation
        first_duration = time.time() - start_time
        
        # Second access - should be fast (cached)
        start_time = time.time()
        result2 = obj.slow_operation
        second_duration = time.time() - start_time
        
        assert result1 == result2 == "computed"
        assert first_duration > 0.005  # Should take at least 5ms
        assert second_duration < 0.001  # Should be very fast (under 1ms)


class TestCloudManagerLazyLoading:
    """Test CloudManager lazy loading patterns."""

    def test_cloud_manager_gcp_credentials_lazy(self):
        """Test that GCP credentials are not fetched until needed."""
        with patch('google.auth.default') as mock_auth:
            from buttermilk._core.cloud import CloudManager
            
            # Create CloudManager
            cm = CloudManager(clouds=[{"type": "gcp", "project": "test-project"}])
            
            # Credentials should not be fetched yet
            mock_auth.assert_not_called()
            
            # Access credentials to trigger lazy loading
            creds = cm.gcp_credentials
            
            # Now credentials should be fetched
            mock_auth.assert_called_once()

    def test_cloud_manager_clients_are_lazy(self):
        """Test that cloud clients are not created until accessed."""
        with patch('google.auth.default'), \
             patch('google.cloud.storage.Client') as mock_storage, \
             patch('google.cloud.bigquery.Client') as mock_bq:
            
            from buttermilk._core.cloud import CloudManager
            
            cm = CloudManager(clouds=[{"type": "gcp", "project": "test-project"}])
            
            # Clients should not be created yet
            mock_storage.assert_not_called()
            mock_bq.assert_not_called()
            
            # Access storage client
            storage_client = cm.gcs
            mock_storage.assert_called_once()
            mock_bq.assert_not_called()
            
            # Access BigQuery client
            bq_client = cm.bq
            mock_bq.assert_called_once()

    def test_cloud_manager_credentials_cached(self):
        """Test that credentials are cached after first access."""
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = ("fake_creds", "fake_project")
            
            from buttermilk._core.cloud import CloudManager
            
            cm = CloudManager(clouds=[{"type": "gcp", "project": "test-project"}])
            
            # Access credentials multiple times
            creds1 = cm.gcp_credentials
            creds2 = cm.gcp_credentials
            
            # Should only call google.auth.default once
            assert mock_auth.call_count == 1
            assert creds1 is creds2


class TestLLMManagerLazyLoading:
    """Test LLM manager lazy loading optimizations."""

    def test_llm_connections_cache_loading(self):
        """Test LLM connections are loaded from cache when available."""
        from buttermilk._core.llms import LLMs
        
        # Mock connections data
        test_connections = {
            "gemini": {"api_key": "test_key", "model": "gemini-pro"},
            "openai": {"api_key": "test_key2", "model": "gpt-4"}
        }
        
        llms = LLMs(connections=test_connections)
        
        # Should store connections
        assert llms.connections == test_connections

    def test_llm_client_lazy_initialization(self):
        """Test that LLM clients are not created until first use."""
        from buttermilk._core.llms import LLMs
        
        test_connections = {
            "gemini": {"api_key": "test_key", "model": "gemini-pro"}
        }
        
        with patch('google.generativeai.configure') as mock_configure:
            llms = LLMs(connections=test_connections)
            
            # Configuration should not happen during LLMs creation
            mock_configure.assert_not_called()


class TestQueryRunnerLazyLoading:
    """Test QueryRunner lazy loading patterns."""

    def test_query_runner_client_dependency(self):
        """Test that QueryRunner depends on lazy-loaded BigQuery client."""
        with patch('google.cloud.bigquery.Client') as mock_bq_client:
            from buttermilk._core.query import QueryRunner
            
            # Create mock BigQuery client
            fake_client = Mock()
            
            # Create QueryRunner with the client
            qr = QueryRunner(bq_client=fake_client)
            
            assert qr.client is fake_client


class TestAsyncBackgroundOperations:
    """Test async background operations don't block startup."""

    async def test_background_config_saving(self):
        """Test that config saving can happen in background."""
        from buttermilk._core import BM
        
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk.utils.save.save') as mock_save:
            
            # Make save async to simulate real behavior
            async def async_save(*args, **kwargs):
                return {"uri": "/tmp/test", "run_id": "test"}
            
            mock_save.return_value = async_save()
            
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # Config saving should not block BM creation
            assert bm.name == "test"

    async def test_ip_fetching_is_background(self):
        """Test that IP address fetching happens in background."""
        from buttermilk._core import BM
        
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk.utils.get_ip') as mock_get_ip:
            
            # Make IP fetching slow to test it doesn't block
            async def slow_ip_fetch():
                await asyncio.sleep(0.1)
                return "192.168.1.1"
            
            mock_get_ip.return_value = slow_ip_fetch()
            
            # BM creation should not wait for IP
            start_time = time.time()
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            creation_time = time.time() - start_time
            
            # Should be fast despite slow IP fetch
            assert creation_time < 0.10
            assert bm.name == "test"


class TestMemoryEfficiency:
    """Test memory efficiency of lazy loading."""

    def test_large_objects_not_created_unnecessarily(self):
        """Test that large objects are not created until needed."""
        from buttermilk._core import BM
        
        with patch('buttermilk._core.bm_init.CloudManager'):
            bm = BM(
                name="test",
                job="test",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )
            
            # These should all be None initially (not created)
            assert bm._llms_instance is None
            assert bm._secret_manager is None
            assert bm._query_runner is None

    def test_cached_properties_save_memory(self):
        """Test that cached properties don't create multiple instances."""
        from buttermilk._core.utils.lazy_loading import cached_property
        
        class TestClass:
            @cached_property
            def large_object(self):
                return {"large": "data" * 1000}  # Simulate large object
        
        obj = TestClass()
        
        # Access multiple times
        data1 = obj.large_object
        data2 = obj.large_object
        
        # Should be the same object (not copied)
        assert data1 is data2


class TestErrorHandlingInLazyLoading:
    """Test error handling in lazy loading scenarios."""

    def test_lazy_loading_with_missing_dependencies(self):
        """Test graceful handling when lazy-loaded dependencies are missing."""
        from buttermilk._core.utils.lazy_loading import cached_property
        
        class TestClass:
            @cached_property
            def missing_dependency(self):
                import non_existent_module  # This will fail
                return non_existent_module.something
        
        obj = TestClass()
        
        with pytest.raises(ImportError):
            obj.missing_dependency

    def test_lazy_loading_error_recovery(self):
        """Test that lazy loading can recover from transient errors."""
        from buttermilk._core.utils.lazy_loading import cached_property
        
        call_count = 0
        
        class TestClass:
            @cached_property
            def sometimes_failing(self):
                nonlocal call_count
                call_count += 1
                if call_count < 3:  # Fail first 2 times
                    raise ConnectionError("Temporary failure")
                return "success"
        
        obj = TestClass()
        
        # First few calls should fail
        with pytest.raises(ConnectionError):
            obj.sometimes_failing
        
        with pytest.raises(ConnectionError):
            obj.sometimes_failing
        
        # Third call should succeed and be cached
        result = obj.sometimes_failing
        assert result == "success"
        assert call_count == 3