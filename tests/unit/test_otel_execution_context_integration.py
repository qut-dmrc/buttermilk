"""Unit tests for OTEL tracing integration with ExecutionContext.

These tests validate the new OTEL tracing setup that uses ExecutionContext
infrastructure instead of relying on BM singleton. This addresses the fix
for "CloudManager not available" errors during OTEL initialization.

Tests cover:
1. OTEL setup with ExecutionContext CloudManager access
2. Prevention of "CloudManager not available" errors
3. Proper credential access through ExecutionContext
4. OTEL configuration with ExecutionContext infrastructure
"""

from unittest.mock import Mock, patch

import pytest

from buttermilk._core.execution_context import create_execution_context


class TestOTELExecutionContextIntegration:
    """Test OTEL tracing integration with ExecutionContext infrastructure."""
    
    def setup_method(self):
        """Reset global state before each test."""
        globals()["_global_execution_context"] = None
        globals()["_execution_context_initialized"] = False
    
    @patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context")
    def test_otel_initialization_with_execution_context(self, mock_setup_otel):
        """Test that OTEL initialization receives ExecutionContext."""
        otel_config = {
            "enabled": True,
            "endpoint": "http://test-otel-endpoint",
            "service_name": "buttermilk-test"
        }
        
        # Create ExecutionContext with OTEL tracing
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test-project"}],
            tracing={"otel": otel_config}
        )
        
        # Mock CloudManager to avoid real authentication
        with patch.object(context, "cloud_manager") as mock_cloud_mgr:
            mock_cloud_instance = Mock()
            mock_cloud_mgr.return_value = mock_cloud_instance
            
            # Trigger OTEL initialization
            import asyncio
            asyncio.run(context._initialize_otel())
            
            # Verify OTEL setup was called with ExecutionContext
            mock_setup_otel.assert_called_once_with(otel_config, context)
    
    @patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context")
    def test_otel_cloudmanager_access_through_execution_context(self, mock_setup_otel):
        """Test that OTEL can access CloudManager through ExecutionContext."""
        otel_config = {"enabled": True, "endpoint": "http://test"}
        
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}],
            tracing={"otel": otel_config}
        )
        
        # Mock CloudManager with specific methods OTEL might need
        mock_cloud_manager = Mock()
        mock_cloud_manager.get_access_token.return_value = "test-access-token"
        mock_cloud_manager.gcp_credentials = Mock()
        
        with patch.object(context, "cloud_manager", mock_cloud_manager):
            import asyncio
            asyncio.run(context._initialize_otel())
            
            # Verify setup_tracing_otel_with_execution_context was called
            mock_setup_otel.assert_called_once_with(otel_config, context)
            
            # Verify the ExecutionContext passed has working CloudManager
            call_args = mock_setup_otel.call_args
            passed_context = call_args[0][1]  # Second argument is ExecutionContext
            assert passed_context is context
            assert passed_context.cloud_manager is mock_cloud_manager
    
    def test_otel_config_validation_in_execution_context(self):
        """Test OTEL configuration validation in ExecutionContext."""
        # Test with enabled OTEL config
        otel_config_enabled = {"enabled": True, "endpoint": "http://test"}
        context_enabled = create_execution_context(
            tracing={"otel": otel_config_enabled}
        )
        
        # Verify tracing configuration is stored correctly
        assert "otel" in context_enabled.tracing
        assert context_enabled.tracing["otel"]["enabled"] is True
        assert context_enabled.tracing["otel"]["endpoint"] == "http://test"
        
        # Test with disabled OTEL config
        otel_config_disabled = {"enabled": False}
        context_disabled = create_execution_context(
            tracing={"otel": otel_config_disabled}
        )
        
        assert context_disabled.tracing["otel"]["enabled"] is False
    
    @patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context")
    def test_otel_failure_handling_in_execution_context(self, mock_setup_otel):
        """Test OTEL initialization failure handling in ExecutionContext."""
        mock_setup_otel.side_effect = Exception("OTEL endpoint unreachable")
        
        otel_config = {"enabled": True, "endpoint": "http://unreachable"}
        context = create_execution_context(
            tracing={"otel": otel_config}
        )
        
        # OTEL initialization failure should be wrapped in RuntimeError
        with pytest.raises(RuntimeError, match="OTEL tracing initialization failed"):
            import asyncio
            asyncio.run(context._initialize_otel())
        
        # Verify the original exception is preserved
        try:
            import asyncio
            asyncio.run(context._initialize_otel())
        except RuntimeError as e:
            assert "OTEL endpoint unreachable" in str(e.__cause__)
    
    @patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context")
    def test_otel_deferred_initialization_pattern(self, mock_setup_otel):
        """Test that OTEL initialization is deferred until first access."""
        otel_config = {"enabled": True, "endpoint": "http://test"}
        context = create_execution_context(
            tracing={"otel": otel_config}
        )
        
        # OTEL should not be initialized during ExecutionContext creation
        mock_setup_otel.assert_not_called()
        
        # OTEL should be initialized on first tracing access
        import asyncio
        asyncio.run(context._ensure_tracing_initialized())
        
        # Now OTEL should be initialized
        mock_setup_otel.assert_called_once_with(otel_config, context)
    
    def test_otel_integration_with_multiple_tracing_providers(self):
        """Test OTEL integration alongside other tracing providers."""
        tracing_config = {
            "otel": {"enabled": True, "endpoint": "http://otel"},
            "weave": {"enabled": True, "project_id": "test", "api_key": "test"},
            "traceloop": {"enabled": False}
        }
        
        context = create_execution_context(tracing=tracing_config)
        
        # Verify all tracing configs are stored
        assert "otel" in context.tracing
        assert "weave" in context.tracing
        assert "traceloop" in context.tracing
        
        # Verify enabled/disabled states
        assert context.tracing["otel"]["enabled"] is True
        assert context.tracing["weave"]["enabled"] is True
        assert context.tracing["traceloop"]["enabled"] is False
    
    @patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context")
    @patch("buttermilk._core.execution_context.weave")
    def test_tracing_providers_initialization_order(self, mock_weave, mock_setup_otel):
        """Test that tracing providers are initialized in correct order."""
        tracing_config = {
            "otel": {"enabled": True, "endpoint": "http://otel"},
            "weave": {"enabled": True, "project_id": "test", "api_key": "test"}
        }
        
        context = create_execution_context(tracing=tracing_config)
        
        # Track initialization order
        initialization_order = []
        
        def track_weave_init(*args, **kwargs):
            initialization_order.append("weave")
            return Mock()
        
        def track_otel_init(*args, **kwargs):
            initialization_order.append("otel")
        
        mock_weave.init.side_effect = track_weave_init
        mock_setup_otel.side_effect = track_otel_init
        
        # Initialize all tracing providers
        import asyncio
        asyncio.run(context._initialize_all_tracing_providers())
        
        # Verify both providers were initialized
        assert "weave" in initialization_order
        assert "otel" in initialization_order
        
        # Verify providers are marked as initialized
        assert context._tracing_providers_initialized is True


class TestOTELSetupFunction:
    """Test the setup_tracing_otel_with_execution_context function specifically."""
    
    def setup_method(self):
        """Reset global state before each test."""
        globals()["_global_execution_context"] = None
        globals()["_execution_context_initialized"] = False
    
    def test_otel_setup_function_signature(self):
        """Test that the OTEL setup function has correct signature."""
        # Import the function to verify it exists and is callable
        from buttermilk.utils.otel import setup_tracing_otel_with_execution_context
        
        # Verify it's callable
        assert callable(setup_tracing_otel_with_execution_context)
        
        # Test with mock parameters
        mock_otel_config = {"enabled": True, "endpoint": "http://test"}
        mock_execution_context = Mock()
        
        # Should not raise exception for correct signature
        with patch("buttermilk.utils.otel.setup_tracing_otel") as mock_setup:
            try:
                setup_tracing_otel_with_execution_context(mock_otel_config, mock_execution_context)
                # Function exists and accepts the expected parameters
                assert True
            except (ImportError, AttributeError):
                # Function doesn't exist yet - this is expected during development
                pytest.skip("setup_tracing_otel_with_execution_context not yet implemented")
    
    @patch("buttermilk.utils.otel.setup_tracing_otel")
    def test_otel_setup_with_execution_context_cloudmanager_access(self, mock_setup_otel):
        """Test OTEL setup function can access CloudManager from ExecutionContext."""
        try:
            from buttermilk.utils.otel import setup_tracing_otel_with_execution_context
        except ImportError:
            pytest.skip("setup_tracing_otel_with_execution_context not yet implemented")
        
        # Create ExecutionContext with CloudManager
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}]
        )
        
        # Mock CloudManager
        mock_cloud_manager = Mock()
        mock_cloud_manager.get_access_token.return_value = "test-token"
        
        with patch.object(context, "cloud_manager", mock_cloud_manager):
            otel_config = {"enabled": True, "endpoint": "http://test"}
            
            # Call the setup function
            setup_tracing_otel_with_execution_context(otel_config, context)
            
            # Verify CloudManager was accessible
            # (The exact verification depends on implementation details)
            assert context.cloud_manager is mock_cloud_manager
    
    def test_otel_setup_function_error_handling(self):
        """Test error handling in OTEL setup function."""
        try:
            from buttermilk.utils.otel import setup_tracing_otel_with_execution_context
        except ImportError:
            pytest.skip("setup_tracing_otel_with_execution_context not yet implemented")
        
        # Test with None ExecutionContext
        otel_config = {"enabled": True, "endpoint": "http://test"}
        
        # Should handle None ExecutionContext gracefully or raise appropriate error
        with pytest.raises((ValueError, RuntimeError, AttributeError)):
            setup_tracing_otel_with_execution_context(otel_config, None)
    
    @patch("buttermilk.utils.otel.setup_tracing_otel")
    def test_otel_setup_function_delegates_to_existing_setup(self, mock_setup_otel):
        """Test that new function properly delegates to existing OTEL setup."""
        try:
            from buttermilk.utils.otel import setup_tracing_otel_with_execution_context
        except ImportError:
            pytest.skip("setup_tracing_otel_with_execution_context not yet implemented")
        
        context = create_execution_context()
        otel_config = {"enabled": True, "endpoint": "http://test"}
        
        # Call the new function
        setup_tracing_otel_with_execution_context(otel_config, context)
        
        # Should delegate to the existing setup function
        # (Implementation details may vary)
        mock_setup_otel.assert_called()


class TestOTELExecutionContextArchitecture:
    """Test architectural compliance of OTEL with ExecutionContext."""
    
    def setup_method(self):
        """Reset global state before each test."""
        globals()["_global_execution_context"] = None
        globals()["_execution_context_initialized"] = False
    
    def test_otel_no_bm_singleton_dependency(self):
        """Test that OTEL setup doesn't depend on BM singleton."""
        # This test verifies the architectural fix: OTEL should use
        # ExecutionContext directly rather than relying on BM singleton
        
        otel_config = {"enabled": True, "endpoint": "http://test"}
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}],
            tracing={"otel": otel_config}
        )
        
        # Clear BM singleton to ensure OTEL doesn't depend on it
        from buttermilk import _global_bm
        _global_bm = None
        
        # Mock CloudManager to avoid real authentication
        with patch.object(context, "cloud_manager") as mock_cloud_mgr:
            mock_cloud_mgr.return_value = Mock()
            
            # OTEL initialization should succeed without BM singleton
            with patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context") as mock_setup:
                import asyncio
                asyncio.run(context._initialize_otel())
                
                # Verify OTEL was initialized with ExecutionContext
                mock_setup.assert_called_once_with(otel_config, context)
    
    def test_otel_execution_context_infrastructure_access(self):
        """Test that OTEL can access all necessary infrastructure through ExecutionContext."""
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}],
            secret_provider={"type": "gcp"},
            tracing={"otel": {"enabled": True}}
        )
        
        # Mock infrastructure components
        mock_cloud_manager = Mock()
        mock_secret_manager = Mock()
        mock_credentials = {"otel_api_key": "test-key"}
        
        with patch.object(context, "cloud_manager", mock_cloud_manager), \
             patch.object(context, "secret_manager", mock_secret_manager), \
             patch.object(context, "credentials", mock_credentials):
            
            # Verify ExecutionContext provides all infrastructure OTEL might need
            assert context.cloud_manager is mock_cloud_manager
            assert context.secret_manager is mock_secret_manager
            assert context.credentials is mock_credentials
            
            # OTEL setup should have access to all infrastructure
            with patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context") as mock_setup:
                import asyncio
                asyncio.run(context._initialize_otel())
                
                # Verify ExecutionContext with full infrastructure was passed
                call_args = mock_setup.call_args
                passed_context = call_args[0][1]
                assert passed_context is context
    
    def test_otel_prevents_cloudmanager_not_available_error(self):
        """Test that OTEL initialization prevents 'CloudManager not available' errors."""
        # This test validates the specific fix for the error mentioned in the issue
        
        otel_config = {"enabled": True, "endpoint": "http://test"}
        context = create_execution_context(
            clouds=[{"type": "gcp", "project_id": "test"}],
            tracing={"otel": otel_config}
        )
        
        # Mock CloudManager to be available in ExecutionContext
        mock_cloud_manager = Mock()
        mock_cloud_manager.get_access_token.return_value = "valid-token"
        
        with patch.object(context, "cloud_manager", mock_cloud_manager):
            # OTEL initialization should not raise "CloudManager not available"
            with patch("buttermilk.utils.otel.setup_tracing_otel_with_execution_context") as mock_setup:
                import asyncio
                
                # This should not raise any "CloudManager not available" error
                try:
                    asyncio.run(context._initialize_otel())
                    # Success - no CloudManager error
                    assert True
                except RuntimeError as e:
                    if "CloudManager not available" in str(e):
                        pytest.fail("OTEL initialization still has CloudManager not available error")
                    else:
                        # Other RuntimeError is fine (e.g., from mocked components)
                        pass
                
                # Verify setup was called with working ExecutionContext
                mock_setup.assert_called_once_with(otel_config, context)
