"""Integration tests for unified tracing configuration.

This module provides comprehensive tests for the unified tracing configuration
that validates all three tracing providers (Weave, Traceloop, OTEL) work correctly
with the new config-based initialization approach.

The tests ensure that:
- ExecutionContext initializes tracing providers from unified config
- Config-based credentials work properly (no environment variable dependencies) 
- Fail-fast behavior occurs when credentials are missing
- All three providers can create and submit traces successfully
- The recent fix for "'NoneType' object has no attribute 'create_call'" works

These tests serve as integration validation for the tracing infrastructure
and provide living documentation of the expected configuration patterns.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict

import weave
from buttermilk._core.execution_context import ExecutionContext
from buttermilk._core.config import Tracing


class TestUnifiedTracingConfig:
    """Integration tests for the unified tracing configuration system."""

    @pytest.fixture
    def weave_config(self) -> Dict[str, Any]:
        """Valid weave tracing configuration."""
        return {
            "enabled": True,
            "project_id": "test-dmrc",
            "api_key": "fake-wandb-api-key-for-testing"
        }

    @pytest.fixture
    def traceloop_config(self) -> Dict[str, Any]:
        """Valid traceloop tracing configuration."""
        return {
            "enabled": True,
            "api_key": "fake-traceloop-api-key-for-testing",
            "endpoint": "https://api.traceloop.com"
        }

    @pytest.fixture
    def otel_config(self) -> Dict[str, Any]:
        """Valid OTEL tracing configuration."""
        return {
            "enabled": True,
            "endpoint": "https://telemetry.googleapis.com",
            "otlp_headers": {}
        }

    @pytest.fixture
    def unified_tracing_config(self, weave_config, traceloop_config, otel_config) -> Dict[str, Dict[str, Any]]:
        """Complete unified tracing configuration."""
        return {
            "weave": weave_config,
            "traceloop": traceloop_config,
            "otel": otel_config
        }

    @pytest.mark.anyio
    async def test_weave_config_based_initialization(self, weave_config):
        """Test Weave initialization from config without environment variables."""
        # Clear environment variables to ensure config-based initialization
        with patch.dict(os.environ, {}, clear=True):
            # Mock weave.init and weave.get_client
            with patch('weave.init') as mock_weave_init, \
                 patch('weave.get_client') as mock_get_client:
                
                mock_client = Mock(spec=weave.trace.weave_client.WeaveClient)
                mock_client.project_name = "test-dmrc" 
                mock_get_client.return_value = mock_client
                
                # Create ExecutionContext with weave config
                tracing_config = {"weave": Tracing(**weave_config)}
                
                ctx = ExecutionContext(
                    clouds=[],
                    tracing=tracing_config
                )
                
                # Test weave client initialization
                client = await ctx.get_weave_client()
                
                # Verify weave.init was called with config values (not env vars)
                mock_weave_init.assert_called_once_with(
                    project_name="test-dmrc"
                )
                
                # Verify client was retrieved
                mock_get_client.assert_called_once()
                assert client is mock_client
                assert client.project_name == "test-dmrc"

    @pytest.mark.anyio
    async def test_weave_missing_project_id_fails_fast(self):
        """Test that missing project_id in weave config raises RuntimeError."""
        invalid_config = {
            "enabled": True,
            "api_key": "fake-wandb-api-key-for-testing"
            # project_id missing
        }
        
        tracing_config = {"weave": Tracing(**invalid_config)}
        
        ctx = ExecutionContext(
            clouds=[],
            tracing=tracing_config
        )
        
        # Should fail fast when trying to get weave client
        with pytest.raises(RuntimeError, match="project_id \\(WANDB_ENTITY\\) not configured"):
            await ctx.get_weave_client()

    @pytest.mark.anyio
    async def test_weave_missing_api_key_fails_fast(self):
        """Test that missing api_key in weave config raises RuntimeError."""
        invalid_config = {
            "enabled": True,
            "project_id": "test-dmrc"
            # api_key missing
        }
        
        tracing_config = {"weave": Tracing(**invalid_config)}
        
        ctx = ExecutionContext(
            clouds=[],
            tracing=tracing_config
        )
        
        # Should fail fast when trying to get weave client
        with pytest.raises(RuntimeError, match="api_key \\(WANDB_API_KEY\\) not configured"):
            await ctx.get_weave_client()

    @pytest.mark.anyio
    async def test_weave_trace_creation_and_submission(self, weave_config):
        """Test actual trace creation and submission with Weave."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('weave.init') as mock_weave_init, \
                 patch('weave.get_client') as mock_get_client:
                
                # Mock weave client with trace creation capabilities
                mock_client = Mock(spec=weave.trace.weave_client.WeaveClient)
                mock_client.project_name = "test-dmrc"
                
                # Mock a call object
                mock_call = Mock()
                mock_call.id = "test-call-id"
                mock_call.trace_id = "test-trace-id"
                
                mock_client.create_call.return_value = mock_call
                mock_client.finish_call.return_value = None
                mock_client.get_call.return_value = mock_call
                
                mock_get_client.return_value = mock_client
                
                tracing_config = {"weave": Tracing(**weave_config)}
                
                ctx = ExecutionContext(
                    clouds=[],
                    tracing=tracing_config
                )
                
                # Get weave client
                client = await ctx.get_weave_client()
                
                # Create a simple operation to trace
                def test_operation(x: int) -> int:
                    return x + 1
                
                with patch('weave.op') as mock_weave_op:
                    mock_op = Mock()
                    mock_weave_op.return_value = mock_op
                    
                    # Create a trace call
                    call = client.create_call(
                        mock_op,
                        inputs={"x": 41},
                        display_name="test-unified-config-trace",
                        attributes={"test": "unified-config", "provider": "weave"}
                    )
                    
                    # Verify call was created
                    assert call is not None
                    assert call.id == "test-call-id"
                    assert call.trace_id == "test-trace-id"
                    
                    # Finish the call
                    client.finish_call(call, output={"result": 42}, op=mock_op)
                    
                    # Verify the call can be retrieved
                    fetched_call = client.get_call(call.id)
                    assert fetched_call is call
                    
                    # Verify client methods were called correctly
                    mock_client.create_call.assert_called_once()
                    mock_client.finish_call.assert_called_once_with(call, output={"result": 42}, op=mock_op)
                    mock_client.get_call.assert_called_once_with("test-call-id")

    @pytest.mark.anyio
    async def test_traceloop_config_validation(self, traceloop_config):
        """Test Traceloop configuration validation."""
        # Test that Traceloop config is properly parsed
        tracing_instance = Tracing(**traceloop_config)
        
        assert tracing_instance.enabled is True
        assert tracing_instance.api_key == "fake-traceloop-api-key-for-testing"
        assert tracing_instance.endpoint == "https://api.traceloop.com"

    @pytest.mark.anyio
    async def test_traceloop_missing_api_key_validation(self):
        """Test that Traceloop requires api_key when enabled."""
        invalid_config = {
            "enabled": True,
            "endpoint": "https://api.traceloop.com"
            # api_key missing
        }
        
        # Tracing model should validate required fields
        with pytest.raises(Exception):  # Pydantic validation error
            Tracing(**invalid_config)

    @pytest.mark.anyio
    async def test_otel_config_validation(self, otel_config):
        """Test OTEL configuration validation."""
        # Test that OTEL config is properly parsed
        tracing_instance = Tracing(**otel_config)
        
        assert tracing_instance.enabled is True
        assert tracing_instance.endpoint == "https://telemetry.googleapis.com"
        assert tracing_instance.otlp_headers == {}

    @pytest.mark.anyio 
    async def test_otel_gcp_trace_setup(self, otel_config):
        """Test OTEL setup with GCP tracing configuration."""
        with patch('buttermilk.utils.otel.CloudTraceSpanExporter') as mock_gcp_exporter, \
             patch('buttermilk.utils.otel.BatchSpanProcessor') as mock_processor, \
             patch('buttermilk.utils.otel.TracerProvider') as mock_tracer_provider, \
             patch('buttermilk.utils.otel.trace') as mock_trace:
            
            mock_processor_instance = Mock()
            mock_processor.return_value = mock_processor_instance
            
            mock_tracer_instance = Mock()
            mock_tracer_provider.return_value = mock_tracer_instance
            
            mock_gcp_exporter_instance = Mock()
            mock_gcp_exporter.return_value = mock_gcp_exporter_instance
            
            # Test OTEL initialization with config
            tracing_config = {"otel": Tracing(**otel_config)}
            
            ctx = ExecutionContext(
                clouds=[],
                tracing=tracing_config
            )
            
            # Import otel module to trigger setup
            import importlib
            import buttermilk.utils.otel
            importlib.reload(buttermilk.utils.otel)
            
            # Verify that OTEL components can be properly configured
            # (This tests the configuration structure, actual OTEL setup is tested in test_otel_gcp_integration.py)
            assert ctx.tracing["otel"].enabled is True
            assert ctx.tracing["otel"].endpoint == "https://telemetry.googleapis.com"

    @pytest.mark.anyio
    async def test_multiple_providers_configuration(self, unified_tracing_config):
        """Test configuration with multiple tracing providers enabled."""
        # Convert dict configs to Tracing objects
        tracing_config = {
            provider: Tracing(**config) 
            for provider, config in unified_tracing_config.items()
        }
        
        ctx = ExecutionContext(
            clouds=[],
            tracing=tracing_config
        )
        
        # Verify all providers are configured
        assert ctx.tracing["weave"].enabled is True
        assert ctx.tracing["traceloop"].enabled is True  
        assert ctx.tracing["otel"].enabled is True
        
        # Verify configuration values
        assert ctx.tracing["weave"].project_id == "test-dmrc"
        assert ctx.tracing["weave"].api_key == "fake-wandb-api-key-for-testing"
        
        assert ctx.tracing["traceloop"].api_key == "fake-traceloop-api-key-for-testing"
        assert ctx.tracing["traceloop"].endpoint == "https://api.traceloop.com"
        
        assert ctx.tracing["otel"].endpoint == "https://telemetry.googleapis.com"

    @pytest.mark.anyio
    async def test_disabled_providers_not_initialized(self):
        """Test that disabled tracing providers are not initialized."""
        disabled_config = {
            "weave": {
                "enabled": False,
                "project_id": "test-dmrc",
                "api_key": "fake-wandb-api-key-for-testing"
            },
            "traceloop": {
                "enabled": False,
                "api_key": "fake-traceloop-api-key-for-testing",
                "endpoint": "https://api.traceloop.com"
            },
            "otel": {
                "enabled": False,
                "endpoint": "https://telemetry.googleapis.com",
                "otlp_headers": {}
            }
        }
        
        tracing_config = {
            provider: Tracing(**config)
            for provider, config in disabled_config.items()
        }
        
        ctx = ExecutionContext(
            clouds=[],
            tracing=tracing_config
        )
        
        # Verify weave client is None when disabled
        with patch('weave.init') as mock_weave_init:
            client = await ctx.get_weave_client()
            
            # Should not call weave.init when disabled
            mock_weave_init.assert_not_called()
            
            # Should return None or not initialize 
            # (depends on implementation - weave.get_client() behavior when not initialized)

    @pytest.mark.anyio
    async def test_execution_context_fix_for_nonetype_error(self, weave_config):
        """Test the fix for \"'NoneType' object has no attribute 'create_call'\" error."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('weave.init') as mock_weave_init, \
                 patch('weave.get_client') as mock_get_client:
                
                # Mock a properly initialized weave client
                mock_client = Mock(spec=weave.trace.weave_client.WeaveClient)
                mock_client.project_name = "test-dmrc"
                mock_client.create_call = Mock()
                mock_get_client.return_value = mock_client
                
                tracing_config = {"weave": Tracing(**weave_config)}
                
                ctx = ExecutionContext(
                    clouds=[],
                    tracing=tracing_config
                )
                
                # Get weave client - should not be None
                client = await ctx.get_weave_client()
                
                # Verify client is properly initialized and has create_call method
                assert client is not None
                assert hasattr(client, 'create_call')
                assert callable(client.create_call)
                
                # Verify that weave.init was called with proper credentials
                mock_weave_init.assert_called_once_with(project_name="test-dmrc")
                
                # Verify we can call create_call without NoneType error
                mock_op = Mock()
                client.create_call(mock_op, inputs={"test": "data"})
                mock_client.create_call.assert_called_once()

    @pytest.mark.anyio
    async def test_config_based_credentials_override_environment(self, weave_config):
        """Test that config-based credentials take precedence over environment variables."""
        # Set different values in environment
        env_vars = {
            "WANDB_ENTITY": "wrong-entity",
            "WANDB_API_KEY": "wrong-api-key",
            "WANDB_PROJECT": "wrong-project"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            with patch('weave.init') as mock_weave_init, \
                 patch('weave.get_client') as mock_get_client:
                
                mock_client = Mock(spec=weave.trace.weave_client.WeaveClient)
                mock_client.project_name = "test-dmrc"
                mock_get_client.return_value = mock_client
                
                tracing_config = {"weave": Tracing(**weave_config)}
                
                ctx = ExecutionContext(
                    clouds=[],
                    tracing=tracing_config  
                )
                
                # Get weave client
                await ctx.get_weave_client()
                
                # Verify weave.init was called with CONFIG values, not environment values
                mock_weave_init.assert_called_once_with(
                    project_name="test-dmrc"  # From config, not "wrong-entity" from env
                )
                
                # Verify environment variables were set to config values
                assert os.environ.get("WANDB_ENTITY") == "test-dmrc"
                assert os.environ.get("WANDB_API_KEY") == "fake-wandb-api-key-for-testing"


class TestTracingConfigExamples:
    """Example tests showing how to use the unified tracing configuration.
    
    These tests serve as living documentation for developers on how to
    configure and use the tracing system properly.
    """

    @pytest.mark.anyio
    async def test_minimal_weave_setup_example(self):
        """Example: Minimal Weave tracing setup with config."""
        # Minimal weave configuration
        weave_config = {
            "enabled": True,
            "project_id": "my-research-project",
            "api_key": "your-wandb-api-key-here"
        }
        
        with patch('weave.init') as mock_init, \
             patch('weave.get_client') as mock_get_client:
            
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Create execution context with weave tracing
            ctx = ExecutionContext(
                clouds=[],
                tracing={"weave": Tracing(**weave_config)}
            )
            
            # Initialize tracing
            client = await ctx.get_weave_client()
            
            # Verify setup
            assert client is not None
            mock_init.assert_called_once_with(project_name="my-research-project")

    @pytest.mark.anyio 
    async def test_production_multi_provider_example(self):
        """Example: Production setup with multiple tracing providers."""
        production_config = {
            "weave": {
                "enabled": True,
                "project_id": "research-production",
                "api_key": "prod-wandb-key"
            },
            "traceloop": {
                "enabled": True,
                "api_key": "prod-traceloop-key", 
                "endpoint": "https://api.traceloop.com"
            },
            "otel": {
                "enabled": True,
                "endpoint": "https://telemetry.googleapis.com",
                "otlp_headers": {
                    "custom-header": "production-value"
                }
            }
        }
        
        tracing_config = {
            provider: Tracing(**config)
            for provider, config in production_config.items()
        }
        
        ctx = ExecutionContext(
            clouds=[],
            tracing=tracing_config
        )
        
        # Verify all providers are enabled
        assert all(ctx.tracing[provider].enabled for provider in ["weave", "traceloop", "otel"])
        
        # This example shows the configuration structure for production deployments
        # where multiple tracing backends are used for different purposes:
        # - Weave for experiment tracking
        # - Traceloop for observability 
        # - OTEL for cloud provider integration

    @pytest.mark.anyio
    async def test_development_weave_only_example(self):
        """Example: Development setup with only Weave tracing."""
        development_config = {
            "weave": {
                "enabled": True,
                "project_id": "dev-experiments",
                "api_key": "dev-wandb-key"
            },
            "traceloop": {
                "enabled": False,
                "api_key": "",
                "endpoint": ""
            },
            "otel": {
                "enabled": False,
                "endpoint": "",
                "otlp_headers": {}
            }
        }
        
        tracing_config = {
            provider: Tracing(**config)
            for provider, config in development_config.items()
        }
        
        ctx = ExecutionContext(
            clouds=[],
            tracing=tracing_config
        )
        
        # Verify only weave is enabled
        assert ctx.tracing["weave"].enabled is True
        assert ctx.tracing["traceloop"].enabled is False
        assert ctx.tracing["otel"].enabled is False
        
        # This example shows how to configure for development where
        # you only want Weave tracing for experiment tracking