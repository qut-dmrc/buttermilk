"""Tests for OpenTelemetry GCP integration."""
import os
import unittest
from unittest.mock import MagicMock, patch


class TestOtelGCPIntegration(unittest.TestCase):
    """Test OpenTelemetry GCP integration configuration."""

    @patch("buttermilk.utils.otel.trace")
    @patch("buttermilk.utils.otel.trace_sdk.TracerProvider")
    @patch("buttermilk.utils.otel.OTLPSpanExporter")
    @patch("buttermilk.utils.otel.BatchSpanProcessor")
    @patch("buttermilk.utils.otel.CloudTraceSpanExporter")
    @patch("buttermilk.utils.otel.bm")
    @patch("buttermilk.utils.otel.logger")
    def test_otel_gcp_and_wandb_exporters_configured(
        self,
        mock_logger,
        mock_bm,
        mock_cloud_trace_exporter,
        mock_batch_processor,
        mock_otlp_exporter,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test that both GCP and W&B exporters are configured when credentials are available."""
        # Setup mocks
        mock_tracer_instance = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        # Mock W&B credentials
        mock_bm.credentials = {
            "WANDB_API_KEY": "test-wandb-key",
            "WANDB_PROJECT": "test-project",
        }

        # Mock GCP config
        mock_gcp_config = MagicMock()
        mock_gcp_config.project_id = "test-gcp-project"
        mock_bm.cloud_cfg.get_cloud_config.return_value = mock_gcp_config

        # Mock processors
        mock_wandb_processor = MagicMock()
        mock_gcp_processor = MagicMock()
        mock_batch_processor.side_effect = [mock_wandb_processor, mock_gcp_processor]

        # Import the module to trigger setup
        import importlib

        import buttermilk.utils.otel
        importlib.reload(buttermilk.utils.otel)

        # Verify W&B exporter was created
        mock_otlp_exporter.assert_called_once()
        otlp_call_args = mock_otlp_exporter.call_args
        assert otlp_call_args[1]["endpoint"] == "https://trace.wandb.ai/otel/v1/traces"
        assert "Authorization" in otlp_call_args[1]["headers"]
        assert otlp_call_args[1]["headers"]["project_id"] == "test-project"

        # Verify GCP exporter was created
        mock_cloud_trace_exporter.assert_called_once_with(project_id="test-gcp-project")

        # Verify both processors were added
        assert mock_tracer_instance.add_span_processor.call_count == 2
        mock_tracer_instance.add_span_processor.assert_any_call(mock_wandb_processor)
        mock_tracer_instance.add_span_processor.assert_any_call(mock_gcp_processor)

        # Verify tracer provider was set
        mock_trace.set_tracer_provider.assert_called_once_with(mock_tracer_instance)

        # Verify success log
        mock_logger.info.assert_any_call(
            "OpenTelemetry (OTEL) tracing initialized with exporters: W&B, GCP (project: test-gcp-project). "
            "Traces will be sent to configured destinations.",
        )

    @patch("buttermilk.utils.otel.trace")
    @patch("buttermilk.utils.otel.trace_sdk.TracerProvider")
    @patch("buttermilk.utils.otel.CloudTraceSpanExporter")
    @patch("buttermilk.utils.otel.BatchSpanProcessor")
    @patch("buttermilk.utils.otel.bm")
    @patch("buttermilk.utils.otel.logger")
    @patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "env-gcp-project"})
    def test_otel_gcp_from_env_variable(
        self,
        mock_logger,
        mock_bm,
        mock_batch_processor,
        mock_cloud_trace_exporter,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test that GCP exporter uses GOOGLE_CLOUD_PROJECT env var when config not available."""
        # Setup mocks
        mock_tracer_instance = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        # No W&B credentials
        mock_bm.credentials = {}

        # No GCP config
        mock_bm.cloud_cfg.get_cloud_config.return_value = None

        # Mock processor
        mock_gcp_processor = MagicMock()
        mock_batch_processor.return_value = mock_gcp_processor

        # Import the module to trigger setup
        import importlib

        import buttermilk.utils.otel
        importlib.reload(buttermilk.utils.otel)

        # Verify GCP exporter was created with env var project
        mock_cloud_trace_exporter.assert_called_once_with(project_id="env-gcp-project")

        # Verify processor was added
        mock_tracer_instance.add_span_processor.assert_called_once_with(mock_gcp_processor)

        # Verify tracer provider was set
        mock_trace.set_tracer_provider.assert_called_once_with(mock_tracer_instance)

    @patch("buttermilk.utils.otel.trace")
    @patch("buttermilk.utils.otel.trace_sdk.TracerProvider")
    @patch("buttermilk.utils.otel.bm")
    @patch("buttermilk.utils.otel.logger")
    @patch.dict(os.environ, {}, clear=True)
    def test_otel_no_exporters_configured(
        self,
        mock_logger,
        mock_bm,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test warning when no exporters can be configured."""
        # Setup mocks
        mock_tracer_instance = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        # No credentials or configs
        mock_bm.credentials = {}
        mock_bm.cloud_cfg.get_cloud_config.return_value = None

        # Clear GOOGLE_CLOUD_PROJECT from env
        if "GOOGLE_CLOUD_PROJECT" in os.environ:
            del os.environ["GOOGLE_CLOUD_PROJECT"]

        # Import the module to trigger setup
        import importlib

        import buttermilk.utils.otel
        importlib.reload(buttermilk.utils.otel)

        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "No OpenTelemetry exporters were configured. Tracing may not function properly.",
        )

        # Verify tracer provider was still set (even with no exporters)
        mock_trace.set_tracer_provider.assert_called_once_with(mock_tracer_instance)


class TestCloudPyGCPTracing(unittest.TestCase):
    """Test cloud.py GCP tracing setup."""

    @patch("buttermilk._core.cloud.logger")
    def test_setup_google_tracing_with_correct_imports(self, mock_logger):
        """Test that _setup_google_tracing uses correct import paths."""
        from buttermilk._core.cloud import CloudService

        cloud_service = CloudService()

        # Mock the imports
        with patch("buttermilk._core.cloud.CloudTraceSpanExporter") as mock_trace_exporter, \
             patch("buttermilk._core.cloud.CloudMonitoringMetricsExporter") as mock_metrics_exporter, \
             patch("buttermilk._core.cloud.CloudLoggingExporter") as mock_logs_exporter, \
             patch("buttermilk._core.cloud.Traceloop") as mock_traceloop:

            # Create mock instances
            mock_trace_instance = MagicMock()
            mock_metrics_instance = MagicMock()
            mock_logs_instance = MagicMock()

            mock_trace_exporter.return_value = mock_trace_instance
            mock_metrics_exporter.return_value = mock_metrics_instance
            mock_logs_exporter.return_value = mock_logs_instance

            # Call the method
            cloud_service._setup_google_tracing()

            # Verify exporters were created
            mock_trace_exporter.assert_called_once()
            mock_metrics_exporter.assert_called_once()
            mock_logs_exporter.assert_called_once()

            # Verify Traceloop was initialized with correct exporters
            mock_traceloop.init.assert_called_once_with(
                app_name="buttermilk",
                exporter=mock_trace_instance,
                metrics_exporter=mock_metrics_instance,
                logging_exporter=mock_logs_instance,
            )

            # Verify success log
            mock_logger.info.assert_called_with("Initialized Google Cloud tracing")
