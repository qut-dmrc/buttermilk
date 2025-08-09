"""Configures OpenTelemetry (OTEL) tracing for the Buttermilk framework.

This module sets up global OpenTelemetry tracing, configured to export trace data to:
1. Weights & Biases (W&B) using the OTLP (OpenTelemetry Protocol) gRPC exporter
2. Google Cloud Platform (GCP) using the Cloud Trace exporter

The setup is performed automatically when this module is imported.

It relies on credentials being available via the global Buttermilk instance:
- For W&B: WANDB_API_KEY, WANDB_PROJECT in `bm.credentials`
- For GCP: Uses Application Default Credentials or environment variables

Key Constants:
    WANDB_BASE_URL (str): Base URL for Weights & Biases tracing.
    OTEL_EXPORTER_OTLP_ENDPOINT (str): The OTLP endpoint URL for W&B traces.
    OTEL_EXPORTER_OTLP_HEADERS (dict): Headers for OTLP exporter, including
        authentication and W&B project ID.

Note:
    This module primarily executes configuration logic upon import and does not
    define reusable public functions or classes for direct invocation beyond setup.
    Actual trace creation (spans, etc.) would use standard OpenTelemetry APIs
    elsewhere in the codebase, relying on this global setup.

"""
import base64
import os  # Added for os.environ usage

from opentelemetry import trace
from opentelemetry.exporter.cloud_logging import CloudLoggingExporter
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Import trace_sdk at the top level for clarity, though original was inline
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from traceloop.sdk import Traceloop

# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
# Buttermilk core imports
from buttermilk._core.config import Tracing
from buttermilk._core.log import logger

# --- OpenTelemetry Tracing benefits from Traceloop enhancements? ---

# --- OpenTelemetry Tracing Setup for GCP collector ---
def setup_tracing(tracing_cfg: Tracing) -> None:
    """Set up Google Cloud tracing with Traceloop.

    Args:
        tracing_cfg: Tracing configuration

    Raises:
        RuntimeError: If GCP project ID is not found

    """
    if not tracing_cfg.enabled:
        return

    # Configure the GCP Cloud Trace Span Exporter
    trace_exporter = CloudTraceSpanExporter()
    metrics_exporter = CloudMonitoringMetricsExporter()
    logs_exporter = CloudLoggingExporter()

    Traceloop.init(
        app_name="buttermilk",
        exporter=trace_exporter,
        metrics_exporter=metrics_exporter,
        logging_exporter=logs_exporter,
    )
    logger.info("Initialized Google Cloud tracing")


# --- OpenTelemetry Tracing Setup for Weights & Biases ---
# This is not currently used.
def setup_wandb_otel_tracing() -> None:
    # Initialize the OpenTelemetry SDK's TracerProvider.
    # This provider manages the creation of tracers.
    tracer_provider = trace_sdk.TracerProvider()

    WANDB_BASE_URL = "https://trace.wandb.ai"
    """Base URL for Weights & Biases tracing services."""

    OTEL_EXPORTER_OTLP_ENDPOINT = f"{WANDB_BASE_URL}/otel/v1/traces"
    """The full OTLP endpoint URL where trace data will be sent for W&B."""

    # Set the OTLP endpoint as an environment variable, which some OTel components might read.
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = OTEL_EXPORTER_OTLP_ENDPOINT

    # Track which exporters were successfully configured
    exporters_configured = []

    # Configure W&B exporter
    try:
        # Retrieve necessary credentials from the global Buttermilk instance.
        # These are expected to be populated during Buttermilk initialization (e.g., from secrets).
        from buttermilk._core.dmrc import get_bm

        bm = get_bm()
        creds = bm.credentials

        wandb_api_key = os.getenv("WANDB_API_KEY") or creds["WANDB_API_KEY"]
        wandb_project = os.getenv("WANDB_PROJECT") or creds["WANDB_PROJECT"]
        wandb_entity = os.getenv("WANDB_ENTITY") or creds["WANDB_ENTITY"]

        # Prepare authentication header for W&B OTLP exporter.
        # The AUTH string is typically "api:<YOUR_WANDB_API_KEY>".
        auth_string = f"api:{wandb_api_key}"
        auth_header_value = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

        # Headers required for the OTLP exporter, including authorization for W&B
        # and the W&B project ID.
        OTEL_EXPORTER_OTLP_HEADERS = {
            "Authorization": f"Basic {auth_header_value}",  # Basic authentication header
            "project_id": wandb_project,           # W&B Project ID for trace grouping
        }

        # Configure the OTLP Span Exporter to send traces to W&B.
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
            headers=OTEL_EXPORTER_OTLP_HEADERS,
            # Other options like `timeout` or `compression` can be set here if needed.
        )

        wandb_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(wandb_processor)
        exporters_configured.append("W&B")
        logger.info("OpenTelemetry W&B exporter configured successfully")
    except Exception as e_wandb:
        logger.warning(f"Error configuring W&B exporter: {e_wandb}")

    # Set the tracer provider globally
    trace.set_tracer_provider(tracer_provider)
