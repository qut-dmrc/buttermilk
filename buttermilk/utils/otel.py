"""Configures OpenTelemetry (OTEL) tracing for the Buttermilk framework.

This module sets up global OpenTelemetry tracing, specifically configured to
export trace data to Weights & Biases (W&B) using the OTLP (OpenTelemetry Protocol)
gRPC exporter. The setup is performed automatically when this module is imported.

It relies on credentials (WANDB_API_KEY, WANDB_PROJECT) being available via
the global Buttermilk instance (`bm.credentials`). If setup fails, a warning
is logged, and tracing may not function.

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
    The `os.environ` import was missing; it's added for completeness as
    `os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]` is used.
"""
import base64
import os  # Added for os.environ usage

from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# Import trace_sdk at the top level for clarity, though original was inline
from opentelemetry.sdk import trace as trace_sdk

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core.log import logger

# --- OpenTelemetry Tracing Setup for Weights & Biases ---

WANDB_BASE_URL = "https://trace.wandb.ai"
"""Base URL for Weights & Biases tracing services."""

OTEL_EXPORTER_OTLP_ENDPOINT = f"{WANDB_BASE_URL}/otel/v1/traces"
"""The full OTLP endpoint URL where trace data will be sent for W&B."""

# Set the OTLP endpoint as an environment variable, which some OTel components might read.
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = OTEL_EXPORTER_OTLP_ENDPOINT

try:
    # Retrieve necessary credentials from the global Buttermilk instance.
    # These are expected to be populated during Buttermilk initialization (e.g., from secrets).
    creds = bm.credentials
    if not creds or "WANDB_API_KEY" not in creds or "WANDB_PROJECT" not in creds:
        raise KeyError("W&B API key or project information not found in bm.credentials.")

    # Prepare authentication header for W&B OTLP exporter.
    # The AUTH string is typically "api:<YOUR_WANDB_API_KEY>".
    auth_string = f"api:{creds['WANDB_API_KEY']}"
    auth_header_value = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

    OTEL_EXPORTER_OTLP_HEADERS = {
        "Authorization": f"Basic {auth_header_value}", # Basic authentication header
        "project_id": creds["WANDB_PROJECT"],          # W&B Project ID for trace grouping
    }
    """Headers required for the OTLP exporter, including authorization for W&B
    and the W&B project ID.
    """

    # Initialize the OpenTelemetry SDK's TracerProvider.
    # This provider manages the creation of tracers.
    tracer_provider = trace_sdk.TracerProvider()

    # Configure the OTLP Span Exporter to send traces to W&B.
    otlp_exporter = OTLPSpanExporter(
        endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
        headers=OTEL_EXPORTER_OTLP_HEADERS,
        # Other options like `timeout` or `compression` can be set here if needed.
    )

    span_processor = BatchSpanProcessor(otlp_exporter)

    # Add the exporter to the tracer provider
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    logger.info(
        "OpenTelemetry (OTEL) tracing components initialized for W&B export. "
        "Ensure a SpanProcessor and global provider are set if direct OTEL API usage is intended. "
        "Weave integration might handle further OTEL setup on its first access."
    )
except KeyError as e_key:
    logger.warning(f"OpenTelemetry tracing setup for W&B skipped: Missing required credential '{e_key.args[0]}' in bm.credentials.")
except Exception as e_otel: # Catch any other errors during setup
    logger.warning(f"Error during OpenTelemetry tracing setup for W&B: {e_otel!s}. OTEL tracing might not function.", exc_info=True)
