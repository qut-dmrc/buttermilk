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
import logging
import os
import urllib  # Added for os.environ usage

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor
from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

# Import trace_sdk at the top level for clarity, though original was inline
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from buttermilk import logger

# Autogen imports (primarily for type hints and base classes/interfaces used in methods)
# Buttermilk core imports
from buttermilk._core.config import FatalError, Tracing

"""Base URL for Weights & Biases tracing services."""
WANDB_BASE_URL = "https://trace.wandb.ai"


import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
import grpc
from google.auth.transport.grpc import AuthMetadataPlugin

from buttermilk import get_bm


def setup_tracing_otel(tracing_cfg: Tracing) -> None:
    """Legacy OTEL setup using BM singleton. Use setup_tracing_otel_with_execution_context instead."""
    # Initialize OpenTelemetry with OTLP exporters

    bm = get_bm()
    creds = bm.gcp_credentials


def setup_tracing_otel_with_execution_context(tracing_cfg: Tracing, execution_context) -> None:
    """Initialize OpenTelemetry with OTLP exporters using ExecutionContext infrastructure."""
    # Get credentials from ExecutionContext instead of BM singleton
    creds = execution_context.gcp_credentials
    
    # Use project_id from tracing config, or fallback to GOOGLE_CLOUD_PROJECT env var
    project_id = tracing_cfg.project_id
    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id is None:
            raise RuntimeError("OTEL tracing requires a project_id but none found in config or GOOGLE_CLOUD_PROJECT environment variable")

    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"gcp.project_id={project_id}"
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = project_id
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = tracing_cfg.endpoint

    # Request used to refresh credentials upon expiry
    request = google.auth.transport.requests.Request()

    # Supply the request and credentials to AuthMetadataPlugin
    # AuthMeatadataPlugin inserts credentials into each request
    auth_metadata_plugin = AuthMetadataPlugin(credentials=creds, request=request)

    # Initialize gRPC channel credentials using the AuthMetadataPlugin
    channel_creds = grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(),
        grpc.metadata_call_credentials(auth_metadata_plugin),
    )

    # Initialize the OTLP gRPC or http exporter
    otlp_grpc_exporter = OTLPSpanExporter(credentials=channel_creds)

    # Initialize OpenTelemetry TracerProvider
    # Set up the tracer provider
    provider = TracerProvider()

    # Instrument libraries (only if not already instrumented)
    if not OpenAIInstrumentor().is_instrumented_by_opentelemetry:
        OpenAIInstrumentor().instrument(tracer_provider=provider)
    if not GoogleGenerativeAiInstrumentor().is_instrumented_by_opentelemetry:
        GoogleGenerativeAiInstrumentor().instrument(tracer_provider=provider)
    if not ChromaInstrumentor().is_instrumented_by_opentelemetry:
        ChromaInstrumentor().instrument(tracer_provider=provider)
    if not VertexAIInstrumentor().is_instrumented_by_opentelemetry:
        VertexAIInstrumentor().instrument(tracer_provider=provider)
    if not AnthropicInstrumentor().is_instrumented_by_opentelemetry:
        AnthropicInstrumentor().instrument(tracer_provider=provider)

    # Configure LoggingInstrumentor to exclude debug logs from traces
    # NOTE: set_logging_format=False prevents calling logging.basicConfig()
    # which would add unwanted StreamHandler to root logger, conflicting with our console setup
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument(
            tracer_provider=provider,
            set_logging_format=False,  # Don't override our logging setup
            log_level=logging.INFO,  # Only include INFO+ logs in OTEL traces
        )

    provider.add_span_processor(BatchSpanProcessor(otlp_grpc_exporter))

    # Configure OpenTelemetry tracing API with the initialized tracer provider
    trace.set_tracer_provider(provider)
    logger.info("Initialized tracing with Google Cloud using ExecutionContext infrastructure")
    
    # Use project_id from tracing config, or fallback to GOOGLE_CLOUD_PROJECT env var
    project_id = tracing_cfg.project_id
    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id is None:
            raise RuntimeError("OTEL tracing requires a project_id but none found in config or GOOGLE_CLOUD_PROJECT environment variable")

    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"gcp.project_id={project_id}"
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = project_id
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = tracing_cfg.endpoint

    # Request used to refresh credentials upon expiry
    request = google.auth.transport.requests.Request()

    # Supply the request and credentials to AuthMetadataPlugin
    # AuthMeatadataPlugin inserts credentials into each request
    auth_metadata_plugin = AuthMetadataPlugin(credentials=creds, request=request)

    # Initialize gRPC channel credentials using the AuthMetadataPlugin
    channel_creds = grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(),
        grpc.metadata_call_credentials(auth_metadata_plugin),
    )

    # Initialize the OTLP gRPC or http exporter
    otlp_grpc_exporter = OTLPSpanExporter(credentials=channel_creds)

    # Initialize OpenTelemetry TracerProvider
    # Set up the tracer provider
    provider = TracerProvider()

    # Instrument libraries (only if not already instrumented)
    if not OpenAIInstrumentor().is_instrumented_by_opentelemetry:
        OpenAIInstrumentor().instrument(tracer_provider=provider)
    if not GoogleGenerativeAiInstrumentor().is_instrumented_by_opentelemetry:
        GoogleGenerativeAiInstrumentor().instrument(tracer_provider=provider)
    if not ChromaInstrumentor().is_instrumented_by_opentelemetry:
        ChromaInstrumentor().instrument(tracer_provider=provider)
    if not VertexAIInstrumentor().is_instrumented_by_opentelemetry:
        VertexAIInstrumentor().instrument(tracer_provider=provider)
    if not AnthropicInstrumentor().is_instrumented_by_opentelemetry:
        AnthropicInstrumentor().instrument(tracer_provider=provider)

    # Configure LoggingInstrumentor to exclude debug logs from traces
    # NOTE: set_logging_format=False prevents calling logging.basicConfig()
    # which would add unwanted StreamHandler to root logger, conflicting with our console setup
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument(
            tracer_provider=provider,
            set_logging_format=False,  # Don't override our logging setup
            log_level=logging.INFO,  # Only include INFO+ logs in OTEL traces
        )

    provider.add_span_processor(BatchSpanProcessor(otlp_grpc_exporter))

    # Configure OpenTelemetry tracing API with the initialized tracer provider
    trace.set_tracer_provider(provider)
    logger.info("Initialized tracing with Google Cloud")


# --- OpenTelemetry Tracing Setup for Traceloop ---
def setup_traceloop_otel() ->  OTLPHttpSpanExporter | None:
    """Initialize Traceloop for OpenTelemetry tracing."""

    try:
        bm = get_bm()
        creds = bm.credentials

        traceloop_api_key = os.getenv("TRACELOOP_API_KEY") or creds["TRACELOOP_API_KEY"]
        traceloop_base_url = os.getenv("TRACELOOP_BASE_URL") or creds["TRACELOOP_BASE_URL"]
        traceloop_endpoint = f"{traceloop_base_url}/v1/traces"
        traceloop_auth_header = urllib.parse.quote(f"Bearer {traceloop_api_key}")
        # Do not URL-encode Authorization headers
        traceloop_headers = {"Authorization": f"Bearer {traceloop_api_key}"}
        traceloop_exporter = OTLPHttpSpanExporter(endpoint=traceloop_endpoint, headers=traceloop_headers)
        return traceloop_exporter

    except Exception as e_traceloop:
        logger.warning("Error configuring traceloop exporter", error=e_traceloop)
        return None


# --- OpenTelemetry Tracing Setup for Weights & Biases ---
def setup_wandb_otel_tracing() -> OTLPSpanExporter | None:
    # set the full OTLP endpoint URL where trace data will be sent for W&B.
    wandb_endpoint = f"{WANDB_BASE_URL}/otel/v1/traces"

    # Configure W&B exporter
    try:
        # Retrieve necessary credentials from the global Buttermilk instance.
        # These are expected to be populated during Buttermilk initialization (e.g., from secrets).
        from buttermilk import get_bm

        bm = get_bm()
        creds = bm.credentials

        wandb_api_key = os.getenv("WANDB_API_KEY") or creds["WANDB_API_KEY"]
        wandb_project = os.getenv("WANDB_PROJECT") or creds["WANDB_PROJECT"]
        wandb_entity = os.getenv("WANDB_ENTITY") or creds["WANDB_ENTITY"]
        if not (wandb_api_key and wandb_project and wandb_entity):
            raise FatalError(
                "W&B tracing is enabled but missing required credentials: "
                "WANDB_API_KEY, WANDB_PROJECT, or WANDB_ENTITY.",
            )

        # Prepare authentication header for W&B OTLP exporter.
        # The AUTH string is typically "api:<YOUR_WANDB_API_KEY>".
        auth_string = f"api:{wandb_api_key}"
        auth_header_value = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

        # Headers required for the OTLP exporter, including authorization for W&B
        # and the W&B project ID.
        wandb_headers = {
            "Authorization": f"Basic {auth_header_value}",  # Basic authentication header
            "project_id": f"{wandb_entity}/{wandb_project}",           # W&B Project ID for trace grouping
        }

        # Use the HTTP OTLP exporter for the HTTPS endpoint
        wandb_exporter = OTLPHttpSpanExporter(
            endpoint=wandb_endpoint,
            headers=wandb_headers,
        )

        return wandb_exporter

    except Exception as e_wandb:
        logger.warning("Error configuring W&B exporter", error=e_wandb)
        return None
