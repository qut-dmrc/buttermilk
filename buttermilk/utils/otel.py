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
    # Initialize the OpenTelemetry SDK's TracerProvider.
    # This provider manages the creation of tracers.
    tracer_provider = trace_sdk.TracerProvider()
    
    # Track which exporters were successfully configured
    exporters_configured = []
    
    # Configure W&B exporter
    try:
        # Retrieve necessary credentials from the global Buttermilk instance.
        # These are expected to be populated during Buttermilk initialization (e.g., from secrets).
        creds = bm.credentials
        if creds and "WANDB_API_KEY" in creds and "WANDB_PROJECT" in creds:
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
        else:
            logger.info("W&B credentials not found, skipping W&B exporter configuration")
    except Exception as e_wandb:
        logger.warning(f"Error configuring W&B exporter: {e_wandb}")
    
    # Configure GCP exporter
    try:
        # Import GCP exporter - using correct package path
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        
        # Get GCP project ID from configuration or environment
        gcp_project_id = None
        
        # First check if we have a GCP cloud config
        if hasattr(bm, 'cloud_cfg') and hasattr(bm.cloud_cfg, 'get_cloud_config'):
            gcp_config = bm.cloud_cfg.get_cloud_config('gcp')
            if gcp_config and hasattr(gcp_config, 'project_id'):
                gcp_project_id = gcp_config.project_id
        
        # Fallback to environment variable
        if not gcp_project_id:
            gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if gcp_project_id:
            # Configure the GCP Cloud Trace Span Exporter
            gcp_exporter = CloudTraceSpanExporter(
                project_id=gcp_project_id,
                # The exporter will use Application Default Credentials automatically
            )
            
            gcp_processor = BatchSpanProcessor(gcp_exporter)
            tracer_provider.add_span_processor(gcp_processor)
            exporters_configured.append(f"GCP (project: {gcp_project_id})")
            logger.info(f"OpenTelemetry GCP Cloud Trace exporter configured for project: {gcp_project_id}")
        else:
            logger.info("GCP project ID not found, skipping GCP exporter configuration")
    except ImportError:
        logger.warning("GCP Cloud Trace exporter not available - check if opentelemetry-exporter-cloud-trace is installed")
    except Exception as e_gcp:
        logger.warning(f"Error configuring GCP exporter: {e_gcp}")
    
    # Set the tracer provider globally
    trace.set_tracer_provider(tracer_provider)
    
    if exporters_configured:
        logger.info(
            f"OpenTelemetry (OTEL) tracing initialized with exporters: {', '.join(exporters_configured)}. "
            "Traces will be sent to configured destinations."
        )
    else:
        logger.warning("No OpenTelemetry exporters were configured. Tracing may not function properly.")
        
except Exception as e_otel: # Catch any other errors during setup
    logger.warning(f"Error during OpenTelemetry tracing setup: {e_otel!s}. OTEL tracing might not function.", exc_info=True)
