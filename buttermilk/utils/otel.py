import base64

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core.log import logger

# Tracing setup (OTEL)
WANDB_BASE_URL = "https://trace.wandb.ai"
OTEL_EXPORTER_OTLP_ENDPOINT = f"{WANDB_BASE_URL}/otel/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = OTEL_EXPORTER_OTLP_ENDPOINT
try:
    creds = bm.credentials
    AUTH = base64.b64encode(f"api:{creds['WANDB_API_KEY']}".encode()).decode()
    OTEL_EXPORTER_OTLP_HEADERS = {
        "Authorization": f"Basic {AUTH}",
        "project_id": creds["WANDB_PROJECT"],
    }
    # Initialize the OpenTelemetry SDK
    from opentelemetry.sdk import trace as trace_sdk

    tracer_provider = trace_sdk.TracerProvider()

    # Configure the OTLP exporter
    exporter = OTLPSpanExporter(
        endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
        headers=OTEL_EXPORTER_OTLP_HEADERS,
    )

    logger.info(
        "Tracing setup configured (OTEL). Weave initialization happens on first access.",
    )
except Exception as e:
    logger.warning(f"Error during OpenTelemetry tracing setup: {e}. Continuing without OTEL tracing.")
