"""Configures OpenTelemetry (OTEL) tracing for the Buttermilk framework.

This module sets up global OpenTelemetry tracing, configured to export trace data to:
1. Google Cloud Platform (GCP) using the Cloud Trace exporter
2. Optional Traceloop integration

The setup is performed automatically when this module is imported.

It relies on credentials being available via the global Buttermilk instance:
- For GCP: Uses Application Default Credentials or environment variables
- For Traceloop: TRACELOOP_API_KEY, TRACELOOP_BASE_URL in `bm.credentials`

Note:
    This module primarily executes configuration logic upon import and does not
    define reusable public functions or classes for direct invocation beyond setup.
    Actual trace creation (spans, etc.) would use standard OpenTelemetry APIs
    elsewhere in the codebase, relying on this global setup.

"""

import logging
import os
from contextlib import contextmanager

import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
import grpc
from google.auth.transport.grpc import AuthMetadataPlugin
from opentelemetry import baggage as otel_baggage, context as otel_context, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPHttpSpanExporter,
)
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor
from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    SpanKind as _SpanKind,
    set_span_in_context as _set_span_in_context,
)

from buttermilk import bm, logger
from buttermilk._core.config import Tracing

# Suppress noisy OpenTelemetry instrumentation debug logs for non-OpenAI models
logging.getLogger("opentelemetry.instrumentation.openai.shared").setLevel(logging.WARNING)


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

    # Get service name (preserve if already set by config_bootstrap.py)
    # Default to "buttermilk" if not set
    service_name = os.environ.get("OTEL_SERVICE_NAME", "buttermilk")

    # Set OTEL_RESOURCE_ATTRIBUTES with BOTH service.name and gcp.project_id
    # This preserves the service name that was set in config_bootstrap.py
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"service.name={service_name},gcp.project_id={project_id}"
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

    # Initialize the OTLP gRPC exporter
    otlp_grpc_exporter = OTLPSpanExporter(credentials=channel_creds)

    # Initialize OpenTelemetry TracerProvider
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
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument(
            tracer_provider=provider,
            set_logging_format=False,  # Don't override our logging setup
            log_level=logging.INFO,  # Only include INFO+ logs in OTEL traces
        )

    # Exporters / processors
    provider.add_span_processor(BatchSpanProcessor(otlp_grpc_exporter))
    # Also lift selected baggage keys onto spans
    try:
        provider.add_span_processor(
            BaggageToAttributesSpanProcessor(
                keys=[
                    "buttermilk.session.id",
                    "buttermilk.project",
                    "buttermilk.execution_context.id",
                ]
            )
        )
    except Exception:
        pass

    # Configure OpenTelemetry tracing API with the initialized tracer provider
    trace.set_tracer_provider(provider)
    logger.info("Initialized tracing with Google Cloud", project_id=project_id)


# ---- Reusable helpers: session-aware spans and baggage ----


def _clean_attrs(attrs: dict | None) -> dict:
    """Remove None values to keep spans tidy."""
    if not attrs:
        return {}
    return {k: v for k, v in attrs.items() if v is not None}


def attach_session_baggage(session_id: str | None, extra: dict | None = None) -> object | None:
    """Attach session metadata as OTEL baggage to current context.

    Returns a context token that must be detached later. Safe no-op if session_id is None.
    """
    if not session_id:
        return None
    baggage = otel_baggage.set_baggage("buttermilk.session.id", session_id)
    if extra:
        for k, v in extra.items():
            if v is not None:
                baggage = otel_baggage.set_baggage(k, str(v), baggage)
    return otel_context.attach(baggage)


def detach_session_baggage(token: object | None) -> None:
    """Detach previously attached baggage token (if any)."""
    if token is not None:
        try:
            otel_context.detach(token)
        except Exception:
            pass


@contextmanager
def span_with_session(
    session_id: str | None,
    name: str,
    attributes: dict | None = None,
    kind: str | _SpanKind | None = None,
):
    """Context manager that binds session baggage and starts a span.

    - Attaches buttermilk.session.id as baggage so nested spans/logs inherit context.
    - Adds session id and project name as span attributes for easy querying.
    - Accepts kind as a string ("internal", "producer", "consumer", "server", "client") or SpanKind.
    """
    # Map kind
    if isinstance(kind, str):
        kind_map = {
            "internal": _SpanKind.INTERNAL,
            "server": _SpanKind.SERVER,
            "client": _SpanKind.CLIENT,
            "producer": _SpanKind.PRODUCER,
            "consumer": _SpanKind.CONSUMER,
        }
        span_kind = kind_map.get(kind.lower(), _SpanKind.INTERNAL)
    else:
        span_kind = kind or _SpanKind.INTERNAL

    # Get project name from BM (with graceful fallback)
    project_name = "unknown"
    try:
        # Access BM instance to get project name
        if bm is not None and hasattr(bm, "session_info") and hasattr(bm.session_info, "project_name"):
            project_name = bm.session_info.project_name
    except Exception:
        # BM not available or no project name set - use unknown
        project_name = "unknown"

    # Merge attributes with session id and project name
    base_attrs = {}
    if session_id:
        base_attrs["buttermilk.session.id"] = session_id
        base_attrs["buttermilk.project.name"] = project_name

    all_attrs = _clean_attrs({**base_attrs, **(attributes or {})})

    tracer = trace.get_tracer(__name__)
    token = attach_session_baggage(session_id)
    try:
        with tracer.start_as_current_span(name, kind=span_kind, attributes=all_attrs) as span:
            yield span
    finally:
        detach_session_baggage(token)


@contextmanager
def start_root_span(
    name: str,
    attributes: dict | None = None,
    kind: str | _SpanKind | None = None,
):
    """Context manager that starts a ROOT span (detached from any parent context).

    This is crucial for batch jobs running in the same worker process - it ensures
    each job gets an independent trace instead of nesting under previous jobs.

    Use this for top-level operations like:
    - Flow execution (buttermilk.flow.run)
    - Batch job processing
    - Any operation that should start a new trace tree

    NOTE: This does NOT attach baggage - baggage should already be set via
    attach_session_baggage() during BM initialization.

    Args:
        name: Span name
        attributes: Span attributes (should include buttermilk.session.id)
        kind: Span kind (string or SpanKind enum)

    Yields:
        The root span
    """
    # Map kind
    if isinstance(kind, str):
        kind_map = {
            "internal": _SpanKind.INTERNAL,
            "server": _SpanKind.SERVER,
            "client": _SpanKind.CLIENT,
            "producer": _SpanKind.PRODUCER,
            "consumer": _SpanKind.CONSUMER,
        }
        span_kind = kind_map.get(kind.lower(), _SpanKind.INTERNAL)
    else:
        span_kind = kind or _SpanKind.INTERNAL

    tracer = trace.get_tracer(__name__)
    all_attrs = _clean_attrs(attributes)

    # Create a detached context (no parent span)
    # This ensures this span becomes a root of a new trace
    from opentelemetry import context as otel_context

    detached_context = otel_context.Context()  # Fresh, empty context

    with tracer.start_as_current_span(
        name,
        context=detached_context,  # Explicitly detach from parent!
        kind=span_kind,
        attributes=all_attrs,
    ) as span:
        yield span


def begin_span(name: str, attributes: dict | None = None, kind: str | _SpanKind | None = None):
    """Simple span context manager without session baggage binding."""
    if isinstance(kind, str):
        kind_map = {
            "internal": _SpanKind.INTERNAL,
            "server": _SpanKind.SERVER,
            "client": _SpanKind.CLIENT,
            "producer": _SpanKind.PRODUCER,
            "consumer": _SpanKind.CONSUMER,
        }
        span_kind = kind_map.get(kind.lower(), _SpanKind.INTERNAL)
    else:
        span_kind = kind or _SpanKind.INTERNAL
    tracer = trace.get_tracer(__name__)
    return tracer.start_as_current_span(name, kind=span_kind, attributes=_clean_attrs(attributes))


def start_session_root_span(session_id: str | None, attributes: dict | None = None):
    """Start a long-lived session root span and set it as current.

    Returns (span, token). Call end_session_root_span(span, token) to close.
    """
    tracer = trace.get_tracer(__name__)
    base_attrs = {"buttermilk.session.id": session_id} if session_id else {}
    attrs = _clean_attrs({**base_attrs, **(attributes or {})})
    span = tracer.start_span("buttermilk.session", kind=_SpanKind.INTERNAL, attributes=attrs)
    token = otel_context.attach(_set_span_in_context(span))
    return span, token


def end_session_root_span(span, token: object | None = None) -> None:
    """End a previously started session root span and detach context token."""
    try:
        if span is not None:
            span.end()
    except Exception:
        pass
    if token is not None:
        try:
            otel_context.detach(token)
        except Exception:
            pass


class BaggageToAttributesSpanProcessor(_SpanProcessor):
    """Span processor that lifts selected baggage keys into span attributes at start."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = keys or []

    def on_start(self, span, parent_context) -> None:  # type: ignore[override]
        try:
            for k in self._keys:
                v = otel_baggage.get_baggage(k, parent_context)
                if v is not None:
                    span.set_attribute(k, v)
        except Exception:
            # Never fail user code due to telemetry
            pass

    def on_end(self, span) -> None:  # type: ignore[override]
        return

    def shutdown(self) -> None:  # type: ignore[override]
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        return True


# --- OpenTelemetry Tracing Setup for Traceloop ---
def setup_traceloop_otel() -> OTLPHttpSpanExporter | None:
    """Initialize Traceloop for OpenTelemetry tracing."""

    try:
        creds = bm.credentials

        traceloop_api_key = os.getenv("TRACELOOP_API_KEY") or creds["TRACELOOP_API_KEY"]
        traceloop_base_url = os.getenv("TRACELOOP_BASE_URL") or creds["TRACELOOP_BASE_URL"]
        traceloop_endpoint = f"{traceloop_base_url}/v1/traces"
        # Do not URL-encode Authorization headers
        traceloop_headers = {"Authorization": f"Bearer {traceloop_api_key}"}
        traceloop_exporter = OTLPHttpSpanExporter(endpoint=traceloop_endpoint, headers=traceloop_headers)
        return traceloop_exporter

    except Exception as e_traceloop:
        logger.warning("Error configuring traceloop exporter", error=e_traceloop)
        return None
