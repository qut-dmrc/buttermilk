"""Live tracing integration tests that send actual traces to providers.

Simple integration tests that use the live BM instance to send real traces
to weave, traceloop, and otel/gcp endpoints.
"""

import asyncio
import time

import pytest
import weave
from opentelemetry import trace

from buttermilk._core.bm_init import BM


class TestTracingProvidersLive:
    """Live integration tests for tracing providers."""

    @pytest.mark.anyio
    async def test_weave_live_submission(self, real_bm: BM) -> None:
        """Send a live trace to Weave."""
        try:
            client = await real_bm.get_weave_client()
            if client is None:
                pytest.skip("Weave not configured")

            def test_operation(x: int) -> int:
                return x + 1

            op = weave.op(test_operation, call_display_name="live-test")
            call = client.create_call(
                op,
                inputs={"x": 42},
                display_name="weave-live-test",
                attributes={"test": "live-integration", "timestamp": time.time()},
            )

            result = test_operation(42)
            client.finish_call(call, output={"result": result}, op=op)

            print(f"✅ Weave trace sent: {call.id}")

        except Exception as e:
            if "not configured" in str(e).lower():
                pytest.skip(f"Weave not configured: {e}")
            else:
                raise

    @pytest.mark.anyio
    async def test_traceloop_live_submission(self) -> None:
        """Send a live trace to Traceloop via OTEL."""
        try:
            from buttermilk.utils.otel import setup_traceloop_otel

            exporter = setup_traceloop_otel()
            if exporter is None:
                pytest.skip("Traceloop not configured")

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("traceloop-live-test") as span:
                span.set_attribute("test.provider", "traceloop")
                span.set_attribute("test.timestamp", time.time())
                time.sleep(0.1)
                span.set_attribute("test.result", "success")

            print("✅ Traceloop trace sent")

        except ImportError:
            pytest.skip("Traceloop dependencies not available")
        except Exception as e:
            if "not configured" in str(e).lower():
                pytest.skip(f"Traceloop not configured: {e}")
            else:
                raise

    @pytest.mark.anyio
    async def test_otel_live_submission(self) -> None:
        """Send a live trace to OTEL/GCP."""
        try:
            tracer_provider = trace.get_tracer_provider()
            if tracer_provider is None:
                pytest.skip("OTEL not configured")

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("otel-gcp-live-test") as span:
                span.set_attribute("test.provider", "otel")
                span.set_attribute("test.timestamp", time.time())

                # Nested span
                with tracer.start_as_current_span("nested-operation") as nested_span:
                    nested_span.set_attribute("operation.type", "test")
                    await asyncio.sleep(1)
                    nested_span.set_attribute("operation.result", "success")

                span.set_attribute("test.result", "success")

            await asyncio.sleep(1)
            print("✅ OTEL/GCP trace sent")

        except Exception as e:
            if "permission_denied" in str(e).lower():
                pytest.xfail(f"Known OTEL/GCP permission issue: {e}")
            elif "not configured" in str(e).lower():
                pytest.skip(f"OTEL not configured: {e}")
            else:
                raise
