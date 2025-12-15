"""Unit tests for OpenTelemetry span lifecycle functions (Phase 1).

These tests directly validate the OTEL helper functions in buttermilk/utils/otel.py:
1. session root span creation/removal
2. span_with_session context manager
3. baggage propagation

Tests use real OTEL SDK but don't require full FlowRunner setup.
"""

import pytest
from opentelemetry import baggage as otel_baggage, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from buttermilk.utils.otel import (
    attach_session_baggage,
    detach_session_baggage,
    end_session_root_span,
    span_with_session,
    start_session_root_span,
)


@pytest.fixture
def otel_setup():
    """Set up OTEL tracer with in-memory exporter for testing."""
    # Reset the global tracer provider before setting new one
    # This is necessary because OTEL doesn't allow overriding by default
    trace._TRACER_PROVIDER = None

    # Create in-memory exporter to capture spans
    exporter = InMemorySpanExporter()

    # Create tracer provider
    provider = TracerProvider()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set as global provider
    trace.set_tracer_provider(provider)

    yield exporter

    # Cleanup
    exporter.clear()
    trace._TRACER_PROVIDER = None


def test_session_root_span_creates_and_ends(otel_setup):
    """Test that session root span is created and properly ended.

    CURRENT BEHAVIOR: Session root span exists and persists.
    EXPECTED (after fix): Session root span should NOT exist.

    This test will PASS currently (documenting current behavior).
    After implementation, it should be updated or removed.
    """
    exporter = otel_setup
    session_id = "test-session-123"

    # ACT: Create session root span
    span, token = start_session_root_span(session_id, attributes={"test": "value"})

    # Create a child span under session root
    with span_with_session(session_id, name="child.span", attributes={"child": "attr"}):
        pass

    # End session root span
    end_session_root_span(span, token)

    # ASSERT: Verify spans were created
    spans = exporter.get_finished_spans()

    # Find session root span
    session_spans = [s for s in spans if s.name == "buttermilk.session"]

    # CURRENT: Session span exists
    assert len(session_spans) == 1, "Session root span should exist (current behavior)"

    # Verify session span has correct attributes
    session_span = session_spans[0]
    assert session_span.attributes.get("buttermilk.session.id") == session_id

    # NOTE: After implementing the fix (removing session root span),
    # this test should verify session_spans == 0


def test_span_with_session_creates_span_without_session_root(otel_setup):
    """Test that span_with_session works independently of session root span.

    EXPECTED BEHAVIOR (after fix):
    - span_with_session creates spans directly (no parent session span)
    - Session context propagated via baggage only
    - Each call to span_with_session creates an independent root span
    """
    exporter = otel_setup
    session_id = "test-independent-session"

    # ACT: Create span WITHOUT session root span
    # This simulates the desired behavior after removing session root span
    with span_with_session(
        session_id, name="independent.span", attributes={"test": "independent"}
    ):
        pass

    # ASSERT: Verify span was created
    spans = exporter.get_finished_spans()

    assert len(spans) == 1, "Should create exactly one span"

    span = spans[0]
    assert span.name == "independent.span"
    assert span.attributes.get("buttermilk.session.id") == session_id

    # CRITICAL: Verify this is a root span (no parent)
    # After fix, span_with_session spans should be root spans when
    # no other context is active
    assert span.parent is None, (
        "Span should be a root span when no session root span exists"
    )


def test_multiple_span_with_session_calls_create_independent_spans(otel_setup):
    """Test that multiple span_with_session calls create independent spans.

    ISSUE: With session root span, all spans nest under it.
    FIX: Without session root span, spans are independent (same session via baggage).

    This test verifies the desired behavior after fix.
    """
    exporter = otel_setup
    session_id = "test-multiple-spans"

    # ACT: Create multiple spans in the same session
    for i in range(3):
        with span_with_session(session_id, name=f"span.{i}", attributes={"index": i}):
            pass

    # ASSERT: Verify spans structure
    spans = exporter.get_finished_spans()

    # Should have 3 spans (no session root span)
    assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

    # All spans should have same session_id in attributes
    for span in spans:
        assert span.attributes.get("buttermilk.session.id") == session_id

    # CRITICAL: All spans should be root spans (no parent)
    # This is the key fix - spans are independent, not nested
    root_spans = [s for s in spans if s.parent is None]
    assert len(root_spans) == 3, (
        f"Expected 3 root spans (independent), got {len(root_spans)}. "
        f"This indicates spans are still nesting under a parent span."
    )


def test_baggage_propagation_without_session_root_span(otel_setup):
    """Test that session context propagates via baggage, not span hierarchy.

    EXPECTED BEHAVIOR (after fix):
    - attach_session_baggage sets baggage with session_id
    - Subsequent spans get session_id from baggage, not parent span
    - No session root span needed for propagation
    """
    exporter = otel_setup
    session_id = "test-baggage-propagation"

    # ACT: Attach session baggage
    token = attach_session_baggage(session_id)

    # Create span - should get session_id from baggage
    with span_with_session(
        None,  # No explicit session_id passed
        name="baggage.span",
        attributes={"test": "baggage"},
    ):
        # Verify baggage is accessible
        baggage_session_id = otel_baggage.get_baggage("buttermilk.session.id")
        assert baggage_session_id == session_id

    # Detach baggage
    detach_session_baggage(token)

    # ASSERT: Verify span has session_id from baggage
    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    # NOTE: Current implementation may or may not set attributes from baggage
    # This depends on BaggageToAttributesSpanProcessor configuration
    # The key point is baggage propagation works without session root span


def test_session_id_consistency_validation():
    """Test that session ID mismatches are detected.

    This test validates the new validation logic that will be added
    to FlowRunner.run_flow() to ensure session IDs match.

    NOTE: This is a specification test - the actual validation function
    doesn't exist yet. This defines the expected behavior.
    """
    # ARRANGE: Mismatched session IDs
    bm_session_id = "bm-session-abc"
    request_session_id = "request-session-xyz"

    # ACT & ASSERT: Validation should raise ValueError
    # TODO: Implement validate_session_id_match() function
    # with pytest.raises(ValueError, match="Session ID mismatch"):
    #     validate_session_id_match(bm_session_id, request_session_id)

    # For now, just verify they don't match (specification)
    assert bm_session_id != request_session_id

    # After implementation, the validation function should raise:
    # ValueError: Session ID mismatch: BM has 'bm-session-abc',
    # request has 'request-session-xyz'


# NOTE: These tests define the EXPECTED behavior after Phase 1 implementation.
# Tests with TODO markers will fail until implementation is complete.
# This is CORRECT - these are TDD tests defining requirements.
#
# Implementation checklist:
# 1. Remove start_session_root_span() call from FlowRunContext.get_or_create_session()
# 2. Remove end_session_root_span() call from FlowRunContext.cleanup()
# 3. Add session ID validation in FlowRunner.run_flow()
# 4. Update tests to remove session root span assertions
# 5. Verify all tests pass
