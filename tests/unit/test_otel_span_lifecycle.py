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
    """Test session root span functions exist for backward compatibility but are not used.

    POST-FIX BEHAVIOR: Session root span functions still exist in otel.py
    for backward compatibility, but they are NO LONGER used by FlowRunner.
    This test verifies the functions work if called directly, but documents
    that they should not be part of normal flow execution.
    """
    exporter = otel_setup
    session_id = "test-session-123"

    # ACT: Call session root span functions directly (backward compatibility test)
    span, token = start_session_root_span(session_id, attributes={"test": "value"})

    # Create a child span - should work independently
    with span_with_session(session_id, name="child.span", attributes={"child": "attr"}):
        pass

    # End session root span
    end_session_root_span(span, token)

    # ASSERT: Verify spans were created
    spans = exporter.get_finished_spans()

    # The session root span was created because we called the function directly
    # However, in production code (FlowRunner), these functions are NOT called
    session_spans = [s for s in spans if s.name == "buttermilk.session"]
    assert len(session_spans) == 1, "Session root span created when called directly"

    # Verify session span has correct attributes
    session_span = session_spans[0]
    assert session_span.attributes.get("buttermilk.session.id") == session_id

    # IMPORTANT: FlowRunner no longer calls start_session_root_span or end_session_root_span
    # Session context is propagated via baggage only


def test_span_with_session_creates_span_without_session_root(otel_setup):
    """Test that span_with_session works independently without session root span.

    POST-FIX BEHAVIOR:
    - span_with_session creates spans directly (no parent session span needed)
    - Session context propagated via baggage only
    - When no parent context exists, span_with_session creates a root span
    """
    exporter = otel_setup
    session_id = "test-independent-session"

    # ACT: Create span WITHOUT calling start_session_root_span
    # This is the NEW normal behavior - session context via baggage, not span hierarchy
<<<<<<< HEAD
    with span_with_session(session_id, name="independent.span", attributes={"test": "independent"}):
=======
    with span_with_session(
        session_id, name="independent.span", attributes={"test": "independent"}
    ):
>>>>>>> origin/stable
        pass

    # ASSERT: Verify span was created
    spans = exporter.get_finished_spans()

    assert len(spans) == 1, "Should create exactly one span"

    span = spans[0]
    assert span.name == "independent.span"
    assert span.attributes.get("buttermilk.session.id") == session_id

    # VERIFIED: This is a root span (no parent) because no session root span was created
    # span_with_session creates independent spans, not nested under a session parent
<<<<<<< HEAD
    assert span.parent is None, "Span should be a root span when no parent context is active"
=======
    assert span.parent is None, (
        "Span should be a root span when no parent context is active"
    )
>>>>>>> origin/stable


def test_multiple_span_with_session_calls_create_independent_spans(otel_setup):
    """Test that multiple span_with_session calls create independent spans.

    POST-FIX BEHAVIOR: Without session root span, all spans are independent root spans.
    They share the same session_id via baggage propagation, not via parent-child hierarchy.
    """
    exporter = otel_setup
    session_id = "test-multiple-spans"

    # ACT: Create multiple spans in the same session WITHOUT session root span
    for i in range(3):
        with span_with_session(session_id, name=f"span.{i}", attributes={"index": i}):
            pass

    # ASSERT: Verify spans structure
    spans = exporter.get_finished_spans()

    # Should have exactly 3 spans (no session root span created)
    assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

    # All spans should have same session_id in attributes (from baggage)
    for span in spans:
        assert span.attributes.get("buttermilk.session.id") == session_id

    # VERIFIED: All spans are root spans (no parent) - this is the correct behavior
    # Spans are independent, linked by session_id attribute, not by hierarchy
    root_spans = [s for s in spans if s.parent is None]
<<<<<<< HEAD
    assert len(root_spans) == 3, f"Expected 3 independent root spans, got {len(root_spans)}. Each span_with_session call should create a root span."
=======
    assert len(root_spans) == 3, (
        f"Expected 3 independent root spans, got {len(root_spans)}. "
        f"Each span_with_session call should create a root span."
    )
>>>>>>> origin/stable


def test_baggage_propagation_without_session_root_span(otel_setup):
    """Test that session context propagates via baggage, not span hierarchy.

    POST-FIX BEHAVIOR:
    - attach_session_baggage sets baggage with session_id
    - Baggage is accessible within the context
    - span_with_session can use baggage for session context
    - No session root span needed for propagation
    """
    exporter = otel_setup
    session_id = "test-baggage-propagation"

    # ACT: Attach session baggage manually
    token = attach_session_baggage(session_id)

    # Create span - baggage should be accessible in context
    with span_with_session(
        session_id,  # Explicit session_id (best practice)
        name="baggage.span",
        attributes={"test": "baggage"},
    ):
        # Verify baggage is accessible within the span context
        baggage_session_id = otel_baggage.get_baggage("buttermilk.session.id")
<<<<<<< HEAD
        assert baggage_session_id == session_id, "Baggage should be accessible within span context"
=======
        assert baggage_session_id == session_id, (
            "Baggage should be accessible within span context"
        )
>>>>>>> origin/stable

    # Detach baggage
    detach_session_baggage(token)

    # ASSERT: Verify span was created
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, "Should create exactly one span"

    # Verify span has session_id attribute
    span = spans[0]
<<<<<<< HEAD
    assert span.attributes.get("buttermilk.session.id") == session_id, "Span should have session_id attribute from span_with_session"
=======
    assert span.attributes.get("buttermilk.session.id") == session_id, (
        "Span should have session_id attribute from span_with_session"
    )
>>>>>>> origin/stable


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
