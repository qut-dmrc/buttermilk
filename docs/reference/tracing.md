# Tracing in Buttermilk

This guide shows how to use OpenTelemetry (OTEL) tracing in Buttermilk with session-aware helpers. You’ll get clean span hierarchies, automatic session context, and logs correlated to spans via trace_id/span_id.

## Setup

Buttermilk configures OpenTelemetry and structlog for you when ExecutionContext is initialized. If tracing is enabled in your infrastructure config, spans will be exported to the configured backend (GCP, Traceloop, W&B via OTLP, etc.).

- Logs are structured and include `trace_id` and `span_id` automatically when a span is active.
- Session context (`buttermilk.session.id`) is bound via OTEL baggage and copied to every span as an attribute.

## Quick patterns

### 1) New child span for a unit of work

Use this to time a meaningful step (e.g., DB call, tool execution). Pass `session_id` to bind explicitly, or `None` if you’re already under a session/flow span.

```python
from buttermilk.utils.otel import span_with_session
from buttermilk import logger


def do_something(session_id: str | None, thing_id: str):
    with span_with_session(
        session_id,
        name="buttermilk.something",
        attributes={
            "buttermilk.thing.id": thing_id,
            "buttermilk.component": "my_module",
        },
        kind="internal",  # internal|client|server|producer|consumer
    ) as span:
        span.add_event("step_started")
        # ... do work ...
        result = {"ok": True}

        # Add attributes/events as you learn more
        span.set_attribute("buttermilk.something.result", "ok")
        span.add_event("step_completed", {"result": "ok"})

        # Logs are auto-correlated with this span
        logger.info("did the thing", thing_id=thing_id, result="ok")
        return result
```

### 2) Nested span inside current span

When you’re already inside a span and just need a quick nested span.

```python
from buttermilk.utils.otel import begin_span


def fetch_user(user_id: str):
    with begin_span(
        "db.query",
        {"db.system": "postgres", "db.operation": "SELECT", "db.table": "users"},
        kind="client",
    ):
        # run query...
        pass
```

### 3) Add an event to the current span (no new span)

Good for breadcrumbs without creating more spans.

```python
from opentelemetry.trace import get_current_span


def cache_lookup(key: str):
    span = get_current_span()
    if span:
        span.add_event("cache_lookup", {"key": key})
    # ... lookup ...
```

### 4) Record exceptions on a span

Ensure errors show up in traces even if they’re handled.

```python
from buttermilk.utils.otel import span_with_session


def risky_op(session_id: str | None):
    with span_with_session(session_id, "buttermilk.risky_op") as span:
        try:
            # do risky stuff...
            raise RuntimeError("boom")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            raise
```

### 5) Background tasks: re-bind session if needed

`contextvars` propagate through `await`, but not always across detached tasks. Re-attach session baggage inside them.

```python
import asyncio
from buttermilk.utils.otel import (
    attach_session_baggage,
    detach_session_baggage,
    begin_span,
)


def schedule_background(session_id: str):
    async def worker():
        token = attach_session_baggage(session_id)
        try:
            with begin_span("background.process", {"kind": "maintenance"}):
                # ... work ...
                pass
        finally:
            detach_session_baggage(token)

    asyncio.create_task(worker())
```

## Session-root span (recommended)

Buttermilk starts a long-lived `buttermilk.session` root span when a session is created and ends it on cleanup. All flow/job spans nest under it, and the span includes `buttermilk.session.status` which updates on transitions.

Benefits:

- Clear tree per session
- Automatic context inheritance for child spans
- Easy session-level timing and status

## Best practices

- Use one span per meaningful unit of work; avoid per-iteration spans in hot loops. Prefer span events or metrics.
- Add attributes you’ll query by: `buttermilk.flow.name`, `buttermilk.job.id`, `buttermilk.session.id`, dataset, model, external service names, `record_id`.
- Avoid logging full payloads; record sizes, counts, and short prefixes instead.
- Exceptions: `span.record_exception(e)` and set `error`/status attributes so traces surface issues.

## Troubleshooting

- No trace_id/span_id in logs? Ensure a span is active where you log. The structlog processor adds IDs only when a current span exists.
- Detached tasks missing context? Re-attach session baggage as shown in the background example.
- No spans in backend? Check that OTEL tracing is enabled in ExecutionContext and that exporter credentials are configured.
