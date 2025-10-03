"""Standalone tracing context management for non-orchestrator workflows.

This module provides utilities for creating parent trace contexts when running
Buttermilk agents outside of an orchestrator context (e.g., in batch processes,
scripts, or one-off commands). It extracts the tracing initialization logic
from the orchestrator to make it reusable.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, TypeVar

import weave
from weave.trace.weave_client import Call

from buttermilk import bm, logger
from buttermilk._core.message_data import clean_empty_values

T = TypeVar("T")


class StandaloneTraceContext:
    """Manages a standalone trace context for non-orchestrator workflows.

    This class provides a way to create and manage parent traces for agents
    running outside of an orchestrator context, such as in batch processing
    scripts or CLI tools.
    """

    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        """Initialize a standalone trace context.

        Args:
            name: Display name for the trace (e.g., "vector_batch_process")
            attributes: Additional attributes to attach to the trace
        """
        self.name = name
        self.attributes = attributes or {}
        self.trace_call: Call | None = None
        self._op = None

    async def __aenter__(self) -> "StandaloneTraceContext":
        """Enter the trace context and create the parent trace."""

        # Create a dummy operation for tracing
        async def _standalone_operation(**kwargs):
            """Standalone operation for tracing."""
            return kwargs

        self._op = weave.op(_standalone_operation, call_display_name=self.name)

        # Create the parent trace call
        client = await bm.get_weave_client()
        self.trace_call = client.create_call(
            self._op,
            inputs=clean_empty_values({"name": self.name, **self.attributes}),
            display_name=self.name,
            attributes=self.attributes,
        )

        logger.debug(f"Created standalone trace context: {self.name} (trace_id: {self.trace_call.trace_id})")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the trace context and finish the trace."""
        if self.trace_call and self._op:
            # Finish the trace call
            output = {"status": "error" if exc_type else "success"}
            if exc_type:
                output["error"] = str(exc_val)

            bm.weave.finish_call(self.trace_call, output=output, op=self._op)
            logger.debug(f"Finished standalone trace context: {self.name}")

            if self.trace_call.ui_url:
                logger.info(f"Trace URL: {self.trace_call.ui_url}")

    def get_trace_id(self) -> str:
        """Get the trace ID for this context."""
        if not self.trace_call:
            raise RuntimeError("Trace context not active. Use within 'async with' block.")
        return self.trace_call.trace_id

    def get_call_id(self) -> str:
        """Get the call ID for this context."""
        if not self.trace_call:
            raise RuntimeError("Trace context not active. Use within 'async with' block.")
        return self.trace_call.id


@asynccontextmanager
async def create_standalone_trace(name: str, **attributes):
    """Context manager for creating a standalone trace.

    This is a convenience function that creates a StandaloneTraceContext
    and manages it as an async context manager.

    Args:
        name: Display name for the trace
        **attributes: Additional attributes to attach to the trace

    Yields:
        StandaloneTraceContext: The active trace context

    Example:
        async with create_standalone_trace("batch_process", batch_size=100) as trace:
            # Your code here - agents will use trace.trace_call as parent
            await agent.invoke(input_data, parent_call=trace.trace_call)
    """
    context = StandaloneTraceContext(name, attributes)
    async with context:
        yield context


def inject_parent_trace(agent_input: Any, trace_context: StandaloneTraceContext) -> Any:
    """Inject parent trace information into agent input.

    This helper function adds the parent_call_id from a standalone trace context
    to an agent input object, allowing the agent to properly nest its traces.

    Args:
        agent_input: The agent input object (should have parent_call_id attribute)
        trace_context: The active standalone trace context

    Returns:
        The modified agent input with parent_call_id set
    """
    if not trace_context.trace_call:
        raise RuntimeError("Trace context not active. Use within 'async with' block.")

    if hasattr(agent_input, "parent_call_id"):
        agent_input.parent_call_id = trace_context.get_call_id()
    else:
        logger.warning(f"Agent input {type(agent_input)} does not have parent_call_id attribute")

    return agent_input
