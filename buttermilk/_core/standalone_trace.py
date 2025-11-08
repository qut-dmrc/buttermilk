"""Standalone tracing context management for non-orchestrator workflows.

This module provides utilities for creating parent trace contexts when running
Buttermilk agents outside of an orchestrator context (e.g., in batch processes,
scripts, or one-off commands). It extracts the tracing initialization logic
from the orchestrator to make it reusable.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, TypeVar

# weave imports removed
from buttermilk import logger

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
        self.trace_call: Any | None = None  # Previously weave.Call, now always None
        self._op = None

    async def __aenter__(self) -> "StandaloneTraceContext":
        """Enter the trace context - weave has been removed."""
        logger.debug(f"Standalone trace context (weave removed): {self.name}")
        self.trace_call = None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the trace context - weave has been removed."""
        logger.debug(f"Finished standalone trace context (weave removed): {self.name}")

    def get_trace_id(self) -> str | None:
        """Get the trace ID for this context - weave has been removed."""
        return None

    def get_call_id(self) -> str | None:
        """Get the call ID for this context - weave has been removed."""
        return None


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
        logger.warning(
            f"Agent input {type(agent_input)} does not have parent_call_id attribute"
        )

    return agent_input
