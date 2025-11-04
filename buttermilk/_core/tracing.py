from typing import Any

from buttermilk import logger

# Buttermilk core imports
from buttermilk._core.contract import (
    AgentInput,
)

"""Legacy tracing module - weave has been removed.

This module previously provided weave trace formatters and parent call retrieval.
These features are no longer needed after weave removal. The module is kept
for backward compatibility but all functions return None.
"""


class NoOpFormatter:
    """A no-op formatter that passes through all traces unchanged.

    This serves as a base implementation and example of the formatter interface.
    It can be extended to add filtering logic in the future.
    """

    def format(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """Process trace data without modifications.

        Args:
            trace_data: The trace data dictionary from weave

        Returns:
            The unmodified trace data
        """
        return trace_data


class EmptyTraceFilter:
    """A formatter that filters out traces with no meaningful output.

    WARNING: This formatter should be used carefully. It filters traces based on
    specific criteria to remove noise from message handlers that don't produce output.

    Filtering criteria (all must be true):
    1. The trace op_name ends with ".message_handler"
    2. The trace has no error or exception
    3. The output is None, empty dict {}, or empty string ""

    This helps reduce clutter from autogen's automatic tracing of all message handlers,
    including those that intentionally ignore messages. Traces with errors or exceptions
    are always preserved for debugging purposes.

    Usage:
        filter = EmptyTraceFilter()
        # Configure weave to use this formatter
        # The formatter's format() method will be called for each trace

    Note:
        This filter is designed specifically for autogen message handlers which
        follow the pattern "ClassName.method_name.message_handler" in their op_name.
    """

    def format(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        """Process trace data and filter empty traces.

        Args:
            trace_data: The trace data dictionary from weave

        Returns:
            The trace data if it should be kept, None to filter it out
        """
        # Check if this is a message handler trace
        op_name = trace_data.get("op_name", "")
        if not op_name.endswith(".message_handler"):
            # Not a message handler, keep the trace
            return trace_data

        # Check if there's an error or exception - always keep these for debugging
        if "error" in trace_data or "exception" in trace_data:
            return trace_data

        # Check if output is empty
        output = trace_data.get("output")
        if output is None or output in ({}, ""):
            # This appears to be a no-op handler, filter it out
            logger.debug(f"Filtering empty message handler trace: op_name={op_name}, inputs={trace_data.get('inputs', {})}")
            return None

        # Output has some value, keep the trace
        return trace_data


async def get_parent_call_weave(
    message: AgentInput | None = None,
) -> None:
    """Legacy function - weave has been removed.

    This function previously retrieved parent weave calls for trace hierarchy.
    After weave removal, it always returns None.

    Args:
        message: AgentInput with optional parent_call_id (ignored)

    Returns:
        None: Weave is no longer used
    """
    logger.debug("get_parent_call_weave called but weave has been removed, returning None")
