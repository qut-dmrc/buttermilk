"""Weave post-processing formatters for trace management.

This module provides formatters that can process weave traces after they are created,
allowing us to filter or modify traces before they are uploaded to the W&B backend.

See: https://weave-docs.wandb.ai/guides/tracking/tracing/#post-process-inputs-and-outputs
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
            logger.debug(
                f"Filtering empty message handler trace: op_name={op_name}, "
                f"inputs={trace_data.get('inputs', {})}"
            )
            return None

        # Output has some value, keep the trace
        return trace_data


# Plan for implementing trace filtering:
#
# 1. Analysis Phase (with admin supervision):
#    - Export current weave traces to analyze structure
#    - Identify patterns of empty/no-op traces
#    - Document what fields indicate a trace should be filtered
#    - Review historical data to ensure we don't lose important info
#
# 2. Implementation Phase:
#    - Update EmptyTraceFilter.format() with specific filtering logic
#    - Add configuration options for what to filter
#    - Add metrics/logging to track what gets filtered
#
# 3. Testing Phase:
#    - Test with sample traces to verify filtering works correctly
#    - Run in parallel with no-op formatter to compare results
#    - Gradually roll out to production
#
# 4. Monitoring:
#    - Track metrics on filtered vs kept traces
#    - Ensure no important data is lost
#    - Adjust filtering criteria based on feedback

