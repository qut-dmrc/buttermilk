from typing import Any

import weave  # For tracing - core dependency
from weave.trace.weave_client import Call, WeaveObject

from buttermilk import bm, logger

# Buttermilk core imports
from buttermilk._core.contract import (
    AgentInput,
)
from buttermilk._core.retry import RetryWrapper

"""Weave post-processing formatters for trace management.

This module provides formatters that can process weave traces after they are created,
allowing us to filter or modify traces before they are uploaded to the W&B backend.

See: https://weave-docs.wandb.ai/guides/tracking/tracing/#post-process-inputs-and-outputs
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
) -> Call | WeaveObject:
    from buttermilk._core.dmrc import get_bm  # Local import to avoid circular dependency

    if await bm.get_weave_client() is None:
        logger.warning("Weave client is not initialized, cannot retrieve parent call.")
        return None
    current_call = weave.get_current_call()

    if message is None or message.parent_call_id is None:
        # If no message or no parent call ID, return current call as parent
        return current_call

    # Unless calls are out of order, there's a good chance the current call is the parent.
    if message.parent_call_id == current_call.id:
        return current_call

    # If not, we have to go out and find the parent call from the Weave API.
    # This sometimes fails because the call hasn't been uploaded yet.
    # We retry a few times to handle this.
    #
    # TODO: check whether this is actually necessary; can we just use the ID to
    # associate calls together in a meaningful hiearchy?

    async def get_weave_call_with_retry(call_id: str) -> Call | WeaveObject:
        """Retry getting weave call to handle async upload timing."""
        bm = get_bm()
        client = await bm.get_weave_client()
        return client.get_call(call_id)

    # Use RetryWrapper with shorter delays for weave call retrieval
    retry_wrapper = RetryWrapper(
        client=None,  # Not using client, just the retry logic
        max_retries=3,
        min_wait_seconds=3.0,
        max_wait_seconds=10.0,
        jitter_seconds=0.1,
        cooldown_seconds=0,
    )

    try:
        parent_call = await retry_wrapper._execute_with_retry(
            get_weave_call_with_retry,
            message.parent_call_id,
        )
        return parent_call
    except Exception as e:  # Broad exception for Weave call retrieval
        logger.error(f"Could not retrieve parent call ID {message.parent_call_id} after retries. Error: {e}")
        return current_call
