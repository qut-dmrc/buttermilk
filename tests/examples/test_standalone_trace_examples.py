"""Test examples for standalone tracing that serve as documentation.

These tests demonstrate how to use standalone tracing for batch processes
and scripts running outside of orchestrator contexts. They ensure
the examples in our documentation remain accurate.
"""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from buttermilk._core.contract import AgentInput
from buttermilk._core.log import logger
from buttermilk._core.standalone_trace import StandaloneTraceContext, create_standalone_trace, inject_parent_trace


class TestStandaloneTraceExamples:
    """Examples for standalone tracing that also serve as tests."""

    @pytest.fixture
    def mock_weave_client(self, real_bm):
        """Mock weave client for standalone trace tests."""
        mock_call = Mock()
        mock_call.trace_id = "mock-trace-id"
        mock_call.id = "mock-call-id"
        mock_call.ui_url = "https://wandb.ai/mock-trace"

        mock_client = Mock()
        mock_client.create_call = Mock(return_value=mock_call)

        async def get_mock_client():
            return mock_client

        with patch('buttermilk._core.standalone_trace.bm') as mock_bm:
            mock_bm.get_weave_client = AsyncMock(return_value=mock_client)
            mock_bm.weave = Mock()
            mock_bm.weave.finish_call = Mock()
            yield mock_client

    @pytest.mark.anyio
    async def test_basic_usage_example(self, real_bm, mock_weave_client):
        """
        Basic usage of standalone tracing with context manager.

        This example demonstrates:
        - How to create a standalone trace context
        - Adding custom attributes to traces
        - Accessing trace and call IDs
        """
        # Create a trace context with custom attributes
        async with create_standalone_trace("batch_processing_example", batch_size=10, job_type="data_import") as trace:
            # The trace context is now active
            assert trace.trace_call is not None

            # You can get IDs for logging or passing to other systems
            trace_id = trace.get_trace_id()
            call_id = trace.get_call_id()

            assert isinstance(trace_id, str)
            assert isinstance(call_id, str)

            # Your batch processing logic would go here
            logger.info("Processing batch", trace_id=trace_id)

    @pytest.mark.anyio
    async def test_agent_integration_example(self, real_bm, mock_weave_client):
        """
        Using standalone traces with agents.

        This example shows:
        - Creating agent input with parent trace
        - Passing trace context to agents
        - Proper trace hierarchy
        """
        async with create_standalone_trace("agent_batch_job") as trace:
            # Create agent input that will be traced
            agent_input = AgentInput(
                inputs={"data": "test_data"},
                parent_call_id=trace.get_call_id(),  # Link to parent trace
            )

            # The agent would use this parent_call_id to nest its traces
            assert agent_input.parent_call_id == trace.get_call_id()

            # Mock agent invocation (in real code, you'd call agent.invoke)
            # result = await agent.invoke(agent_input)

    @pytest.mark.anyio
    async def test_batch_processing_pattern(self, real_bm, mock_weave_client):
        """
        Common pattern for batch processing with traces.

        This demonstrates:
        - Processing multiple items in one trace
        - Error handling within trace context
        - Trace completion with status
        """
        items_to_process = ["item1", "item2", "item3"]
        processed_items = []

        async with create_standalone_trace("batch_processor", total_items=len(items_to_process)):
            for item in items_to_process:
                try:
                    # Process each item (mock processing here)
                    await self._mock_process_item(item)
                    processed_items.append(item)
                except Exception as e:
                    logger.error("Failed to process item", item=item, error=e)
                    # Continue processing other items

            # Trace will automatically close with success/error status
            assert len(processed_items) == len(items_to_process)

    @pytest.mark.anyio
    async def test_inject_parent_trace_example(self, real_bm, mock_weave_client):
        """
        Using the inject_parent_trace helper function.

        This shows:
        - Helper function for adding trace context
        - Handling objects without parent_call_id
        """
        async with create_standalone_trace("helper_example") as trace:
            # Object with parent_call_id attribute
            agent_input = AgentInput(inputs={"test": "data"})
            injected = inject_parent_trace(agent_input, trace)
            assert injected.parent_call_id == trace.get_call_id()

            # Object without parent_call_id (logs warning)
            regular_dict = {"data": "value"}
            with patch("buttermilk._core.standalone_trace.logger.warning") as mock_warning:
                result = inject_parent_trace(regular_dict, trace)
                assert result == regular_dict  # Returns unchanged
                mock_warning.assert_called_once()

    @pytest.mark.anyio
    async def test_error_handling_example(self, real_bm, mock_weave_client):
        """
        Proper error handling with trace contexts.

        Demonstrates:
        - Trace behavior during exceptions
        - Error status in trace output
        - Context cleanup on error
        """
        with pytest.raises(ValueError):
            async with create_standalone_trace("error_example") as trace:
                # Trace is created successfully
                assert trace.trace_call is not None

                # Simulate an error during processing
                raise ValueError("Simulated processing error")

        # Trace context properly exits even with error
        # The trace would show error status in Weave UI

    @pytest.mark.anyio
    async def test_manual_context_management(self, real_bm, mock_weave_client):
        """
        Using StandaloneTraceContext directly for more control.

        Shows:
        - Direct class usage vs context manager
        - Manual enter/exit handling
        """
        context = StandaloneTraceContext("manual_trace", {"custom_field": "custom_value"})

        # Manual context management
        await context.__aenter__()
        try:
            assert context.trace_call is not None
            # Processing logic here
        finally:
            await context.__aexit__(None, None, None)

    async def _mock_process_item(self, item: str):
        """Mock processing function for examples."""
        # Simulate some async work
        await asyncio.sleep(0.01)
        return f"processed_{item}"


class TestStandaloneTraceEdgeCases:
    """Edge cases and error scenarios for standalone tracing."""

    @pytest.fixture
    def mock_weave_client(self, real_bm):
        """Mock weave client for standalone trace tests."""
        mock_call = Mock()
        mock_call.trace_id = "mock-trace-id"
        mock_call.id = "mock-call-id"
        mock_call.ui_url = "https://wandb.ai/mock-trace"

        mock_client = Mock()
        # Return different call IDs for nested traces
        call_counter = [0]
        def create_call_side_effect(*args, **kwargs):
            call_counter[0] += 1
            call = Mock()
            call.trace_id = f"mock-trace-id-{call_counter[0]}"
            call.id = f"mock-call-id-{call_counter[0]}"
            call.ui_url = f"https://wandb.ai/mock-trace-{call_counter[0]}"
            return call

        mock_client.create_call = Mock(side_effect=create_call_side_effect)

        async def get_mock_client():
            return mock_client

        with patch('buttermilk._core.standalone_trace.bm') as mock_bm:
            mock_bm.get_weave_client = AsyncMock(return_value=mock_client)
            mock_bm.weave = Mock()
            mock_bm.weave.finish_call = Mock()
            yield mock_client

    @pytest.mark.anyio
    async def test_trace_not_active_error(self):
        """Test error when accessing trace before activation."""
        context = StandaloneTraceContext("inactive_trace")

        with pytest.raises(RuntimeError, match="Trace context not active"):
            context.get_trace_id()

        with pytest.raises(RuntimeError, match="Trace context not active"):
            context.get_call_id()

    @pytest.mark.anyio
    async def test_nested_traces(self, real_bm, mock_weave_client):
        """Example of nested trace contexts."""
        async with create_standalone_trace("outer_trace") as outer:
            outer_id = outer.get_trace_id()

            async with create_standalone_trace("inner_trace") as inner:
                inner_id = inner.get_trace_id()

                # Each context maintains its own trace
                assert outer_id != inner_id
                assert outer.trace_call != inner.trace_call
