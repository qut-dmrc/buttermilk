"""Tests for ParallelProcessor."""

import asyncio
from typing import Any, AsyncGenerator

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.processors.parallel import ParallelProcessor


class MockProcessor(ProcessorCore):
    """Simple mock processor for testing.

    A proper ProcessorCore subclass following the _process_record protocol.
    """

    suffix: str = "mock"
    delay: float = 0.0
    fail: bool = False
    multi_output: int = 1
    call_count: int = 0
    received_record_ids: list[str] = []
    received_contents: list[str] = []

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """Process record, optionally with delay or failure."""
        self.call_count += 1
        self.received_record_ids.append(getattr(context.record, "record_id", ""))
        self.received_contents.append(getattr(context.record, "content", ""))

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise ValueError(f"MockProcessor '{self.suffix}' configured to fail")

        for i in range(self.multi_output):
            suffix = f"_{i}" if self.multi_output > 1 else ""
            yield context.record.model_copy(
                update={
                    "content": f"{context.record.content}_{self.suffix}{suffix}",
                }
            )


class TestParallelProcessorUnit:
    """Unit tests for ParallelProcessor instantiation."""

    def test_instantiation_empty_processors(self):
        """Test that empty processor list is valid."""
        proc = ParallelProcessor(processors=[])
        assert len(proc.processors) == 0

    def test_instantiation_with_fail_on_error(self):
        """Test fail_on_error configuration."""
        proc = ParallelProcessor(processors=[], fail_on_error=True)
        assert proc.fail_on_error is True

        proc = ParallelProcessor(processors=[], fail_on_error=False)
        assert proc.fail_on_error is False


class TestParallelProcessorExecution:
    """Tests for parallel execution behavior."""

    @pytest.fixture
    def parallel_processor_with_mocks(self):
        """Create ParallelProcessor with mock sub-processors."""
        mock_a = MockProcessor(suffix="A", delay=0.1)
        mock_b = MockProcessor(suffix="B", delay=0.05)
        mock_c = MockProcessor(suffix="C", delay=0.15)

        return ParallelProcessor(processors=[mock_a, mock_b, mock_c])

    @pytest.mark.anyio
    async def test_all_processors_receive_same_input(self):
        """Test that all sub-processors receive the SAME input record."""
        mock_a = MockProcessor(suffix="A")
        mock_b = MockProcessor(suffix="B")
        mock_c = MockProcessor(suffix="C")

        proc = ParallelProcessor(processors=[mock_a, mock_b, mock_c])
        record = BaseRecord(record_id="test-1", content="original")
        context = ProcessingContext(session_id="test", record=record)

        async for _ in proc.process(context):
            pass

        assert mock_a.call_count == 1
        assert mock_b.call_count == 1
        assert mock_c.call_count == 1

        assert mock_a.received_record_ids[0] == "test-1"
        assert mock_b.received_record_ids[0] == "test-1"
        assert mock_c.received_record_ids[0] == "test-1"
        assert mock_a.received_contents[0] == "original"
        assert mock_b.received_contents[0] == "original"
        assert mock_c.received_contents[0] == "original"

    @pytest.mark.anyio
    async def test_yields_all_outputs(self, parallel_processor_with_mocks):
        """Test that outputs from all processors are yielded."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        assert len(outputs) == 3
        contents = {o.content for o in outputs}
        assert contents == {"hello_A", "hello_B", "hello_C"}

    @pytest.mark.anyio
    async def test_parallel_metadata_added(self, parallel_processor_with_mocks):
        """Test that parallel metadata is added to outputs."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        for output in outputs:
            assert "parallel" in output.metadata
            meta = output.metadata["parallel"]
            assert "processor_index" in meta
            assert meta["total_processors"] == 3
            assert meta["processor_class"] == "MockProcessor"

    @pytest.mark.anyio
    async def test_results_stream_as_completed(self, parallel_processor_with_mocks):
        """Test that faster processors yield results first."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        completion_order = []
        async for output in proc.process(context):
            completion_order.append(output.content)

        # B (0.05s) should complete before A (0.1s) before C (0.15s)
        assert completion_order[0] == "hello_B"
        assert completion_order[1] == "hello_A"
        assert completion_order[2] == "hello_C"

    @pytest.mark.anyio
    async def test_preserves_record_id(self, parallel_processor_with_mocks):
        """Test that original record_id is preserved in all outputs."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="original-id-xyz", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        async for output in proc.process(context):
            assert output.record_id == "original-id-xyz"


class TestParallelProcessorErrorHandling:
    """Tests for error handling behavior."""

    @pytest.mark.anyio
    async def test_failed_processor_continues_others_default(self):
        """Test that one failing processor doesn't stop others (default behavior)."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(suffix="OK1"),
                MockProcessor(suffix="FAIL", fail=True),
                MockProcessor(suffix="OK2"),
            ],
            fail_on_error=False,
        )

        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        assert len(outputs) == 2
        contents = {o.content for o in outputs}
        assert contents == {"hello_OK1", "hello_OK2"}

    @pytest.mark.anyio
    async def test_fail_on_error_raises(self):
        """Test that fail_on_error=True raises on first failure."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(suffix="OK", delay=0.1),
                MockProcessor(suffix="FAIL", fail=True),
            ],
            fail_on_error=True,
        )

        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        with pytest.raises(ValueError, match="configured to fail"):
            async for _ in proc.process(context):
                pass

    @pytest.mark.anyio
    async def test_empty_processors_yields_nothing(self):
        """Test that empty processor list yields no outputs."""
        proc = ParallelProcessor(processors=[])
        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        assert len(outputs) == 0


class TestParallelProcessorMultiOutput:
    """Tests for processors that yield multiple outputs."""

    @pytest.mark.anyio
    async def test_handles_multi_output_processors(self):
        """Test that processors yielding multiple records work correctly."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(suffix="Single", multi_output=1),
                MockProcessor(suffix="Multi", multi_output=3),
            ]
        )

        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        assert len(outputs) == 4
        contents = {o.content for o in outputs}
        assert "hello_Single" in contents
        assert "hello_Multi_0" in contents
        assert "hello_Multi_1" in contents
        assert "hello_Multi_2" in contents


class TestParallelProcessorWithVariantProcessor:
    """Integration tests combining ParallelProcessor with VariantProcessor pattern."""

    @pytest.mark.anyio
    async def test_parallel_wraps_variant_like_processors(self):
        """Test ParallelProcessor wrapping processors that simulate VariantProcessor behavior."""

        class LLMVariantSimulator(ProcessorCore):
            """Simulates VariantProcessor(LLMCore) with 3 model variants."""

            async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[Any, None]:
                for model in ["gpt", "claude", "gemini"]:
                    yield context.record.model_copy(update={"content": f"{context.record.content}_llm_{model}"})

        class APIVariantSimulator(ProcessorCore):
            """Simulates VariantProcessor(Cope) with 2 repetitions."""

            async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[Any, None]:
                for run in [1, 2]:
                    yield context.record.model_copy(update={"content": f"{context.record.content}_api_run{run}"})

        proc = ParallelProcessor(processors=[LLMVariantSimulator(), APIVariantSimulator()])

        record = BaseRecord(record_id="test-1", content="input")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        assert len(outputs) == 5
        contents = {o.content for o in outputs}
        assert "input_llm_gpt" in contents
        assert "input_llm_claude" in contents
        assert "input_llm_gemini" in contents
        assert "input_api_run1" in contents
        assert "input_api_run2" in contents


class TestParallelProcessorFlush:
    """Tests for flush behavior."""

    @pytest.mark.anyio
    async def test_flush_delegates_to_children(self):
        """Test that flush() delegates to all child processors."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(suffix="A"),
                MockProcessor(suffix="B"),
            ],
        )

        # Default ProcessorCore.flush() yields nothing
        outputs = []
        async for output in proc.flush():
            outputs.append(output)
        assert len(outputs) == 0
