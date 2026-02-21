"""Tests for ParallelProcessor."""

import asyncio

import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.processors.parallel import ParallelProcessor


class MockProcessor:
    """Simple mock processor for testing.

    Follows the Processor protocol: async process() method that yields BaseRecord.
    """

    def __init__(
        self,
        name: str = "mock",
        delay: float = 0.0,
        fail: bool = False,
        multi_output: int = 1,
    ):
        self.name = name
        self.delay = delay
        self.fail = fail
        self.multi_output = multi_output
        self.call_count = 0
        self.received_records: list[BaseRecord] = []

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs,
    ):
        """Process record, optionally with delay or failure."""
        self.call_count += 1
        self.received_records.append(record)

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise ValueError(f"MockProcessor '{self.name}' configured to fail")

        # Yield one or more modified records
        for i in range(self.multi_output):
            suffix = f"_{i}" if self.multi_output > 1 else ""
            yield record.model_copy(
                update={
                    "content": f"{record.content}_{self.name}{suffix}",
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
        mock_a = MockProcessor(name="A", delay=0.1)
        mock_b = MockProcessor(name="B", delay=0.05)
        mock_c = MockProcessor(name="C", delay=0.15)

        return ParallelProcessor(processors=[mock_a, mock_b, mock_c])

    @pytest.mark.anyio
    async def test_all_processors_receive_same_input(self):
        """Test that all sub-processors receive the SAME input record."""
        mock_a = MockProcessor(name="A")
        mock_b = MockProcessor(name="B")
        mock_c = MockProcessor(name="C")

        proc = ParallelProcessor(processors=[mock_a, mock_b, mock_c])
        record = BaseRecord(record_id="test-1", content="original")

        # Consume all outputs
        async for _ in proc.process(record, processor_stage="test/parallel"):
            pass

        # All processors should have been called once
        assert mock_a.call_count == 1
        assert mock_b.call_count == 1
        assert mock_c.call_count == 1

        # All should have received the SAME record
        assert mock_a.received_records[0].record_id == "test-1"
        assert mock_b.received_records[0].record_id == "test-1"
        assert mock_c.received_records[0].record_id == "test-1"
        assert mock_a.received_records[0].content == "original"
        assert mock_b.received_records[0].content == "original"
        assert mock_c.received_records[0].content == "original"

    @pytest.mark.anyio
    async def test_yields_all_outputs(self, parallel_processor_with_mocks):
        """Test that outputs from all processors are yielded."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(record, processor_stage="test/parallel"):
            outputs.append(output)

        assert len(outputs) == 3
        contents = {o.content for o in outputs}
        assert contents == {"hello_A", "hello_B", "hello_C"}

    @pytest.mark.anyio
    async def test_parallel_metadata_added(self, parallel_processor_with_mocks):
        """Test that parallel metadata is added to outputs."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(record, processor_stage="test/parallel/stage1"):
            outputs.append(output)

        for output in outputs:
            assert "parallel" in output.metadata
            meta = output.metadata["parallel"]
            assert "processor_index" in meta
            assert meta["total_processors"] == 3
            assert meta["processor_class"] == "MockProcessor"
            assert meta["stage"] == "test/parallel/stage1"

    @pytest.mark.anyio
    async def test_results_stream_as_completed(self, parallel_processor_with_mocks):
        """Test that faster processors yield results first."""
        proc = parallel_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")

        # Track order of completion
        completion_order = []
        async for output in proc.process(record, processor_stage="test/parallel"):
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

        async for output in proc.process(record, processor_stage="test/parallel"):
            assert output.record_id == "original-id-xyz"


class TestParallelProcessorErrorHandling:
    """Tests for error handling behavior."""

    @pytest.mark.anyio
    async def test_failed_processor_continues_others_default(self):
        """Test that one failing processor doesn't stop others (default behavior)."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(name="OK1"),
                MockProcessor(name="FAIL", fail=True),
                MockProcessor(name="OK2"),
            ],
            fail_on_error=False,  # Default
        )

        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(record, processor_stage="test/parallel"):
            outputs.append(output)

        # Should get 2 successful outputs (failed processor is logged but doesn't yield)
        assert len(outputs) == 2

        # Check successful outputs
        contents = {o.content for o in outputs}
        assert contents == {"hello_OK1", "hello_OK2"}

    @pytest.mark.anyio
    async def test_fail_on_error_raises(self):
        """Test that fail_on_error=True raises on first failure."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(name="OK", delay=0.1),  # Slower, won't complete
                MockProcessor(name="FAIL", fail=True),  # Fast failure
            ],
            fail_on_error=True,
        )

        record = BaseRecord(record_id="test-1", content="hello")

        with pytest.raises(ValueError, match="configured to fail"):
            async for _ in proc.process(record, processor_stage="test/parallel"):
                pass

    @pytest.mark.anyio
    async def test_empty_processors_yields_nothing(self):
        """Test that empty processor list yields no outputs."""
        proc = ParallelProcessor(processors=[])
        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(record, processor_stage="test/parallel"):
            outputs.append(output)

        assert len(outputs) == 0


class TestParallelProcessorMultiOutput:
    """Tests for processors that yield multiple outputs."""

    @pytest.mark.anyio
    async def test_handles_multi_output_processors(self):
        """Test that processors yielding multiple records work correctly."""
        proc = ParallelProcessor(
            processors=[
                MockProcessor(name="Single", multi_output=1),
                MockProcessor(name="Multi", multi_output=3),
            ]
        )

        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(record, processor_stage="test/parallel"):
            outputs.append(output)

        # 1 from Single + 3 from Multi = 4 total
        assert len(outputs) == 4

        contents = {o.content for o in outputs}
        assert "hello_Single" in contents
        assert "hello_Multi_0" in contents
        assert "hello_Multi_1" in contents
        assert "hello_Multi_2" in contents


class TestParallelProcessorWithVariantProcessor:
    """Integration tests combining ParallelProcessor with VariantProcessor pattern.

    These tests verify that ParallelProcessor works correctly when wrapping
    VariantProcessor instances, which is the primary use case for the
    reliability pipeline.
    """

    @pytest.mark.anyio
    async def test_parallel_wraps_variant_like_processors(self):
        """Test ParallelProcessor wrapping processors that simulate VariantProcessor behavior."""
        # Simulate what VariantProcessor does: run variants of a single processor type
        # Here we simulate two "branches": LLM-like (3 variants) and API-like (2 variants)

        class LLMVariantSimulator:
            """Simulates VariantProcessor(LLMCore) with 3 model variants."""

            async def process(self, record, *, processor_stage, **kwargs):
                for model in ["gpt", "claude", "gemini"]:
<<<<<<< HEAD
                    yield record.model_copy(update={"content": f"{record.content}_llm_{model}"})
=======
                    yield record.model_copy(
                        update={"content": f"{record.content}_llm_{model}"}
                    )
>>>>>>> origin/stable

        class APIVariantSimulator:
            """Simulates VariantProcessor(Cope) with 2 repetitions."""

            async def process(self, record, *, processor_stage, **kwargs):
                for run in [1, 2]:
<<<<<<< HEAD
                    yield record.model_copy(update={"content": f"{record.content}_api_run{run}"})

        proc = ParallelProcessor(processors=[LLMVariantSimulator(), APIVariantSimulator()])
=======
                    yield record.model_copy(
                        update={"content": f"{record.content}_api_run{run}"}
                    )

        proc = ParallelProcessor(
            processors=[LLMVariantSimulator(), APIVariantSimulator()]
        )
>>>>>>> origin/stable

        record = BaseRecord(record_id="test-1", content="input")

        outputs = []
        async for output in proc.process(record, processor_stage="test/parallel"):
            outputs.append(output)

        # 3 from LLM branch + 2 from API branch = 5 total
        assert len(outputs) == 5

        contents = {o.content for o in outputs}
        assert "input_llm_gpt" in contents
        assert "input_llm_claude" in contents
        assert "input_llm_gemini" in contents
        assert "input_api_run1" in contents
        assert "input_api_run2" in contents
