"""Tests for VariantProcessor."""

import asyncio

import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.processors.variants import VariantProcessor


class MockProcessor:
    """Simple mock processor for testing."""

    def __init__(self, suffix: str = "", delay: float = 0.0, fail: bool = False):
        self.suffix = suffix
        self.delay = delay
        self.fail = fail
        self.call_count = 0

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

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise ValueError(f"MockProcessor configured to fail: {self.suffix}")

        # Yield modified record
        yield record.model_copy(
            update={
                "content": f"{record.content}_{self.suffix}",
            }
        )


class TestVariantProcessorUnit:
    """Unit tests for VariantProcessor."""

    def test_instantiation_no_variants(self):
        """Test instantiation with no variants creates single processor."""
        # This will fail because we need a real processor class path
        # But we can test the config validation
        with pytest.raises(ValueError, match="Failed to load processor class"):
            VariantProcessor(
                processor_obj="nonexistent.Processor",
                variants={},
                parameters={"key": "value"},
            )

    def test_instantiation_with_mock_path(self):
        """Test that instantiation tries to load the processor class."""
        # VariantProcessor requires a valid processor_obj path
        # This tests that validation happens at instantiation
        with pytest.raises(ValueError):
            VariantProcessor(
                processor_obj="invalid.path.Processor",
                variants={"param": ["a", "b"]},
            )


class TestVariantProcessorIntegration:
    """Integration tests using mock processors injected after init."""

    @pytest.fixture
    def variant_processor_with_mocks(self):
        """Create VariantProcessor and inject mock processors."""
        # Create with a placeholder (will fail to instantiate)
        # We'll manually inject mock processors
        proc = object.__new__(VariantProcessor)
        # Initialize Pydantic model fields manually
        proc.__dict__.update(
            {
                "processor_obj": "mock.Processor",
                "variants": {"suffix": ["A", "B", "C"]},
                "num_runs": 1,
                "parameters": {},
                "fail_on_error": False,
            }
        )
        # Inject mock processors directly
        proc._processors = [
            MockProcessor(suffix="A", delay=0.1),
            MockProcessor(suffix="B", delay=0.05),
            MockProcessor(suffix="C", delay=0.15),
        ]
        return proc

    @pytest.mark.anyio
    async def test_parallel_execution_yields_all_results(
        self, variant_processor_with_mocks
    ):
        """Test that all variants produce outputs."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(
            record, processor_stage="test/00.Variant/abc123"
        ):
            outputs.append(output)

        assert len(outputs) == 3
        contents = {o.content for o in outputs}
        assert contents == {"hello_A", "hello_B", "hello_C"}

    @pytest.mark.anyio
    async def test_variant_metadata_added(self, variant_processor_with_mocks):
        """Test that variant metadata is added to outputs."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(
            record, processor_stage="test/00.Variant/abc123"
        ):
            outputs.append(output)

        for output in outputs:
            assert "variant" in output.metadata
            assert "index" in output.metadata["variant"]
            assert "total" in output.metadata["variant"]
            assert output.metadata["variant"]["total"] == 3
            assert output.metadata["variant"]["stage"] == "test/00.Variant/abc123"

    @pytest.mark.anyio
    async def test_results_stream_as_completed(self, variant_processor_with_mocks):
        """Test that faster variants yield results first."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")

        # Track order of completion
        completion_order = []
        async for output in proc.process(
            record, processor_stage="test/00.Variant/abc123"
        ):
            completion_order.append(output.content)

        # B (0.05s) should complete before A (0.1s) before C (0.15s)
        assert completion_order[0] == "hello_B"
        assert completion_order[1] == "hello_A"
        assert completion_order[2] == "hello_C"

    @pytest.mark.anyio
    async def test_failed_variant_continues_others(self):
        """Test that one failing variant doesn't stop others."""
        proc = object.__new__(VariantProcessor)
        proc.__dict__.update(
            {
                "processor_obj": "mock.Processor",
                "variants": {},
                "num_runs": 1,
                "parameters": {},
                "fail_on_error": False,
            }
        )
        proc._processors = [
            MockProcessor(suffix="OK1"),
            MockProcessor(suffix="FAIL", fail=True),
            MockProcessor(suffix="OK2"),
        ]

        record = BaseRecord(record_id="test-1", content="hello")

        outputs = []
        async for output in proc.process(
            record, processor_stage="test/00.Variant/abc123"
        ):
            outputs.append(output)

        # Should get 3 outputs: 2 successful + 1 error record
        assert len(outputs) == 3

        # Check successful outputs
        successful = [o for o in outputs if not o.error]
        assert len(successful) == 2
        contents = {o.content for o in successful}
        assert contents == {"hello_OK1", "hello_OK2"}

        # Check error record
        error_records = [o for o in outputs if o.error]
        assert len(error_records) == 1
        assert "configured to fail" in str(error_records[0].error)

    @pytest.mark.anyio
    async def test_fail_on_error_raises(self):
        """Test that fail_on_error=True raises on first failure."""
        proc = object.__new__(VariantProcessor)
        proc.__dict__.update(
            {
                "processor_obj": "mock.Processor",
                "variants": {},
                "num_runs": 1,
                "parameters": {},
                "fail_on_error": True,
            }
        )
        proc._processors = [
            MockProcessor(suffix="OK"),
            MockProcessor(suffix="FAIL", fail=True),
        ]

        record = BaseRecord(record_id="test-1", content="hello")

        with pytest.raises(ValueError, match="configured to fail"):
            async for _ in proc.process(
                record, processor_stage="test/00.Variant/abc123"
            ):
                pass

    @pytest.mark.anyio
    async def test_preserves_record_id(self, variant_processor_with_mocks):
        """Test that original record_id is preserved in outputs."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="original-id-123", content="hello")

        async for output in proc.process(
            record, processor_stage="test/00.Variant/abc123"
        ):
            assert output.record_id == "original-id-123"


class TestProcessorVariantsConfig:
    """Test ProcessorVariants config class used by VariantProcessor."""

    def test_expand_single_variant(self):
        """Test expanding a single variant parameter."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        # Use a real processor path that exists
        cfg = ProcessorVariants(
            processor_obj="buttermilk.processors.JMESPathTransform",
            variants={"expression": ["content", "metadata"]},
            parameters={"output_field": "result"},
        )

        configs = cfg.get_configs()
        assert len(configs) == 2

        # Check parameters were expanded correctly
        params = [c[1] for c in configs]
        expressions = {p["expression"] for p in params}
        assert expressions == {"content", "metadata"}

        # Check base params preserved
        for p in params:
            assert p["output_field"] == "result"

    def test_expand_multiple_variants(self):
        """Test expanding multiple variant parameters (cartesian product)."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        cfg = ProcessorVariants(
            processor_obj="buttermilk.processors.JMESPathTransform",
            variants={
                "expression": ["content", "metadata"],
                "output_field": ["out1", "out2"],
            },
        )

        configs = cfg.get_configs()
        # 2 expressions × 2 output_fields = 4 combinations
        assert len(configs) == 4

    def test_num_runs_multiplies(self):
        """Test that num_runs replicates configurations."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        cfg = ProcessorVariants(
            processor_obj="buttermilk.processors.JMESPathTransform",
            variants={"expression": ["content"]},
            num_runs=3,
        )

        configs = cfg.get_configs()
        assert len(configs) == 3
