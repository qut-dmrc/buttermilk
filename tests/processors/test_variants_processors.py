"""Tests for VariantProcessor."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.processors.variants import VariantProcessor


class MockProcessor(ProcessorCore):
    """Simple mock processor for testing.

    A proper ProcessorCore subclass with _process_record.
    """

    suffix: str = ""
    delay: float = 0.0
    fail: bool = False
    call_count: int = 0

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """Process record, optionally with delay or failure."""
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise ValueError(f"MockProcessor configured to fail: {self.suffix}")

        yield context.record.model_copy(
            update={
                "content": f"{context.record.content}_{self.suffix}",
            }
        )


class TestVariantProcessorUnit:
    """Unit tests for VariantProcessor."""

    def test_instantiation_no_variants(self):
        """Test instantiation with no variants creates single processor."""
        with pytest.raises(ValueError, match="Failed to load processor class"):
            VariantProcessor(
                processor_obj="nonexistent.Processor",
                variants={},
                parameters={"key": "value"},
            )

    def test_instantiation_with_mock_path(self):
        """Test that instantiation tries to load the processor class."""
        with pytest.raises(ValueError):
            VariantProcessor(
                processor_obj="invalid.path.Processor",
                variants={"param": ["a", "b"]},
            )

    def test_fail_fast_invalid_variant_keys(self):
        """Test that invalid variant keys raise ValueError at init time."""
        with pytest.raises((ValueError, Exception), match="does not accept variant params"):
            VariantProcessor(
                processor_obj="buttermilk.processors.JMESPathTransform",
                variants={"nonexistent_field": ["a", "b"]},
            )

    def test_fail_fast_invalid_parameter_keys(self):
        """Test that invalid parameter keys raise ValueError at init time."""
        with pytest.raises((ValueError, Exception), match="does not accept parameters"):
            VariantProcessor(
                processor_obj="buttermilk.processors.JMESPathTransform",
                parameters={"nonexistent_field": "value"},
            )

    def test_valid_variant_keys_accepted(self):
        """Test that valid variant keys are accepted."""
        proc = VariantProcessor(
            processor_obj="buttermilk.processors.JMESPathTransform",
            variants={"mappings": [{"out1": "content"}, {"out2": "metadata"}]},
        )
        assert len(proc._processors) == 2


def _make_variant_processor_with_mocks(
    mocks: list[MockProcessor],
    fail_on_error: bool = False,
) -> VariantProcessor:
    """Create VariantProcessor bypassing model_post_init, injecting mock processors."""
    with patch.object(VariantProcessor, "model_post_init", lambda self, ctx: None):
        proc = VariantProcessor(
            processor_obj="mock.Processor",
            variants={"suffix": ["A", "B", "C"]},
            parameters={},
            fail_on_error=fail_on_error,
        )
    proc._processors = mocks
    return proc


class TestVariantProcessorIntegration:
    """Integration tests using mock processors injected after init."""

    @pytest.fixture
    def variant_processor_with_mocks(self):
        """Create VariantProcessor and inject mock processors."""
        return _make_variant_processor_with_mocks(
            mocks=[
                MockProcessor(suffix="A", delay=0.1),
                MockProcessor(suffix="B", delay=0.05),
                MockProcessor(suffix="C", delay=0.15),
            ],
            fail_on_error=False,
        )

    @pytest.mark.anyio
    async def test_parallel_execution_yields_all_results(self, variant_processor_with_mocks):
        """Test that all variants produce outputs."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        assert len(outputs) == 3
        contents = {o.content for o in outputs}
        assert contents == {"hello_A", "hello_B", "hello_C"}

    @pytest.mark.anyio
    async def test_variant_metadata_added(self, variant_processor_with_mocks):
        """Test that variant metadata is added to outputs."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="test-1", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        outputs = []
        async for output in proc.process(context):
            outputs.append(output)

        for output in outputs:
            assert "variant" in output.metadata
            assert "index" in output.metadata["variant"]
            assert "total" in output.metadata["variant"]
            assert output.metadata["variant"]["total"] == 3
            assert "params" in output.metadata["variant"]

    @pytest.mark.anyio
    async def test_results_stream_as_completed(self, variant_processor_with_mocks):
        """Test that faster variants yield results first."""
        proc = variant_processor_with_mocks
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
    async def test_failed_variant_continues_others(self):
        """Test that one failing variant doesn't stop others."""
        proc = _make_variant_processor_with_mocks(
            mocks=[
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
        proc = _make_variant_processor_with_mocks(
            mocks=[
                MockProcessor(suffix="OK"),
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
    async def test_preserves_record_id(self, variant_processor_with_mocks):
        """Test that original record_id is preserved in outputs."""
        proc = variant_processor_with_mocks
        record = BaseRecord(record_id="original-id-123", content="hello")
        context = ProcessingContext(session_id="test", record=record)

        async for output in proc.process(context):
            assert output.record_id == "original-id-123"

    @pytest.mark.anyio
    async def test_flush_delegates_to_children(self):
        """Test that flush() delegates to all child processors."""
        proc = _make_variant_processor_with_mocks(
            mocks=[
                MockProcessor(suffix="A"),
                MockProcessor(suffix="B"),
            ],
        )

        # flush() should not raise - default ProcessorCore.flush() yields nothing
        outputs = []
        async for output in proc.flush():
            outputs.append(output)
        assert len(outputs) == 0


class TestProcessorVariantsConfig:
    """Test ProcessorVariants config class used by VariantProcessor."""

    def test_expand_single_variant(self):
        """Test expanding a single variant parameter."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        cfg = ProcessorVariants(
            processor_obj="buttermilk.processors.JMESPathTransform",
            variants={"mappings": [{"out1": "content"}, {"out2": "metadata"}]},
        )

        configs = cfg.get_configs()
        assert len(configs) == 2

        params = [c[1] for c in configs]
        mappings_set = [p["mappings"] for p in params]
        assert {"out1": "content"} in mappings_set
        assert {"out2": "metadata"} in mappings_set

    def test_expand_multiple_variants(self):
        """Test expanding multiple variant parameters (cartesian product).

        Uses MockProcessor since JMESPathTransform only has one field.
        """
        from buttermilk._core.pipeline_config import ProcessorVariants

        # Use the test MockProcessor path for multi-field expansion
        cfg = ProcessorVariants(
            processor_obj=f"{MockProcessor.__module__}.MockProcessor",
            variants={
                "suffix": ["A", "B"],
                "delay": [0.0, 0.1],
            },
        )

        configs = cfg.get_configs()
        # 2 suffixes x 2 delays = 4 combinations
        assert len(configs) == 4

    def test_num_runs_does_not_multiply_configs(self):
        """Test that num_runs doesn't multiply configs (it's handled at source level)."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        cfg = ProcessorVariants(
            processor_obj="buttermilk.processors.JMESPathTransform",
            variants={"mappings": [{"out": "content"}]},
            num_runs=3,
        )

        configs = cfg.get_configs()
        assert len(configs) == 1
