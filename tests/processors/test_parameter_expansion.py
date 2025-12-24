"""Tests for ParameterExpansionProcessor config and protocol."""

import pytest

from buttermilk.processors.unified_processors import ParameterExpansionProcessor


class TestParameterExpansionProcessorConfig:
    """Test Pydantic model creation and validation."""

    def test_instantiation_with_variants(self):
        """Test that processor can be instantiated with variants dict."""
        processor = ParameterExpansionProcessor(
            variants={"criteria": ["A", "B"], "model": ["X", "Y"]}
        )
        assert processor.variants == {"criteria": ["A", "B"], "model": ["X", "Y"]}

    def test_instantiation_without_variants_fails(self):
        """Test that processor requires variants (fail-fast, no silent defaults)."""
        with pytest.raises(ValueError, match="Field required"):
            ParameterExpansionProcessor()


class TestParameterExpansionProcessorProtocol:
    """Test Processor protocol implementation."""

    def test_implements_processor_protocol(self):
        """Test that processor implements required async _process_record method."""
        processor = ParameterExpansionProcessor(variants={"criteria": ["A"]})
        assert hasattr(processor, "_process_record")
        assert callable(getattr(processor, "_process_record"))


class TestParameterExpansionProcessorLogic:
    """Test core expansion logic."""

    @pytest.mark.anyio
    async def test_expands_cartesian_product(self):
        """Verify cartesian product expansion creates correct number of records."""
        from buttermilk._core.types import BaseRecord
        from buttermilk._core.processing_context import ProcessingContext

        processor = ParameterExpansionProcessor(
            variants={"criteria": ["A", "B"], "model": ["X", "Y"]}
        )

        record = BaseRecord(
            record_id="test-001",
            content="test content",
            metadata={"original_key": "original_value"},
        )

        # Create minimal ProcessingContext
        context = ProcessingContext(
            session_id="test-session",
            record=record,
        )

        results = []
        async for output in processor._process_record(context):
            results.append(output)

        # Should produce 2 × 2 = 4 records
        assert len(results) == 4

        # Verify each record has correct metadata
        for r in results:
            assert "criteria" in r.metadata
            assert "model" in r.metadata
            assert r.metadata["original_key"] == "original_value"  # Preserved
            assert "expansion_source_id" in r.metadata
            assert r.metadata["expansion_source_id"] == "test-001"

        # Verify unique record_ids
        record_ids = [r.record_id for r in results]
        assert len(set(record_ids)) == 4  # All unique

    @pytest.mark.anyio
    async def test_single_variant_expansion(self):
        """Verify single variant key expands to correct number of records."""
        from buttermilk._core.types import BaseRecord
        from buttermilk._core.processing_context import ProcessingContext

        processor = ParameterExpansionProcessor(
            variants={"criteria": ["A", "B", "C"]}
        )
        record = BaseRecord(record_id="test-002", content="test", metadata={})
        context = ProcessingContext(session_id="test-session", record=record)

        results = [r async for r in processor._process_record(context)]

        assert len(results) == 3
        criteria_values = [r.metadata["criteria"] for r in results]
        assert set(criteria_values) == {"A", "B", "C"}

    @pytest.mark.anyio
    async def test_preserves_original_metadata(self):
        """Verify original metadata is preserved in expanded records."""
        from buttermilk._core.types import BaseRecord
        from buttermilk._core.processing_context import ProcessingContext

        processor = ParameterExpansionProcessor(
            variants={"model": ["X"]}
        )
        record = BaseRecord(
            record_id="test-003",
            content="test",
            metadata={"existing_key": "existing_value", "score": 0.95}
        )
        context = ProcessingContext(session_id="test-session", record=record)

        results = [r async for r in processor._process_record(context)]

        assert len(results) == 1
        assert results[0].metadata["existing_key"] == "existing_value"
        assert results[0].metadata["score"] == 0.95
        assert results[0].metadata["model"] == "X"

    @pytest.mark.anyio
    async def test_unique_record_ids_generated(self):
        """Verify record_ids follow pattern and are unique."""
        from buttermilk._core.types import BaseRecord
        from buttermilk._core.processing_context import ProcessingContext

        processor = ParameterExpansionProcessor(
            variants={"a": ["1", "2"], "b": ["x", "y"]}
        )
        record = BaseRecord(record_id="orig", content="test", metadata={})
        context = ProcessingContext(session_id="test-session", record=record)

        results = [r async for r in processor._process_record(context)]

        record_ids = [r.record_id for r in results]
        assert len(set(record_ids)) == 4  # All unique
        # Verify pattern: original_a=value_b=value (sorted keys)
        assert all(r.record_id.startswith("orig_") for r in results)
        assert all("a=" in r.record_id and "b=" in r.record_id for r in results)
