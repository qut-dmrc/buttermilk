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
