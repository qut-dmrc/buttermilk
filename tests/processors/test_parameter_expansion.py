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
