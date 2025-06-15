"""Unit tests for core types in buttermilk._core.types module."""

import pytest
from buttermilk._core.types import Record


def test_record_creation():
    """Test that Record can be created with basic content."""
    record = Record(
        content="Test content",
        mime="text/plain"
    )
    assert record.content == "Test content"
    assert record.mime == "text/plain"
    assert record.record_id is not None  # Should auto-generate


def test_record_with_metadata():
    """Test Record creation with metadata."""
    metadata = {"source": "test", "category": "unit_test"}
    record = Record(
        content="Test with metadata",
        mime="text/plain",
        metadata=metadata
    )
    assert record.metadata == metadata
    assert record.metadata["source"] == "test"


def test_record_serialization():
    """Test that Record can be serialized and deserialized."""
    record = Record(
        content="Serialization test",
        mime="text/plain",
        metadata={"test": True}
    )
    
    # Test model_dump
    dumped = record.model_dump()
    assert dumped["content"] == "Serialization test"
    assert dumped["mime"] == "text/plain"
    assert dumped["metadata"]["test"] is True
    
    # Test that we can reconstruct from dumped data
    new_record = Record(**dumped)
    assert new_record.content == record.content
    assert new_record.mime == record.mime
    assert new_record.metadata == record.metadata


def test_record_with_custom_id():
    """Test Record with a custom record_id."""
    custom_id = "test_record_123"
    record = Record(
        content="Custom ID test",
        mime="text/plain",
        record_id=custom_id
    )
    assert record.record_id == custom_id


def test_record_equality():
    """Test Record equality comparison."""
    record1 = Record(content="Same content", mime="text/plain", record_id="test1")
    record2 = Record(content="Same content", mime="text/plain", record_id="test1")
    record3 = Record(content="Different content", mime="text/plain", record_id="test1")
    
    assert record1 == record2
    assert record1 != record3