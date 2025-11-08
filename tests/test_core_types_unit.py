"""Unit tests for core types in buttermilk._core.types module."""

import time

from buttermilk._core.types import ProcessingSummary, Record


def test_record_creation():
    """Test that Record can be created with basic content."""
    record = Record(content="Test content", mime="text/plain")
    assert record.content == "Test content"
    assert record.mime == "text/plain"
    assert record.record_id is not None  # Should auto-generate


def test_record_with_metadata():
    """Test Record creation with metadata."""
    metadata = {"source": "test", "category": "unit_test"}
    record = Record(content="Test with metadata", mime="text/plain", metadata=metadata)
    assert record.metadata == metadata
    assert record.metadata["source"] == "test"


def test_record_serialization():
    """Test that Record can be serialized and deserialized."""
    record = Record(
        content="Serialization test", mime="text/plain", metadata={"test": True}
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
    record = Record(content="Custom ID test", mime="text/plain", record_id=custom_id)
    assert record.record_id == custom_id


def test_record_equality():
    """Test Record equality comparison."""
    record1 = Record(content="Same content", mime="text/plain", record_id="test1")
    record2 = Record(content="Same content", mime="text/plain", record_id="test1")
    record3 = Record(content="Different content", mime="text/plain", record_id="test1")

    assert record1 == record2
    assert record1 != record3


def test_record_hash_computation():
    """Test that record_hash is computed correctly from as_markdown() output."""
    record = Record(
        content="Test content for hashing",
        metadata={"title": "Test Record"},
        record_id="test_hash_123",
    )

    # Access the computed field to trigger computation
    hash_value = record.record_hash

    # Verify hash is a valid SHA256 hex string (64 characters, hex)
    assert len(hash_value) == 64
    assert all(c in "0123456789abcdef" for c in hash_value)

    # Verify hash is stored in metadata
    assert record.metadata["record_hash"] == hash_value

    # Verify hash is consistent - same content should produce same hash
    record2 = Record(
        content="Test content for hashing",
        metadata={"title": "Test Record"},
        record_id="test_hash_123",
    )
    assert record2.record_hash == hash_value


def test_record_hash_uniqueness():
    """Test that different records produce different hashes."""
    record1 = Record(content="Content A", record_id="test1")
    record2 = Record(content="Content B", record_id="test2")

    hash1 = record1.record_hash
    hash2 = record2.record_hash

    assert hash1 != hash2
    assert record1.metadata["record_hash"] == hash1
    assert record2.metadata["record_hash"] == hash2


def test_record_hash_changes_with_content():
    """Test that record hash changes when content changes."""
    record1 = Record(content="Original content", record_id="test_change")
    original_hash = record1.record_hash

    # Create new record with different content
    record2 = Record(content="Modified content", record_id="test_change")
    new_hash = record2.record_hash

    assert original_hash != new_hash
    assert record2.metadata["record_hash"] == new_hash


def test_ground_truth_hash_none():
    """Test ground_truth_hash when ground_truth is None."""
    record = Record(content="Test content")

    # Access the computed field
    gt_hash = record.ground_truth_hash

    assert gt_hash is None
    assert record.metadata["ground_truth_hash"] is None


def test_ground_truth_hash_computation():
    """Test ground_truth_hash computation with actual ground truth data."""
    ground_truth_data = {"answer": "test", "category": "unit_test", "score": 5}
    record = Record(content="Test content", ground_truth=ground_truth_data)

    # Access the computed field
    gt_hash = record.ground_truth_hash

    # Verify hash is a valid SHA256 hex string
    assert len(gt_hash) == 64
    assert all(c in "0123456789abcdef" for c in gt_hash)

    # Verify hash is stored in metadata
    assert record.metadata["ground_truth_hash"] == gt_hash


def test_ground_truth_hash_consistency():
    """Test that same ground truth data produces same hash."""
    ground_truth_data = {"answer": "test", "score": 5}

    record1 = Record(content="Content 1", ground_truth=ground_truth_data)
    record2 = Record(content="Content 2", ground_truth=ground_truth_data)

    hash1 = record1.ground_truth_hash
    hash2 = record2.ground_truth_hash

    # Same ground truth data should produce same hash regardless of other content
    assert hash1 == hash2


def test_ground_truth_hash_key_order_independence():
    """Test that ground truth hash is consistent regardless of key order."""
    # Same data in different key orders
    gt1 = {"b": 2, "a": 1, "c": 3}
    gt2 = {"a": 1, "c": 3, "b": 2}

    record1 = Record(content="Test 1", ground_truth=gt1)
    record2 = Record(content="Test 2", ground_truth=gt2)

    # Should produce same hash due to sorted keys in JSON serialization
    assert record1.ground_truth_hash == record2.ground_truth_hash


def test_ground_truth_hash_uniqueness():
    """Test that different ground truth data produces different hashes."""
    record1 = Record(content="Test", ground_truth={"answer": "A"})
    record2 = Record(content="Test", ground_truth={"answer": "B"})

    hash1 = record1.ground_truth_hash
    hash2 = record2.ground_truth_hash

    assert hash1 != hash2


def test_ground_truth_hash_complex_data():
    """Test ground truth hash with complex nested data structures."""
    complex_gt = {
        "answers": ["A", "B", "C"],
        "metadata": {"source": "test", "nested": {"deep": "value"}},
        "scores": [1, 2, 3],
    }

    record = Record(content="Test", ground_truth=complex_gt)
    gt_hash = record.ground_truth_hash

    # Should handle complex data without errors
    assert len(gt_hash) == 64
    assert record.metadata["ground_truth_hash"] == gt_hash


def test_hash_fields_metadata_accessibility():
    """Test that hash values are easily accessible via metadata."""
    record = Record(
        content="Test content",
        ground_truth={"test": "data"},
        metadata={"existing": "data"},
    )

    # Trigger computation by accessing computed fields
    record_hash = record.record_hash
    gt_hash = record.ground_truth_hash

    # Verify both hashes are accessible via metadata
    assert record.metadata["record_hash"] == record_hash
    assert record.metadata["ground_truth_hash"] == gt_hash

    # Verify existing metadata is preserved
    assert record.metadata["existing"] == "data"


def test_hash_fields_excluded_from_dump():
    """Test that computed hash fields are excluded from model_dump by default."""
    record = Record(content="Test content", ground_truth={"test": "data"})

    # Trigger hash computation
    _ = record.record_hash
    _ = record.ground_truth_hash

    # Dump should exclude computed fields
    dumped = record.model_dump()
    assert "record_hash" not in dumped
    assert "ground_truth_hash" not in dumped

    # But metadata should contain the hash values
    assert "record_hash" in dumped["metadata"]
    assert "ground_truth_hash" in dumped["metadata"]


# ProcessingSummary tests


def test_processing_summary_creation():
    """Test ProcessingSummary can be created with default values."""
    summary = ProcessingSummary()

    assert summary.attempted == 0
    assert summary.processed == 0
    assert summary.skipped == 0
    assert summary.failed == 0
    assert summary.start_time > 0  # Should auto-initialize


def test_processing_summary_increment_attempted():
    """Test increment_attempted method."""
    summary = ProcessingSummary()

    summary.increment_attempted()
    assert summary.attempted == 1

    summary.increment_attempted()
    assert summary.attempted == 2


def test_processing_summary_increment_processed():
    """Test increment_processed method."""
    summary = ProcessingSummary()

    summary.increment_processed()
    assert summary.processed == 1

    summary.increment_processed()
    assert summary.processed == 2


def test_processing_summary_increment_skipped():
    """Test increment_skipped method."""
    summary = ProcessingSummary()

    summary.increment_skipped()
    assert summary.skipped == 1

    summary.increment_skipped()
    assert summary.skipped == 2


def test_processing_summary_increment_failed():
    """Test increment_failed method."""
    summary = ProcessingSummary()

    summary.increment_failed()
    assert summary.failed == 1

    summary.increment_failed()
    assert summary.failed == 2


def test_processing_summary_duration_ms():
    """Test duration_ms calculation."""
    summary = ProcessingSummary()

    # Should be very small initially
    duration1 = summary.duration_ms()
    assert duration1 >= 0
    assert duration1 < 100  # Less than 100ms

    # Wait a bit and check duration increased
    time.sleep(0.1)
    duration2 = summary.duration_ms()
    assert duration2 > duration1
    assert duration2 >= 100  # At least 100ms


def test_processing_summary_success_rate_zero_attempted():
    """Test success_rate when nothing attempted."""
    summary = ProcessingSummary()

    assert summary.success_rate() == 0.0


def test_processing_summary_success_rate_all_successful():
    """Test success_rate when all processed successfully."""
    summary = ProcessingSummary()

    summary.increment_attempted()
    summary.increment_processed()
    summary.increment_attempted()
    summary.increment_processed()

    assert summary.success_rate() == 1.0


def test_processing_summary_success_rate_partial_success():
    """Test success_rate with partial success."""
    summary = ProcessingSummary()

    # 3 attempted, 2 processed
    summary.increment_attempted()
    summary.increment_processed()
    summary.increment_attempted()
    summary.increment_processed()
    summary.increment_attempted()
    summary.increment_failed()

    assert summary.attempted == 3
    assert summary.processed == 2
    assert summary.failed == 1
    assert abs(summary.success_rate() - 0.6667) < 0.001


def test_processing_summary_success_rate_all_failed():
    """Test success_rate when all failed."""
    summary = ProcessingSummary()

    summary.increment_attempted()
    summary.increment_failed()
    summary.increment_attempted()
    summary.increment_failed()

    assert summary.success_rate() == 0.0


def test_processing_summary_format_for_console():
    """Test format_for_console output."""
    summary = ProcessingSummary()

    summary.increment_attempted()
    summary.increment_processed()
    summary.increment_attempted()
    summary.increment_skipped()
    summary.increment_attempted()
    summary.increment_failed()

    formatted = summary.format_for_console()

    # Verify it's a string with expected content
    assert isinstance(formatted, str)
    assert "attempted=3" in formatted
    assert "processed=1" in formatted
    assert "skipped=1" in formatted
    assert "failed=1" in formatted
    assert "success=" in formatted
    assert "duration=" in formatted
    assert "%" in formatted  # success percentage
    assert "s)" in formatted  # seconds


def test_processing_summary_as_dict():
    """Test as_dict export."""
    summary = ProcessingSummary()

    summary.increment_attempted()
    summary.increment_processed()
    summary.increment_skipped()

    result = summary.as_dict()

    # Verify all expected fields
    assert result["attempted"] == 1
    assert result["processed"] == 1
    assert result["skipped"] == 1
    assert result["failed"] == 0
    assert "start_time" in result
    assert "duration_ms" in result
    assert "success_rate" in result

    # Verify computed fields are correct
    assert result["duration_ms"] >= 0
    assert result["success_rate"] == 1.0


def test_processing_summary_as_dict_structure():
    """Test as_dict returns correct structure and types."""
    summary = ProcessingSummary()

    result = summary.as_dict()

    # Check types
    assert isinstance(result["attempted"], int)
    assert isinstance(result["processed"], int)
    assert isinstance(result["skipped"], int)
    assert isinstance(result["failed"], int)
    assert isinstance(result["start_time"], float)
    assert isinstance(result["duration_ms"], int)
    assert isinstance(result["success_rate"], float)


def test_processing_summary_typical_workflow():
    """Test ProcessingSummary in a typical workflow scenario."""
    summary = ProcessingSummary()

    # Simulate processing 5 items: 3 success, 1 skip, 1 fail
    items = ["item1", "item2", "item3", "item4", "item5"]

    for i, item in enumerate(items):
        summary.increment_attempted()

        if i == 1:
            # Skip item 2
            summary.increment_skipped()
        elif i == 4:
            # Fail item 5
            summary.increment_failed()
        else:
            # Process successfully
            summary.increment_processed()

    # Verify final counts
    assert summary.attempted == 5
    assert summary.processed == 3
    assert summary.skipped == 1
    assert summary.failed == 1

    # Verify success rate (3/5 = 60%)
    assert abs(summary.success_rate() - 0.6) < 0.001

    # Verify format_for_console works
    formatted = summary.format_for_console()
    assert "attempted=5" in formatted
    assert "processed=3" in formatted
