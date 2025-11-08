"""Test to verify ProcessingResult correctly accepts BaseRecord after fix.

This test confirms that the ProcessingResult validation error with BaseRecord
has been resolved.
"""

from buttermilk._core.types import BaseRecord, ProcessingResult, Record


def test_processingresult_accepts_record():
    """ProcessingResult should accept Record type."""
    # Create a Record (full record with content)
    record = Record(
        record_id="test_123", content="Test content", metadata={"title": "Test Record"}
    )

    # This should work - ProcessingResult accepts Record
    result = ProcessingResult(
        record=record,
        status="processed",
        reason="successfully processed",
        chunks_created=5,
        embedding_model="text-embedding-ada-002",
        processing_time_ms=100.0,
    )

    assert result.record == record
    assert result.status == "processed"


def test_processingresult_accepts_baserecord():
    """ProcessingResult should now accept BaseRecord (this was the bug)."""
    # Create a BaseRecord (minimal record without content)
    base_record = BaseRecord(
        record_id="test_456", metadata={"title": "Test Base Record"}
    )

    # This should now work after the fix
    result = ProcessingResult(
        record=base_record,  # Passing BaseRecord instead of Record
        status="processed",
        reason="successfully processed",
        chunks_created=10,
        embedding_model="text-embedding-ada-002",
        processing_time_ms=200.0,
    )

    # Verify it works correctly
    assert result.record == base_record
    assert result.status == "processed"
    assert result.chunks_created == 10
    assert result.embedding_model == "text-embedding-ada-002"


def test_processingresult_accepts_none():
    """ProcessingResult should accept None for record field."""
    # This should work - ProcessingResult accepts None
    result = ProcessingResult(
        record=None,
        status="skipped",
        reason="Already processed",
        chunks_created=0,
        embedding_model="text-embedding-ada-002",
        processing_time_ms=50.0,
    )

    assert result.record is None
    assert result.status == "skipped"
    assert result.reason == "Already processed"


def test_processingresult_with_different_statuses():
    """Test ProcessingResult with various status values."""
    base_record = BaseRecord(record_id="test_789")

    # Test "processed" status
    result1 = ProcessingResult(record=base_record, status="processed", chunks_created=5)
    assert result1.status == "processed"

    # Test "skipped" status
    result2 = ProcessingResult(record=None, status="skipped", reason="Duplicate record")
    assert result2.status == "skipped"
    assert result2.reason == "Duplicate record"

    # Test "failed" status
    result3 = ProcessingResult(
        record=None, status="failed", reason="Processing error occurred"
    )
    assert result3.status == "failed"
    assert result3.reason == "Processing error occurred"
