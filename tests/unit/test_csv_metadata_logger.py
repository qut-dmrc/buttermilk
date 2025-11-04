import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.processors.csv_metadata_logger import CSVMetadataLogger


@pytest.mark.anyio
async def test_csv_logger_accumulates_records():
    """Test that logger accumulates records without modifying them."""
    logger = CSVMetadataLogger(bucket="test-bucket", base_path="test-images")

    record = BaseRecord(
        record_id="r1",
        content="test prompt",
        metadata={
            "session_id": "s1",
            "model_class": "VertexImagen3Fast",
            "storage_uri": "gs://bucket/image1.png",
            "scenario": "office",
            "repetition": 0,
        },
    )

    # Process should yield record unchanged
    results = [r async for r in logger.process(record, processor_stage="log")]

    assert len(results) == 1
    assert results[0].record_id == "r1"
    assert results[0].content == "test prompt"

    # Verify internal state
    assert len(logger._accumulated_records) == 1
    assert logger._session_id == "s1"


@pytest.mark.anyio
async def test_csv_logger_captures_session_id():
    """Test that logger captures session_id from first record."""
    logger = CSVMetadataLogger(bucket="test-bucket", base_path="image-generation")

    records = [
        BaseRecord(
            record_id="r1",
            content="prompt 1",
            metadata={
                "session_id": "sess123",
                "model_class": "VertexImagen3Fast",
                "storage_uri": "gs://bucket/img1.png",
                "scenario": "office",
                "repetition": 0,
                "timestamp": "2025-01-04T10:00:00",
            },
        ),
        BaseRecord(
            record_id="r2",
            content="prompt 2",
            metadata={
                "session_id": "sess123",
                "model_class": "VertexImagen4Fast",
                "storage_uri": "gs://bucket/img2.png",
                "scenario": "coffee shop",
                "repetition": 1,
                "timestamp": "2025-01-04T10:01:00",
            },
        ),
    ]

    for rec in records:
        async for _ in logger.process(rec, processor_stage="log"):
            pass

    # Verify accumulation
    assert len(logger._accumulated_records) == 2
    assert logger._session_id == "sess123"
    assert logger._accumulated_records[0]["prompt"] == "prompt 1"
    assert logger._accumulated_records[1]["prompt"] == "prompt 2"


@pytest.mark.anyio
async def test_csv_logger_handles_empty_finalize():
    """Test finalize with no records logs warning and returns early."""
    logger = CSVMetadataLogger(bucket="test-bucket", base_path="test")

    # Finalize without processing any records - should not crash
    await logger.finalize_processing()

    # Should have logged warning (checked via logger, not asserting here)
    assert logger._session_id is None
