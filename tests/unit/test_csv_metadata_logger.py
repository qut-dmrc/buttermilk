import pandas as pd
import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.processors.csv_metadata_logger import CSVMetadataLogger


@pytest.mark.anyio
async def test_csv_logger_accumulates_records(tmp_path):
    """Test that logger accumulates records without modifying them."""
    output_path = tmp_path / "log.csv"
    logger = CSVMetadataLogger(output_path=str(output_path))

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

    # CSV not written yet (happens in finalize)
    assert not output_path.exists()


@pytest.mark.anyio
async def test_csv_logger_writes_on_finalize(tmp_path):
    """Test CSV is written with correct schema on finalize."""
    output_path = tmp_path / "generation_log.csv"
    logger = CSVMetadataLogger(output_path=str(output_path))

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

    # Finalize writes CSV
    await logger.finalize_processing()

    assert output_path.exists()
    df = pd.read_csv(output_path)

    assert len(df) == 2
    assert list(df.columns) == ["prompt", "model", "timestamp", "filename", "scenario", "session_id", "repetition"]
    assert df["prompt"].tolist() == ["prompt 1", "prompt 2"]
    assert df["session_id"].unique().tolist() == ["sess123"]


@pytest.mark.anyio
async def test_csv_logger_handles_empty_finalize(tmp_path):
    """Test finalize with no records doesn't crash."""
    output_path = tmp_path / "empty.csv"
    logger = CSVMetadataLogger(output_path=str(output_path))

    # Finalize without processing any records
    await logger.finalize_processing()

    # Should create empty CSV with header
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 0
    assert "prompt" in df.columns
