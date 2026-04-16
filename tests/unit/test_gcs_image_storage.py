"""Unit tests for GCSImageStorageProcessor."""

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.processors.gcs_image_storage import GCSImageStorageProcessor


@pytest.mark.anyio
async def test_gcs_storage_constructs_structured_path(tmp_path):
    """Test GCS path construction with session_id and scenario."""
    processor = GCSImageStorageProcessor(
        bucket="test-bucket",
        base_path="test-images",
        local_only=True,  # For testing without actual GCS upload
        local_dir=str(tmp_path),
    )

    # Create test image file
    test_image = tmp_path / "test_image.png"
    test_image.write_bytes(b"fake image data")

    record = BaseRecord(
        record_id="test-001",
        content="test prompt",
        metadata={
            "session_id": "sess123",
            "scenario": "working in office",
            "model_prefix": "imagen3fast_",
            "repetition": 0,
            "image_uri": test_image.as_uri(),
        },
    )

    results = [r async for r in processor.process(ProcessingContext(session_id="store", record=record))]

    assert len(results) == 1
    result = results[0]
    assert "storage_uri" in result.metadata
    # Path should be: test-images/working_in_office/sess123/imagen3fast_*.png
    assert "working_in_office" in result.metadata["storage_uri"]
    assert "sess123" in result.metadata["storage_uri"]


@pytest.mark.anyio
async def test_gcs_storage_sanitizes_scenario_names(tmp_path):
    """Test that scenario names are sanitized for file paths."""
    processor = GCSImageStorageProcessor(
        bucket="test-bucket",
        base_path="test-images",
        local_only=True,
        local_dir=str(tmp_path),
    )

    test_image = tmp_path / "test.png"
    test_image.write_bytes(b"fake")

    record = BaseRecord(
        record_id="test-002",
        content="prompt",
        metadata={
            "session_id": "s1",
            "scenario": "working in an office!",  # Has spaces and punctuation
            "model_prefix": "test_",
            "repetition": 0,
            "image_uri": test_image.as_uri(),
        },
    )

    results = [r async for r in processor.process(ProcessingContext(session_id="store", record=record))]

    # Scenario should be sanitized (spaces replaced, special chars removed)
    assert "working_in_an_office" in results[0].metadata["storage_uri"]


@pytest.mark.anyio
async def test_gcs_storage_preserves_original_metadata(tmp_path):
    """Test that original metadata is preserved."""
    processor = GCSImageStorageProcessor(
        bucket="test-bucket",
        base_path="test-images",
        local_only=True,
        local_dir=str(tmp_path),
    )

    test_image = tmp_path / "test.png"
    test_image.write_bytes(b"fake")

    record = BaseRecord(
        record_id="test-003",
        content="prompt",
        metadata={
            "session_id": "s1",
            "scenario": "office",
            "model_prefix": "test_",
            "repetition": 0,
            "image_uri": test_image.as_uri(),
            "custom_field": "custom_value",
        },
    )

    results = [r async for r in processor.process(ProcessingContext(session_id="store", record=record))]

    result = results[0]
    assert result.metadata["session_id"] == "s1"
    assert result.metadata["scenario"] == "office"
    assert result.metadata["custom_field"] == "custom_value"
    assert "storage_uri" in result.metadata


@pytest.mark.anyio
async def test_gcs_storage_local_file_written(tmp_path):
    """Test that in local_only mode, files are actually written."""
    local_dir = tmp_path / "output"
    local_dir.mkdir()

    processor = GCSImageStorageProcessor(
        bucket="test-bucket",
        base_path="test-images",
        local_only=True,
        local_dir=str(local_dir),
    )

    test_image = tmp_path / "source.png"
    test_image.write_bytes(b"image content")

    record = BaseRecord(
        record_id="test-004",
        content="prompt",
        metadata={
            "session_id": "s1",
            "scenario": "test",
            "model_prefix": "test_",
            "repetition": 0,
            "image_uri": test_image.as_uri(),
        },
    )

    # Process the record (triggers file write)
    _ = [r async for r in processor.process(ProcessingContext(session_id="store", record=record))]

    # Check that a file was written in the local directory
    written_files = list(local_dir.rglob("*.png"))
    assert len(written_files) > 0, "No PNG files were written to local directory"


@pytest.mark.anyio
async def test_gcs_storage_fails_fast_without_image_uri(tmp_path):
    """Test fail-fast behavior when image_uri is missing."""
    processor = GCSImageStorageProcessor(
        bucket="test-bucket",
        base_path="test-images",
        local_only=True,
        local_dir=str(tmp_path),
    )

    record = BaseRecord(
        record_id="test-005",
        content="prompt",
        metadata={
            "session_id": "s1",
            "scenario": "test",
            # Missing image_uri
        },
    )

    # Should fail fast with clear error
    with pytest.raises((ValueError, KeyError)):
        async for _ in processor.process(ProcessingContext(session_id="store", record=record)):
            pass
