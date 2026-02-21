"""Test Storage async iterator implementation."""

import json
import tempfile
from pathlib import Path

import pytest

from buttermilk._core.storage_config import FileStorageConfig
from buttermilk.storage.file import FileStorage


@pytest.mark.anyio
async def test_storage_async_iterator_protocol():
    """Test that Storage objects implement the async iterator protocol correctly."""

    # Create test data
    test_data = [
        {
            "record_id": "1",
            "content": "First record",
            "dataset_name": "test",
            "split_type": "train",
        },
        {
            "record_id": "2",
            "content": "Second record",
            "dataset_name": "test",
            "split_type": "train",
        },
        {
            "record_id": "3",
            "content": "Third record",
            "dataset_name": "test",
            "split_type": "train",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in test_data:
            json.dump(record, f)
            f.write("\n")
        temp_path = f.name

    try:
        # Create storage with randomize=False to preserve order for testing
<<<<<<< HEAD
        config = FileStorageConfig(type="file", path=temp_path, dataset_name="test", split_type="train", randomize=False)
=======
        config = FileStorageConfig(
            type="file", path=temp_path, dataset_name="test", split_type="train", randomize=False
        )
>>>>>>> origin/stable
        storage = FileStorage(config)

        # Test 1: Has required async iterator methods
        assert hasattr(storage, "__aiter__")
        assert hasattr(storage, "__anext__")

        # Test 2: Pipeline pattern compatibility (from pipeline.py:179)
        source_iter = storage if hasattr(storage, "__anext__") else storage.__aiter__()
<<<<<<< HEAD
        assert source_iter is storage  # Should use storage directly since it has __anext__
=======
        assert (
            source_iter is storage
        )  # Should use storage directly since it has __anext__
>>>>>>> origin/stable

        # Test 3: Actual async iteration
        records = []
        async for record in storage:
            records.append(record)

        assert len(records) == 3
        assert records[0].record_id == "1"
        assert records[1].record_id == "2"
        assert records[2].record_id == "3"

        # Test 4: Multiple iterations should work
        records2 = []
        async for record in storage:
            records2.append(record)
            if len(records2) >= 2:  # Just test first two
                break

        assert len(records2) == 2
        assert records2[0].record_id == "1"
        assert records2[1].record_id == "2"

    finally:
        Path(temp_path).unlink()


@pytest.mark.anyio
async def test_storage_async_iterator_empty():
    """Test async iterator with empty storage."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Empty file
        temp_path = f.name

    try:
        config = FileStorageConfig(type="file", path=temp_path, dataset_name="test")
        storage = FileStorage(config)

        records = []
        async for record in storage:
            records.append(record)

        assert len(records) == 0

    finally:
        Path(temp_path).unlink()


@pytest.mark.anyio
async def test_manual_anext_usage():
    """Test manual usage of __anext__ method."""

    test_data = [{"record_id": "1", "content": "Test record", "dataset_name": "test"}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        json.dump(test_data[0], f)
        temp_path = f.name

    try:
        config = FileStorageConfig(type="file", path=temp_path, dataset_name="test")
        storage = FileStorage(config)

        # Get async iterator
        async_iter = storage.__aiter__()
        assert async_iter is storage

        # Get first record
        record1 = await storage.__anext__()
        assert record1.record_id == "1"

        # Should get StopAsyncIteration on next call
        with pytest.raises(StopAsyncIteration):
            await storage.__anext__()

    finally:
        Path(temp_path).unlink()
