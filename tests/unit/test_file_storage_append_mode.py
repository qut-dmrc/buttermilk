"""Test FileStorage append mode functionality.

This test verifies that FileStorage correctly appends to existing files
when the append parameter is set to True, preventing data loss.
"""

import json
import tempfile
from pathlib import Path

import pytest

from buttermilk.storage.file import FileStorage
from buttermilk._core.storage_config import FileStorageConfig
from buttermilk._core.types import Record


class TestFileStorageAppendMode:
    """Test FileStorage append mode functionality."""

    def test_append_mode_jsonl_format(self, real_bm):
        """Test append mode with JSONL format files."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create config with append=True
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test",
                append=True
            )

            storage = FileStorage(config)

            # Create first batch of records
            records1 = [
                Record(
                    record_id="test_001",
                    content="First record",
                    dataset_name="test",
                    split_type="test"
                ),
                Record(
                    record_id="test_002",
                    content="Second record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save first batch
            storage.save(records1)

            # Verify first batch was saved
            assert storage.exists()
            saved_records = list(storage)
            assert len(saved_records) == 2
            assert saved_records[0].content == "First record"
            assert saved_records[1].content == "Second record"

            # Create second batch of records
            records2 = [
                Record(
                    record_id="test_003",
                    content="Third record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save second batch (should append)
            storage.save(records2)

            # Verify all records are present (no overwrite)
            all_records = list(storage)
            assert len(all_records) == 3
            assert all_records[0].content == "First record"
            assert all_records[1].content == "Second record"
            assert all_records[2].content == "Third record"

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_append_mode_json_format(self, real_bm):
        """Test append mode with JSON format files."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create config with append=True
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test",
                append=True
            )

            storage = FileStorage(config)

            # Create first batch of records
            records1 = [
                Record(
                    record_id="test_001",
                    content="First record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save first batch
            storage.save(records1)

            # Create second batch of records
            records2 = [
                Record(
                    record_id="test_002",
                    content="Second record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save second batch (should append by merging)
            storage.save(records2)

            # Verify all records are present
            all_records = list(storage)
            assert len(all_records) == 2
            assert all_records[0].content == "First record"
            assert all_records[1].content == "Second record"

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_overwrite_mode_default(self, real_bm):
        """Test that default behavior (append=False) still overwrites."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create config with default append=False
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test"
                # append defaults to False
            )

            storage = FileStorage(config)

            # Create first batch of records
            records1 = [
                Record(
                    record_id="test_001",
                    content="First record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save first batch
            storage.save(records1)

            # Create second batch of records
            records2 = [
                Record(
                    record_id="test_002",
                    content="Second record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save second batch (should overwrite)
            storage.save(records2)

            # Verify only second batch records are present (overwritten)
            all_records = list(storage)
            assert len(all_records) == 1
            assert all_records[0].content == "Second record"

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_append_mode_nonexistent_file(self, real_bm):
        """Test append mode when file doesn't exist initially."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            tmp_path = tmp.name
            # File is automatically deleted, so it doesn't exist

        try:
            # Create config with append=True
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test",
                append=True
            )

            storage = FileStorage(config)

            # Create records
            records = [
                Record(
                    record_id="test_001",
                    content="First record",
                    dataset_name="test",
                    split_type="test"
                )
            ]

            # Save to non-existent file (should create new file)
            storage.save(records)

            # Verify record was saved
            saved_records = list(storage)
            assert len(saved_records) == 1
            assert saved_records[0].content == "First record"

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)