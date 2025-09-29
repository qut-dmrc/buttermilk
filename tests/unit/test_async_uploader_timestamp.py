"""Test AsyncDataUploader timestamp suffix functionality.

This test verifies that AsyncDataUploader correctly creates timestamped
files when configured to do so, preventing accidental overwrites.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from buttermilk.utils.uploader import AsyncDataUploader
from buttermilk.storage.file import FileStorage
from buttermilk._core.storage_config import FileStorageConfig
from buttermilk._core.types import Record


class TestAsyncDataUploaderTimestamp:
    """Test AsyncDataUploader timestamp suffix functionality."""

    @pytest.mark.anyio
    async def test_timestamp_suffix_when_file_exists(self, real_bm):
        """Test that timestamp suffixes are used by default when file exists."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create an existing file first
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test"
            )

            storage = FileStorage(config)

            # Create initial record
            initial_record = Record(
                record_id="initial_001",
                content="Initial record",
                dataset_name="test",
                split_type="test"
            )
            storage.save([initial_record])

            # Verify file exists
            assert storage.exists()

            # Create uploader (should default to use_timestamp_suffix=True since file exists)
            uploader = AsyncDataUploader(
                storage=storage,
                buffer_size=1,  # Small buffer for immediate flush
                flush_interval=1  # Short interval
            )

            # Verify timestamp suffix is enabled by default
            assert uploader.use_timestamp_suffix is True

            # Add a record
            test_record = Record(
                record_id="test_001",
                content="Test record via uploader",
                dataset_name="test",
                split_type="test"
            )

            await uploader.add(test_record)

            # Wait for flush
            await asyncio.sleep(2)

            # Check that original file still contains only initial record
            original_records = list(storage)
            assert len(original_records) == 1
            assert original_records[0].content == "Initial record"

            # Check that a timestamped file was created
            parent_dir = Path(tmp_path).parent
            base_name = Path(tmp_path).stem
            timestamp_files = list(parent_dir.glob(f"{base_name}-*.json"))
            assert len(timestamp_files) >= 1

            # Verify timestamped file contains the new record
            timestamped_file = timestamp_files[0]
            timestamped_config = FileStorageConfig(
                type="file",
                path=str(timestamped_file),
                dataset_name="test",
                split_type="test"
            )
            timestamped_storage = FileStorage(timestamped_config)
            timestamped_records = list(timestamped_storage)
            assert len(timestamped_records) == 1
            assert timestamped_records[0].content == "Test record via uploader"

            # Cleanup uploader
            uploader.shutdown()

        finally:
            # Cleanup all files
            Path(tmp_path).unlink(missing_ok=True)
            parent_dir = Path(tmp_path).parent
            base_name = Path(tmp_path).stem
            for f in parent_dir.glob(f"{base_name}-*.json"):
                f.unlink(missing_ok=True)

    @pytest.mark.anyio
    async def test_no_timestamp_suffix_when_file_not_exists(self, real_bm):
        """Test that timestamp suffixes are NOT used by default when file doesn't exist."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            tmp_path = tmp.name
            # File is automatically deleted

        try:
            # Create storage config for non-existent file
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test"
            )

            storage = FileStorage(config)

            # Verify file doesn't exist
            assert not storage.exists()

            # Create uploader (should default to use_timestamp_suffix=False since file doesn't exist)
            uploader = AsyncDataUploader(
                storage=storage,
                buffer_size=1,  # Small buffer for immediate flush
                flush_interval=1  # Short interval
            )

            # Verify timestamp suffix is disabled by default
            assert uploader.use_timestamp_suffix is False

            # Add a record
            test_record = Record(
                record_id="test_001",
                content="Test record via uploader",
                dataset_name="test",
                split_type="test"
            )

            await uploader.add(test_record)

            # Wait for flush
            await asyncio.sleep(2)

            # Check that the original file was created with the record
            assert storage.exists()
            records = list(storage)
            assert len(records) == 1
            assert records[0].content == "Test record via uploader"

            # Check that no timestamped files were created
            parent_dir = Path(tmp_path).parent
            base_name = Path(tmp_path).stem
            timestamp_files = list(parent_dir.glob(f"{base_name}-*.json"))
            assert len(timestamp_files) == 0

            # Cleanup uploader
            uploader.shutdown()

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.anyio
    async def test_explicit_timestamp_suffix_override(self, real_bm):
        """Test explicit timestamp suffix parameter overrides default behavior."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create an existing file first
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test"
            )

            storage = FileStorage(config)

            # Create initial record
            initial_record = Record(
                record_id="initial_001",
                content="Initial record",
                dataset_name="test",
                split_type="test"
            )
            storage.save([initial_record])

            # Create uploader with explicit use_timestamp_suffix=False
            uploader = AsyncDataUploader(
                storage=storage,
                buffer_size=1,
                flush_interval=1,
                use_timestamp_suffix=False  # Explicit override
            )

            # Verify timestamp suffix is disabled despite file existing
            assert uploader.use_timestamp_suffix is False

            # Add a record
            test_record = Record(
                record_id="test_001",
                content="Test record via uploader",
                dataset_name="test",
                split_type="test"
            )

            await uploader.add(test_record)

            # Wait for flush
            await asyncio.sleep(2)

            # Check that original file was overwritten
            records = list(storage)
            assert len(records) == 1
            assert records[0].content == "Test record via uploader"

            # Check that no timestamped files were created
            parent_dir = Path(tmp_path).parent
            base_name = Path(tmp_path).stem
            timestamp_files = list(parent_dir.glob(f"{base_name}-*.json"))
            assert len(timestamp_files) == 0

            # Cleanup uploader
            uploader.shutdown()

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.anyio
    async def test_timestamp_format(self, real_bm):
        """Test that timestamp format is correct and predictable."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create storage
            config = FileStorageConfig(
                type="file",
                path=tmp_path,
                dataset_name="test",
                split_type="test"
            )

            storage = FileStorage(config)

            # Create initial file
            storage.save([Record(record_id="init", content="init", dataset_name="test", split_type="test")])

            # Create uploader
            uploader = AsyncDataUploader(
                storage=storage,
                buffer_size=1,
                flush_interval=1,
                use_timestamp_suffix=True
            )

            # Mock datetime to get predictable timestamp
            with patch('buttermilk.utils.uploader.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250929-123456"

                # Test the _create_timestamped_storage method directly
                timestamped_storage = uploader._create_timestamped_storage()

                # Verify the path format
                expected_path = tmp_path.replace('.json', '-20250929-123456.json')
                assert str(timestamped_storage.path) == expected_path

            # Cleanup uploader
            uploader.shutdown()

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)