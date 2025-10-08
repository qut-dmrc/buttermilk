"""Test FileStorage uses cloudpathlib correctly for GCS paths.

This test verifies the fix for the AsyncDataUploader false success issue
where FileStorage.save() was using Python's open() instead of cloudpathlib's
.open() method, causing files to be created locally instead of uploaded to GCS.
"""

from unittest.mock import mock_open, patch

from buttermilk.storage.file import FileStorage


class TestFileStorageCloudPath:
    """Test that FileStorage correctly uses cloudpathlib for cloud storage paths."""

    def test_file_storage_uses_cloudpath_open_for_gcs(self, real_bm):
        """Test that FileStorage.save() uses cloudpathlib's .open() method for GCS paths."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        # Create a GCS path config
        config = FileStorageConfig(
            type="file",
            path="gs://test-bucket/test-file.json",
            dataset_name="test",
            split_type="test"
        )

        storage = FileStorage(config)

        # Create test record
        test_record = Record(
            record_id="test_001",
            content="Test content for GCS",
            dataset_name="test",
            split_type="test"
        )

        # Mock the cloudpathlib AnyPath.open method
        with patch.object(storage.path, "open", mock_open()) as mock_path_open:
            with patch.object(storage.path.parent, "mkdir") as mock_mkdir:
                # Call save
                storage.save([test_record])

                # Verify cloudpathlib's .open() was called, not Python's open()
                mock_path_open.assert_called_once_with("w", encoding="utf-8")
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

                # Verify content was written through cloudpathlib
                mock_file = mock_path_open.return_value.__enter__.return_value
                assert mock_file.write.called

    def test_file_storage_uses_cloudpath_open_for_local_paths(self, real_bm):
        """Test that FileStorage.save() uses cloudpathlib's .open() method for local paths too."""
        from buttermilk._core.storage_config import FileStorageConfig
        from buttermilk._core.types import Record

        # Create a local path config
        config = FileStorageConfig(
            type="file",
            path="/tmp/test-local.json",
            dataset_name="test",
            split_type="test"
        )

        storage = FileStorage(config)

        # Create test record
        test_record = Record(
            record_id="test_002",
            content="Test content for local",
            dataset_name="test",
            split_type="test"
        )

        # Mock the cloudpathlib AnyPath.open method
        with patch.object(storage.path, "open", mock_open()) as mock_path_open:
            with patch.object(storage.path.parent, "mkdir") as mock_mkdir:
                # Call save
                storage.save([test_record])

                # Verify cloudpathlib's .open() was called
                mock_path_open.assert_called_once_with("w", encoding="utf-8")
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_file_storage_create_uses_cloudpath_open(self, real_bm):
        """Test that FileStorage.create() also uses cloudpathlib's .open() method."""
        from buttermilk._core.storage_config import FileStorageConfig

        # Create a GCS path config
        config = FileStorageConfig(
            type="file",
            path="gs://test-bucket/test-create.json",
            dataset_name="test",
            split_type="test"
        )

        storage = FileStorage(config)

        # Mock the cloudpathlib AnyPath.open and exists methods
        with patch.object(storage.path, "open", mock_open()) as mock_path_open:
            with patch.object(storage.path, "exists", return_value=False):
                with patch.object(storage.path.parent, "mkdir") as mock_mkdir:
                    # Call create
                    storage.create()

                    # Verify cloudpathlib's .open() was called
                    mock_path_open.assert_called_once_with("w", encoding="utf-8")
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
