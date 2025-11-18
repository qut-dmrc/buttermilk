"""Tests for BigQuery storage read-only mode initialization.

Tests that BigQueryStorage correctly handles read_only=True initialization
without requiring schema_path, while still requiring schema_path for write mode.
"""

import pytest

from buttermilk._core.exceptions import StorageError
from buttermilk._core.storage_config import BigQueryStorageConfig
from buttermilk.storage.bigquery import BigQueryStorage


class TestBigQueryStorageReadOnly:
    """Test BigQueryStorage read-only mode initialization."""

    def test_readonly_without_schema_path_succeeds(self):
        """Test that read_only=True allows initialization without schema_path.

        ARRANGE: Create config with read_only=True and schema_path=None
        ACT: Instantiate BigQueryStorage
        ASSERT: Storage initializes successfully with read_only=True
        """
        # Arrange
        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset_id",
            table_id="test_table",
            schema_path=None,
            read_only=True,
        )

        # Act
        storage = BigQueryStorage(config)

        # Assert
        assert storage.config.read_only is True
        assert storage.config.schema_path is None

    def test_writemode_without_schema_path_raises_error(self):
        """Test that read_only=False with schema_path=None raises StorageError.

        ARRANGE: Create config with read_only=False and schema_path=None
        ACT: Attempt to instantiate BigQueryStorage
        ASSERT: Raises StorageError mentioning schema_path requirement and read_only option
        """
        # Arrange
        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset_id",
            table_id="test_table",
            schema_path=None,
            read_only=False,
        )

        # Act & Assert
        # Implementation should suggest read_only=True as alternative
        with pytest.raises(
            StorageError,
            match=r".*(schema_path|read_only).*",
        ):
            BigQueryStorage(config)

    def test_validate_schema_skips_in_readonly(self):
        """Test that _validate_schema() skips validation in read-only mode.

        ARRANGE: Create BigQueryStorage with read_only=True and schema_path=None
        ACT: Call _validate_schema() directly
        ASSERT: Method returns without error (validation skipped)
        """
        # Arrange
        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset_id",
            table_id="test_table",
            schema_path=None,
            read_only=True,
        )
        storage = BigQueryStorage(config)

        # Act - should not raise any errors
        storage._validate_schema()

        # Assert - if we get here, validation was skipped successfully
        assert storage.config.read_only is True

    def test_validate_schema_raises_in_writemode_without_schema_path(self):
        """Test that _validate_schema() raises error in write mode without schema_path.

        ARRANGE: Create BigQueryStorage with read_only=False and schema_path=None
        ACT: This should fail during initialization, but if we could call _validate_schema()
        ASSERT: Error message mentions "write operations" and "schema_path"

        Note: Since BigQueryStorage.__init__ already validates, this test documents
        the expected _validate_schema() behavior for write mode.
        """
        # This test documents expected behavior - initialization should already fail
        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset_id",
            table_id="test_table",
            schema_path=None,
            read_only=False,
        )

        # Act & Assert - should fail with specific error about write operations
        with pytest.raises(
            StorageError,
            match=r".*(write operations|schema_path).*",
        ):
            storage = BigQueryStorage(config)
            storage._validate_schema()

    def test_save_fails_in_readonly(self):
        """Test that save() raises StorageError with clear message in read-only mode.

        ARRANGE: Create BigQueryStorage with read_only=True and a simple Record
        ACT: Attempt to call storage.save([record])
        ASSERT: Raises StorageError mentioning "read-only mode" and "cannot save"
        """
        # Arrange
        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset_id",
            table_id="test_table",
            schema_path=None,
            read_only=True,
        )
        storage = BigQueryStorage(config)

        # Create a simple test record
        from buttermilk._core.types import BaseRecord

        test_record = BaseRecord(
            record_id="test_001",
            dataset_name="test_dataset",
            content="Test content for read-only validation",
            metadata={"test_key": "test_value"},
        )

        # Act & Assert
        with pytest.raises(
            StorageError,
            match=r".*read-only mode.*cannot save.*",
        ):
            storage.save([test_record])
