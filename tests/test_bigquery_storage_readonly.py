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
