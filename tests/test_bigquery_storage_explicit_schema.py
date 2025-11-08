"""Tests to ensure BigQuery storage requires explicit schemas and never uses implicit defaults."""

from unittest.mock import Mock, patch

import pytest
from google.cloud import bigquery

from buttermilk._core.exceptions import StorageError
from buttermilk._core.storage_config import StorageFactory
from buttermilk.storage.bigquery import BigQueryStorage


class TestBigQueryExplicitSchema:
    """Test that BigQuery storage requires explicit schemas with no implicit defaults."""

    def test_bigquery_storage_requires_schema_path(self):
        """BigQuery storage should fail if no schema_path is provided."""
        config = StorageFactory.create_config(
            {
                "type": "bigquery",
                "project_id": "test-project",
                "dataset_id": "test-dataset",
                "table_id": "test-table",
                "dataset_name": "test",
                # Deliberately missing schema_path
            }
        )

        with pytest.raises(StorageError) as exc_info:
            BigQueryStorage(config)

        assert "explicit schema_path" in str(exc_info.value)

    @patch("buttermilk.storage.bigquery.bigquery.Client")
    def test_bigquery_storage_fails_on_missing_schema_file(self, mock_client):
        """BigQuery storage should fail if schema file doesn't exist."""
        config = StorageFactory.create_config(
            {
                "type": "bigquery",
                "project_id": "test-project",
                "dataset_id": "test-dataset",
                "table_id": "test-table",
                "dataset_name": "test",
                "schema_path": "/non/existent/schema.json",
            }
        )

        # Mock the get_schema to simulate file not found
        with patch.object(BigQueryStorage, "get_schema") as mock_get_schema:
            mock_get_schema.side_effect = StorageError(
                "Failed to load schema from /non/existent/schema.json"
            )

            # Storage creation should succeed since validation is lazy
            storage = BigQueryStorage(config)

            # Should fail when trying to use storage (lazy validation)
            with pytest.raises(StorageError) as exc_info:
                storage.save([])

            assert "Failed to load schema" in str(exc_info.value)

    @patch("buttermilk.storage.bigquery.bigquery.Client")
    def test_create_never_uses_record_schema_fallback(self, mock_client):
        """create() should never fall back to get_record_bigquery_schema()."""
        # Mock the client and table operations
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Setup config with valid schema
        config = StorageFactory.create_config(
            {
                "type": "bigquery",
                "project_id": "test-project",
                "dataset_id": "test-dataset",
                "table_id": "test-table",
                "dataset_name": "test",
                "schema_path": "flow.json",
            }
        )

        # Mock schema loading
        with patch.object(BigQueryStorage, "get_schema") as mock_get_schema:
            mock_get_schema.return_value = None  # Simulate missing schema

            storage = BigQueryStorage(config)
            storage._client = mock_client_instance

            # Should fail when trying to create without schema
            with pytest.raises(StorageError) as exc_info:
                storage.create()

            assert "Failed to load schema" in str(
                exc_info.value
            ) or "schema file" in str(exc_info.value)

    @patch("buttermilk.storage.bigquery.bigquery.Client")
    def test_create_does_not_modify_existing_tables(self, mock_client):
        """create() should not modify schema of existing tables."""
        # Mock the client
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Mock existing table
        mock_table = Mock()
        mock_table.schema = [
            bigquery.SchemaField("existing_field", "STRING"),
        ]
        mock_client_instance.get_table.return_value = mock_table

        config = StorageFactory.create_config(
            {
                "type": "bigquery",
                "project_id": "test-project",
                "dataset_id": "test-dataset",
                "table_id": "test-table",
                "dataset_name": "test",
                "schema_path": "flow.json",
                "auto_create": True,
            }
        )

        # Mock schema with different fields
        new_schema = [
            bigquery.SchemaField("new_field", "STRING"),
        ]

        with patch.object(BigQueryStorage, "get_schema") as mock_get_schema:
            mock_get_schema.return_value = new_schema

            storage = BigQueryStorage(config)
            storage._client = mock_client_instance

            # Should not update the table
            storage.create()

            # Verify update_table was NOT called
            mock_client_instance.update_table.assert_not_called()

    def test_save_accepts_any_pydantic_model(self):
        """save() should accept any Pydantic model, not just Record."""
        from pydantic import BaseModel

        class CustomModel(BaseModel):
            id: str
            data: dict

        config = StorageFactory.create_config(
            {
                "type": "bigquery",
                "project_id": "test-project",
                "dataset_id": "test-dataset",
                "table_id": "test-table",
                "dataset_name": "test",
                "schema_path": "custom.json",
            }
        )

        with patch("buttermilk.storage.bigquery.bigquery.Client"):
            with patch.object(BigQueryStorage, "get_schema") as mock_get_schema:
                mock_get_schema.return_value = [
                    bigquery.SchemaField("id", "STRING"),
                    bigquery.SchemaField("data", "JSON"),
                ]

                BigQueryStorage(config)

                # Should not raise type errors
                CustomModel(id="test", data={"key": "value"})

                # This should work without type errors
                # (actual implementation will be updated to support this)
                # For now, this test documents the intended behavior
