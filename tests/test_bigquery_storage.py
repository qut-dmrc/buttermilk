"""Tests for BigQuery storage implementation."""

from unittest.mock import MagicMock, Mock, patch

from google.cloud import bigquery

from buttermilk._core.storage_config import BigQueryStorageConfig
from buttermilk._core.types import Record
from buttermilk.storage.bigquery import BigQueryStorage


class TestBigQueryStorage:
    """Test BigQuery storage functionality."""

    def test_clustering_fields_validation_without_dataset_name(self):
        """Test that clustering fields are validated when dataset_name field is missing."""
        # Mock schema without dataset_name
        schema = [
            bigquery.SchemaField("record_id", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("content", "STRING"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
            # clustering_fields will auto-determine
        )

        storage = BigQueryStorage(config)
        storage._client = MagicMock()
        storage._schema_cache = schema
        storage._schema_validated = True
        storage.exists = Mock(return_value=False)

        mock_table = MagicMock()
        storage.client.create_table = Mock(return_value=mock_table)

        # Call create
        storage.create()

        # Check that clustering was set correctly (only record_id exists)
        create_call = storage.client.create_table.call_args[0][0]
        assert create_call.clustering_fields == ["record_id"]

    def test_clustering_fields_with_both_fields(self):
        """Test clustering fields when both dataset_name and record_id exist."""
        schema = [
            bigquery.SchemaField("dataset_name", "STRING"),
            bigquery.SchemaField("record_id", "STRING"),
            bigquery.SchemaField("title", "STRING"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
        )

        storage = BigQueryStorage(config)
        storage._client = MagicMock()
        storage._schema_cache = schema
        storage._schema_validated = True
        storage.exists = Mock(return_value=False)
        storage.client.create_table = Mock(return_value=MagicMock())

        storage.create()

        create_call = storage.client.create_table.call_args[0][0]
        # Order changed - now record_id first
        assert create_call.clustering_fields == ["record_id", "dataset_name"]

    def test_explicit_clustering_fields_validation(self):
        """Test that explicit clustering fields are validated against schema."""
        schema = [
            bigquery.SchemaField("dataset_name", "STRING"),
            bigquery.SchemaField("record_id", "STRING"),
            bigquery.SchemaField("title", "STRING"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
            clustering_fields=["dataset_name", "record_id", "non_existent_field"],
        )

        storage = BigQueryStorage(config)
        storage._client = MagicMock()
        storage._schema_cache = schema
        storage._schema_validated = True
        storage.exists = Mock(return_value=False)
        storage.client.create_table = Mock(return_value=MagicMock())

        with patch("buttermilk.storage.bigquery.logger") as mock_logger:
            storage.create()

            # Check warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args
            assert "Some clustering fields not found in schema" in warning_call[0][0]
            assert warning_call[1]["extra"]["invalid_fields"] == ["non_existent_field"]

        create_call = storage.client.create_table.call_args[0][0]
        assert create_call.clustering_fields == ["dataset_name", "record_id"]

    def test_no_valid_clustering_fields(self):
        """Test behavior when no valid clustering fields exist in schema."""
        schema = [
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("content", "STRING"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
        )

        storage = BigQueryStorage(config)
        storage._client = MagicMock()
        storage._schema_cache = schema
        storage._schema_validated = True
        storage.exists = Mock(return_value=False)
        storage.client.create_table = Mock(return_value=MagicMock())

        storage.create()

        create_call = storage.client.create_table.call_args[0][0]
        assert create_call.clustering_fields is None

    def test_parse_record_with_structured_types(self):
        """Test that _parse_record correctly handles structured types from BigQuery."""
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("record_id", 123),  # Integer ID to test conversion
            ("title", "Test Movie"),
            ("metadata", {"director": "Test Director", "genre": ["Action", "Drama"]}),
            ("ground_truth", {"rating": 8.5}),
            ("error", [{"code": "E001"}]),
            ("content", "Test movie content"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
        )

        storage = BigQueryStorage(config)
        storage._client = MagicMock()

        record = storage._parse_record(mock_row)

        assert record.record_id == "123"  # Converted to string
        assert isinstance(record.metadata, dict)
        assert record.metadata["director"] == "Test Director"
        assert record.metadata["genre"] == ["Action", "Drama"]
        assert isinstance(record.ground_truth, dict)
        assert record.ground_truth["rating"] == 8.5
        assert isinstance(record.error, list)
        assert record.error[0]["code"] == "E001"
        assert record.content == "Test movie content"

    def test_parse_record_with_already_parsed_fields(self):
        """Test that _parse_record handles already-parsed dict/list fields."""
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("record_id", "movie456"),
            ("metadata", {"already": "parsed"}),
            ("ground_truth", ["list", "data"]),
            ("error", []),
            ("content", "Another test"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
        )

        storage = BigQueryStorage(config)
        record = storage._parse_record(mock_row)

        assert record.record_id == "movie456"
        assert record.metadata == {"already": "parsed"}
        assert record.ground_truth == ["list", "data"]
        assert record.error == []

    def test_parse_record_with_invalid_json(self):
        """Test that _parse_record handles invalid JSON gracefully."""
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("record_id", "movie789"),
            ("metadata", "not valid json"),
            ("ground_truth", "also invalid"),
            ("error", "invalid too"),
            ("content", "Test content"),
        ]

        config = BigQueryStorageConfig(
            type="bigquery",
            dataset_name="test_dataset",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            schema_path="/tmp/test_schema.json",
        )

        storage = BigQueryStorage(config)
        record = storage._parse_record(mock_row)

        assert record.record_id == "movie789"
        assert record.metadata == {}  # Invalid JSON defaults to empty dict
        assert isinstance(record.ground_truth, str)  # Kept as string
        assert record.error == []  # Invalid JSON defaults to empty list

    def test_integer_record_id_conversion(self):
        """Test that integer record_id values are converted to strings."""
        # Test with direct Record instantiation
        record = Record(record_id=12345, content="Test content")
        assert record.record_id == "12345"
        assert isinstance(record.record_id, str)

        # Test with float (should also convert)
        record2 = Record(record_id=678.9, content="Test content")
        assert record2.record_id == "678.9"
        assert isinstance(record2.record_id, str)
