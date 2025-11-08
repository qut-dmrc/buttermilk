"""Tests for DuckDB storage configuration."""

import pytest

from buttermilk._core.storage_config import StorageFactory


class TestDuckDBStorageConfig:
    """Test DuckDB storage configuration."""

    def test_duckdb_config_creation(self):
        """DuckDBStorageConfig should be creatable with required fields."""
        from buttermilk._core.storage_config import DuckDBStorageConfig

        config = DuckDBStorageConfig(
            type="duckdb", database="test.db", table_name="test_table"
        )

        assert config.type == "duckdb"
        assert config.database == "test.db"
        assert config.table_name == "test_table"
        assert config.schema_name is None  # Default
        assert config.custom_query is None  # Default

    def test_duckdb_config_with_optional_fields(self):
        """DuckDBStorageConfig should support optional fields."""
        from buttermilk._core.storage_config import DuckDBStorageConfig

        config = DuckDBStorageConfig(
            type="duckdb",
            database="test.db",
            table_name="test_table",
            schema_name="custom_schema",
            custom_query="SELECT * FROM test WHERE id > 100",
        )

        assert config.schema_name == "custom_schema"
        assert config.custom_query == "SELECT * FROM test WHERE id > 100"

    def test_duckdb_config_has_common_base_fields(self):
        """DuckDBStorageConfig should have all common base fields."""
        from buttermilk._core.storage_config import DuckDBStorageConfig

        config = DuckDBStorageConfig(
            type="duckdb", database="test.db", table_name="test_table"
        )

        common_fields = [
            "type",
            "dataset_name",
            "randomize",
            "batch_size",
            "auto_create",
            "filter",
            "columns",
            "limit",
            "name",
            "schema_path",
            "uri",
            "db",
        ]

        for field in common_fields:
            assert hasattr(config, field), (
                f"DuckDBStorageConfig missing common field: {field}"
            )

    def test_duckdb_config_has_relevant_fields_only(self):
        """DuckDBStorageConfig should only have DuckDB-specific fields."""
        from buttermilk._core.storage_config import DuckDBStorageConfig

        config = DuckDBStorageConfig(
            type="duckdb", database="test.db", table_name="test_table"
        )

        # Should have DuckDB-specific fields
        assert hasattr(config, "database")
        assert hasattr(config, "table_name")
        assert hasattr(config, "schema_name")
        assert hasattr(config, "custom_query")

        # Should NOT have file-specific fields
        assert not hasattr(config, "path")
        assert not hasattr(config, "glob")
        assert not hasattr(config, "index")

        # Should NOT have vector-specific fields
        assert not hasattr(config, "persist_directory")
        assert not hasattr(config, "collection_name")
        assert not hasattr(config, "embedding_model")
        assert not hasattr(config, "dimensionality")
        assert not hasattr(config, "multi_field_embedding")

        # Should NOT have BigQuery-specific fields
        assert not hasattr(config, "project_id")
        assert not hasattr(config, "dataset_id")
        assert not hasattr(config, "table_id")
        assert not hasattr(config, "clustering_fields")

    def test_duckdb_type_validation(self):
        """DuckDB config should only accept 'duckdb' type."""
        from buttermilk._core.storage_config import DuckDBStorageConfig

        # Valid type
        config = DuckDBStorageConfig(
            type="duckdb", database="test.db", table_name="test_table"
        )
        assert config.type == "duckdb"

        # Invalid type should fail
        with pytest.raises(ValueError):
            DuckDBStorageConfig(
                type="invalid_type", database="test.db", table_name="test_table"
            )

    def test_storage_factory_creates_duckdb_config(self):
        """StorageFactory should handle DuckDB config dicts."""
        config_dict = {
            "type": "duckdb",
            "database": "test.db",
            "table_name": "test_table",
            "schema_name": "main",
        }

        config = StorageFactory.create_config(config_dict)
        assert config.type == "duckdb"
        assert config.database == "test.db"
        assert config.table_name == "test_table"
        assert config.schema_name == "main"

    def test_storage_factory_creates_duckdb_storage(self):
        """StorageFactory should create DuckDBStorage instances."""
        config_dict = {
            "type": "duckdb",
            "database": ":memory:",
            "table_name": "test_table",
        }

        storage = StorageFactory.create_storage(config_dict)
        assert storage is not None
        assert storage.__class__.__name__ == "DuckDBStorage"

    def test_duckdb_config_rejects_extra_fields(self):
        """DuckDBStorageConfig should reject extra fields (extra='forbid')."""
        from buttermilk._core.storage_config import DuckDBStorageConfig

        with pytest.raises(ValueError):
            DuckDBStorageConfig(
                type="duckdb",
                database="test.db",
                table_name="test_table",
                invalid_field="should_fail",
            )
