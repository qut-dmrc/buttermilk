"""Unit tests for configuration models and validation."""

import pytest
from pydantic import ValidationError
from pathlib import Path

pytestmark = pytest.mark.anyio


class TestStorageConfigValidation:
    """Test StorageConfig validation and computed fields."""

    def test_bigquery_storage_config_valid(self):
        """Test valid BigQuery StorageConfig creation."""
        from buttermilk._core.storage_config import StorageConfig
        
        config = StorageConfig(
            type="bigquery",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table"
        )
        
        assert config.type == "bigquery"
        assert config.project_id == "test-project"
        assert config.dataset_id == "test_dataset"
        assert config.table_id == "test_table"
        assert config.full_table_id == "test-project.test_dataset.test_table"

    def test_file_storage_config_valid(self):
        """Test valid file StorageConfig creation."""
        from buttermilk._core.storage_config import StorageConfig
        
        config = StorageConfig(
            type="file",
            path="/path/to/data.json",
            glob="*.json"
        )
        
        assert config.type == "file"
        assert config.path == "/path/to/data.json"
        assert config.glob == "*.json"

    def test_storage_config_full_table_id_computation(self):
        """Test full_table_id computed field logic."""
        from buttermilk._core.storage_config import StorageConfig
        
        # Complete config - should compute full table ID
        complete_config = StorageConfig(
            type="bigquery",
            project_id="proj",
            dataset_id="dataset", 
            table_id="table"
        )
        assert complete_config.full_table_id == "proj.dataset.table"
        
        # Incomplete config - should return None
        incomplete_config = StorageConfig(
            type="bigquery",
            project_id="proj"
            # Missing dataset_id and table_id
        )
        assert incomplete_config.full_table_id is None

    def test_storage_config_with_columns_mapping(self):
        """Test StorageConfig with column mapping."""
        from buttermilk._core.storage_config import StorageConfig
        
        columns = {
            "content": "text_field",
            "metadata": "meta_field", 
            "record_id": "id_field"
        }
        
        config = StorageConfig(
            type="bigquery",
            columns=columns
        )
        
        assert config.columns == columns
        assert config.columns["content"] == "text_field"

    def test_storage_config_defaults(self):
        """Test StorageConfig default values."""
        from buttermilk._core.storage_config import StorageConfig
        
        config = StorageConfig(type="bigquery")
        
        # Test default values
        assert config.randomize is True
        assert config.batch_size == 1000
        assert config.auto_create is True
        assert config.clustering_fields == ["record_id", "dataset_name"]

    def test_storage_config_merge_defaults(self):
        """Test merging StorageConfig with defaults."""
        from buttermilk._core.storage_config import StorageConfig
        
        defaults = StorageConfig(
            type="bigquery",
            project_id="default-project",
            batch_size=500,
            randomize=False
        )
        
        config = StorageConfig(
            type="bigquery",
            dataset_id="specific-dataset",
            batch_size=1000  # Should override default
        )
        
        merged = config.merge_defaults(defaults)
        
        assert merged.type == "bigquery"
        assert merged.dataset_id == "specific-dataset"
        assert merged.batch_size == 1000  # Config value overrides default
        assert merged.project_id == "default-project"  # From defaults

    def test_storage_config_serialization(self):
        """Test StorageConfig serialization."""
        from buttermilk._core.storage_config import StorageConfig
        
        config = StorageConfig(
            type="bigquery",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            columns={"content": "text"}
        )
        
        # Test model_dump
        dumped = config.model_dump()
        assert dumped["type"] == "bigquery"
        assert dumped["project_id"] == "test-project"
        
        # Exclude computed fields for reconstruction
        serializable_data = {k: v for k, v in dumped.items() if k != "full_table_id"}
        new_config = StorageConfig(**serializable_data)
        
        assert new_config.type == config.type
        assert new_config.project_id == config.project_id
        assert new_config.columns == config.columns


class TestBigQueryDefaults:
    """Test BigQueryDefaults configuration after Phase 1 optimizations."""

    def test_bigquery_defaults_no_hardcoded_values(self):
        """Test that BigQueryDefaults has no hardcoded values."""
        from buttermilk._core.storage_config import BigQueryDefaults
        
        defaults = BigQueryDefaults()
        
        # After Phase 1 optimizations, these should be None
        assert defaults.dataset_id is None
        assert defaults.table_id is None
        assert defaults.randomize is True  # Non-None defaults are OK
        assert defaults.batch_size == 1000
        assert defaults.auto_create is True

    def test_bigquery_defaults_with_explicit_values(self):
        """Test BigQueryDefaults with explicit values."""
        from buttermilk._core.storage_config import BigQueryDefaults
        
        defaults = BigQueryDefaults(
            dataset_id="explicit_dataset",
            table_id="explicit_table",
            batch_size=2000
        )
        
        assert defaults.dataset_id == "explicit_dataset"
        assert defaults.table_id == "explicit_table"
        assert defaults.batch_size == 2000


class TestDataSourceConfigValidation:
    """Test DataSourceConfig validation and conversion."""

    def test_data_source_config_creation(self):
        """Test DataSourceConfig creation with various source types."""
        from buttermilk._core.config import DataSourceConfig
        
        # File source
        file_config = DataSourceConfig(
            type="file",
            path="/data/file.json",
            columns={"content": "text"}
        )
        assert file_config.type == "file"
        assert file_config.path == "/data/file.json"
        
        # BigQuery source  
        bq_config = DataSourceConfig(
            type="bigquery",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table"
        )
        assert bq_config.type == "bigquery"
        assert bq_config.project_id == "test-project"

    def test_data_source_config_column_mapping(self):
        """Test DataSourceConfig column mapping functionality."""
        from buttermilk._core.config import DataSourceConfig
        
        columns = {
            "content": "source_text",
            "metadata": "source_meta",
            "record_id": "source_id"
        }
        
        config = DataSourceConfig(
            type="file",
            path="/data/test.json",
            columns=columns
        )
        
        assert config.columns == columns

    def test_data_source_config_with_aggregation(self):
        """Test DataSourceConfig with aggregation settings."""
        from buttermilk._core.config import DataSourceConfig
        
        config = DataSourceConfig(
            type="bigquery",
            project_id="test-project",
            dataset_id="test_dataset", 
            table_id="test_table",
            agg="sum",
            group=["category", "date"]
        )
        
        assert config.agg == "sum"
        assert config.group == ["category", "date"]


class TestCloudProviderConfigValidation:
    """Test CloudProviderCfg validation."""

    def test_gcp_provider_config(self):
        """Test GCP CloudProviderCfg validation."""
        from buttermilk._core.config import CloudProviderCfg
        
        config = CloudProviderCfg(
            type="gcp",
            project="test-project-123",
            quota_project_id="quota-project-456"
        )
        
        assert config.type == "gcp"
        assert config.project == "test-project-123"
        assert config.quota_project_id == "quota-project-456"

    def test_azure_provider_config(self):
        """Test Azure CloudProviderCfg validation."""
        from buttermilk._core.config import CloudProviderCfg
        
        config = CloudProviderCfg(
            type="azure",
            vault="https://test-vault.vault.azure.net/"
        )
        
        assert config.type == "azure"
        assert config.vault == "https://test-vault.vault.azure.net/"

    def test_vertex_provider_config(self):
        """Test Vertex AI CloudProviderCfg validation.""" 
        from buttermilk._core.config import CloudProviderCfg
        
        config = CloudProviderCfg(
            type="vertex",
            project="test-project",
            region="us-central1",
            location="us-central1"
        )
        
        assert config.type == "vertex"
        assert config.project == "test-project"
        assert config.region == "us-central1"
        assert config.location == "us-central1"


class TestRecordTypeValidation:
    """Test Record type validation and serialization."""

    def test_record_creation_with_string_content(self):
        """Test Record creation with string content."""
        from buttermilk._core.types import Record
        
        record = Record(
            content="Test content",
            mime="text/plain"
        )
        
        assert record.content == "Test content"
        assert record.mime == "text/plain"
        assert record.record_id is not None

    def test_record_creation_with_custom_id(self):
        """Test Record creation with custom record_id."""
        from buttermilk._core.types import Record
        
        custom_id = "custom_record_123"
        record = Record(
            content="Test content",
            mime="text/plain",
            record_id=custom_id
        )
        
        assert record.record_id == custom_id

    def test_record_with_metadata(self):
        """Test Record with metadata."""
        from buttermilk._core.types import Record
        
        metadata = {
            "source": "test_source",
            "category": "unit_test",
            "tags": ["test", "validation"]
        }
        
        record = Record(
            content="Test content with metadata",
            mime="text/plain",
            metadata=metadata
        )
        
        assert record.metadata == metadata
        assert record.metadata["source"] == "test_source"

    def test_record_serialization_round_trip(self):
        """Test Record serialization and deserialization."""
        from buttermilk._core.types import Record
        
        original = Record(
            content="Serialization test",
            mime="text/plain",
            metadata={"test": True},
            record_id="test_123"
        )
        
        # Serialize
        dumped = original.model_dump()
        
        # Deserialize
        restored = Record(**dumped)
        
        assert restored.content == original.content
        assert restored.mime == original.mime
        assert restored.metadata == original.metadata
        assert restored.record_id == original.record_id

    def test_record_equality(self):
        """Test Record equality comparison."""
        from buttermilk._core.types import Record
        
        record1 = Record(
            content="Same content",
            mime="text/plain",
            record_id="same_id"
        )
        
        record2 = Record(
            content="Same content", 
            mime="text/plain",
            record_id="same_id"
        )
        
        record3 = Record(
            content="Different content",
            mime="text/plain", 
            record_id="same_id"
        )
        
        assert record1 == record2
        assert record1 != record3


class TestValidationErrorHandling:
    """Test configuration validation error handling."""

    def test_invalid_storage_type(self):
        """Test error handling for invalid storage type."""
        from buttermilk._core.storage_config import StorageConfig
        
        with pytest.raises(ValidationError):
            StorageConfig(type="invalid_type")

    def test_missing_required_bigquery_fields(self):
        """Test validation of required BigQuery fields when needed."""
        from buttermilk._core.storage_config import StorageConfig
        
        # This should work - minimal valid config
        config = StorageConfig(type="bigquery")
        assert config.type == "bigquery"
        
        # But full_table_id should be None without all fields
        assert config.full_table_id is None

    def test_invalid_cloud_provider_type(self):
        """Test error handling for invalid cloud provider type."""
        from buttermilk._core.config import CloudProviderCfg
        
        with pytest.raises(ValidationError):
            CloudProviderCfg(type="invalid_provider")


class TestConfigurationCaching:
    """Test configuration caching and performance."""

    def test_config_model_creation_performance(self):
        """Test that config model creation is reasonably fast."""
        import time
        from buttermilk._core.storage_config import StorageConfig
        
        start_time = time.time()
        
        # Create many config instances
        configs = []
        for i in range(100):
            config = StorageConfig(
                type="bigquery",
                project_id=f"project_{i}",
                dataset_id=f"dataset_{i}",
                table_id=f"table_{i}"
            )
            configs.append(config)
        
        creation_time = time.time() - start_time
        
        # Should be able to create 100 configs quickly
        assert creation_time < 0.1, f"Config creation took {creation_time:.3f}s for 100 instances"
        assert len(configs) == 100

    def test_config_validation_caching(self):
        """Test that Pydantic validation benefits from caching."""
        from buttermilk._core.storage_config import StorageConfig
        
        # Create identical configs - Pydantic should optimize this
        config_data = {
            "type": "bigquery",
            "project_id": "test-project",
            "dataset_id": "test_dataset",
            "table_id": "test_table"
        }
        
        configs = [StorageConfig(**config_data) for _ in range(10)]
        
        # All should be valid and identical
        for config in configs:
            assert config.type == "bigquery"
            assert config.full_table_id == "test-project.test_dataset.test_table"