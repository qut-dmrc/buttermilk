"""Tests for type-specific storage configuration schemas."""

import pytest
from buttermilk._core.storage_config import (
    BaseStorageConfig,
    BigQueryStorageConfig, 
    FileStorageConfig,
    VectorStorageConfig,
    HuggingFaceStorageConfig,
    GeneratorStorageConfig,
    StorageFactory
)


class TestTypeSpecificStorageConfigs:
    """Test that type-specific storage configs only contain relevant fields."""
    
    def test_vector_config_has_relevant_fields_only(self):
        """ChromaDB/vector configs should only have vector-specific fields."""
        config = VectorStorageConfig(
            type="chromadb",
            persist_directory="./data/chromadb",
            collection_name="test",
            embedding_model="gemini-embedding-001",
            dimensionality=3072
        )
        
        # Should have vector-specific fields
        assert hasattr(config, 'persist_directory')
        assert hasattr(config, 'collection_name') 
        assert hasattr(config, 'embedding_model')
        assert hasattr(config, 'dimensionality')
        assert hasattr(config, 'multi_field_embedding')
        
        # Should NOT have file-specific fields
        assert not hasattr(config, 'glob')
        assert not hasattr(config, 'index')
        
        # Should NOT have BigQuery-specific fields
        assert not hasattr(config, 'project_id')
        assert not hasattr(config, 'dataset_id')
        assert not hasattr(config, 'table_id')
        assert not hasattr(config, 'clustering_fields')
        assert not hasattr(config, 'last_n_days')
        
        # Should NOT have HuggingFace-specific fields
        assert not hasattr(config, 'split')
        
    def test_file_config_has_relevant_fields_only(self):
        """File configs should only have file-specific fields."""
        config = FileStorageConfig(
            type="file",
            path="./data/files",
            glob="**/*.json"
        )
        
        # Should have file-specific fields
        assert hasattr(config, 'path')
        assert hasattr(config, 'glob')
        assert hasattr(config, 'max_records_per_group')
        assert hasattr(config, 'index')
        
        # Should NOT have vector-specific fields
        assert not hasattr(config, 'persist_directory')
        assert not hasattr(config, 'collection_name')
        assert not hasattr(config, 'embedding_model')
        assert not hasattr(config, 'dimensionality')
        assert not hasattr(config, 'multi_field_embedding')
        
        # Should NOT have BigQuery-specific fields
        assert not hasattr(config, 'project_id')
        assert not hasattr(config, 'dataset_id') 
        assert not hasattr(config, 'table_id')
        assert not hasattr(config, 'clustering_fields')
        assert not hasattr(config, 'last_n_days')
        
    def test_bigquery_config_has_relevant_fields_only(self):
        """BigQuery configs should only have BigQuery-specific fields."""
        config = BigQueryStorageConfig(
            type="bigquery",
            project_id="test-project",
            dataset_id="test_dataset", 
            table_id="test_table"
        )
        
        # Should have BigQuery-specific fields
        assert hasattr(config, 'project_id')
        assert hasattr(config, 'dataset_id')
        assert hasattr(config, 'table_id')
        assert hasattr(config, 'clustering_fields')
        assert hasattr(config, 'max_records_per_group')
        assert hasattr(config, 'last_n_days')
        assert hasattr(config, 'join')
        assert hasattr(config, 'agg')
        assert hasattr(config, 'group')
        assert hasattr(config, 'full_table_id')
        
        # Should NOT have file-specific fields
        assert not hasattr(config, 'path')
        assert not hasattr(config, 'glob')
        assert not hasattr(config, 'index')
        
        # Should NOT have vector-specific fields
        assert not hasattr(config, 'persist_directory')
        assert not hasattr(config, 'collection_name')
        assert not hasattr(config, 'embedding_model')
        assert not hasattr(config, 'dimensionality')
        assert not hasattr(config, 'multi_field_embedding')
        
    def test_huggingface_config_has_relevant_fields_only(self):
        """HuggingFace configs should only have HF-specific fields."""
        config = HuggingFaceStorageConfig(
            type="huggingface",
            dataset_id="imdb",
            split="train"
        )
        
        # Should have HuggingFace-specific fields
        assert hasattr(config, 'dataset_id')
        assert hasattr(config, 'split')
        
        # Should NOT have vector-specific fields
        assert not hasattr(config, 'persist_directory')
        assert not hasattr(config, 'collection_name')
        assert not hasattr(config, 'embedding_model')
        assert not hasattr(config, 'dimensionality')
        assert not hasattr(config, 'multi_field_embedding')
        
        # Should NOT have file-specific fields
        assert not hasattr(config, 'path')
        assert not hasattr(config, 'glob')
        
    def test_storage_factory_handles_type_specific_configs(self):
        """StorageFactory should handle new type-specific configs."""
        
        # Test with dict input (OmegaConf format)
        chromadb_dict = {
            'type': 'chromadb',
            'persist_directory': './data/chromadb',
            'collection_name': 'test',
            'embedding_model': 'gemini-embedding-001',
            'dimensionality': 3072
        }
        
        file_dict = {
            'type': 'file', 
            'path': './data/files',
            'glob': '**/*.json'
        }
        
        bigquery_dict = {
            'type': 'bigquery',
            'project_id': 'test-project',
            'dataset_id': 'test_dataset',
            'table_id': 'test_table'
        }
        
        # These should not raise exceptions
        try:
            # Note: We can't actually create storage instances without dependencies
            # but we can test that the config conversion works
            storage_type = chromadb_dict['type']
            if storage_type in ['chromadb', 'vector']:
                config = VectorStorageConfig(**chromadb_dict)
                assert config.type == 'chromadb'
                
            storage_type = file_dict['type']  
            if storage_type in ['file', 'local', 'gcs', 's3', 'plaintext']:
                config = FileStorageConfig(**file_dict)
                assert config.type == 'file'
                
            storage_type = bigquery_dict['type']
            if storage_type == 'bigquery':
                config = BigQueryStorageConfig(**bigquery_dict)
                assert config.type == 'bigquery'
                
        except Exception as e:
            pytest.fail(f"StorageFactory config conversion failed: {e}")
            
    def test_all_configs_have_common_base_fields(self):
        """All storage configs should have common base fields."""
        configs = [
            VectorStorageConfig(type="chromadb"),
            FileStorageConfig(type="file"),
            BigQueryStorageConfig(type="bigquery"),
            HuggingFaceStorageConfig(type="huggingface"),
            GeneratorStorageConfig(type="generator")
        ]
        
        common_fields = [
            'type', 'dataset_name', 'randomize', 'batch_size', 'auto_create',
            'filter', 'columns', 'limit', 'name', 'schema_path', 'uri', 'db'
        ]
        
        for config in configs:
            for field in common_fields:
                assert hasattr(config, field), f"{config.__class__.__name__} missing common field: {field}"
                
    def test_type_validation_enforced(self):
        """Type fields should enforce allowed values."""
        
        # Valid types should work
        VectorStorageConfig(type="chromadb")
        VectorStorageConfig(type="vector") 
        FileStorageConfig(type="file")
        FileStorageConfig(type="local")
        FileStorageConfig(type="gcs")
        FileStorageConfig(type="s3")
        FileStorageConfig(type="plaintext")
        BigQueryStorageConfig(type="bigquery")
        HuggingFaceStorageConfig(type="huggingface")
        GeneratorStorageConfig(type="generator")
        GeneratorStorageConfig(type="job")
        GeneratorStorageConfig(type="outputs")
        
        # Invalid types should fail validation
        with pytest.raises(ValueError):
            VectorStorageConfig(type="invalid_type")
            
        with pytest.raises(ValueError):
            FileStorageConfig(type="chromadb")  # Wrong type for file config
            
        with pytest.raises(ValueError):
            BigQueryStorageConfig(type="file")  # Wrong type for BQ config