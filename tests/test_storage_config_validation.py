"""Test storage configuration validation in orchestrators."""

import pytest
from typing import Any
from unittest.mock import Mock, patch

from buttermilk._core.orchestrator import OrchestratorProtocol, Orchestrator
from buttermilk._core.storage_config import (
    BaseStorageConfig,
    BigQueryStorageConfig,
    FileStorageConfig,
    VectorStorageConfig,
    GeneratorStorageConfig,
    StorageFactory,
)
from buttermilk._core.config import DataSourceConfig


class TestStorageConfigValidation:
    """Test storage config validation in orchestrators."""
    
    def test_file_storage_config_validation(self):
        """Test that file storage configs are properly validated."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "tja": {
                    "type": "file",
                    "path": "/data/test.jsonl",
                    "index": ["test_index"]  # index should be a list
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Verify storage was converted to FileStorageConfig
        assert "tja" in orchestrator_config.storage
        tja_config = orchestrator_config.storage["tja"]
        assert isinstance(tja_config, FileStorageConfig)
        assert tja_config.path == "/data/test.jsonl"
        assert tja_config.index == ["test_index"]
    
    def test_bigquery_storage_config_validation(self):
        """Test that BigQuery storage configs are properly validated."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "tox_train": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "table_id": "test_table",
                    "split_type": "random",
                    "auto_create": True
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Verify storage was converted to BigQueryStorageConfig
        assert "tox_train" in orchestrator_config.storage
        tox_config = orchestrator_config.storage["tox_train"]
        assert isinstance(tox_config, BigQueryStorageConfig)
        assert tox_config.project_id == "test-project"
        assert tox_config.dataset_id == "test_dataset"
        assert tox_config.table_id == "test_table"
    
    def test_vector_storage_config_validation(self):
        """Test that vector storage configs are properly validated."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow", 
            "storage": {
                "osb_vector": {
                    "type": "chromadb",  # Use chromadb instead of vector
                    "persist_directory": "/data/embeddings",
                    "collection_name": "osb_documents"
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Verify storage was converted to VectorStorageConfig
        assert "osb_vector" in orchestrator_config.storage
        vector_config = orchestrator_config.storage["osb_vector"]
        assert isinstance(vector_config, VectorStorageConfig)
        assert vector_config.persist_directory == "/data/embeddings"
        assert vector_config.collection_name == "osb_documents"
    
    def test_legacy_datasource_config_compatibility(self):
        """Test backward compatibility with DataSourceConfig."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "legacy": {
                    "type": "outputs",  # Use a type that DataSourceConfig supports but StorageConfig might not have specific handling for
                    "path": "/data/legacy.json"
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Verify storage was converted
        assert "legacy" in orchestrator_config.storage
        legacy_config = orchestrator_config.storage["legacy"]
        # It will be converted to GeneratorStorageConfig for type="outputs"
        assert isinstance(legacy_config, (GeneratorStorageConfig, DataSourceConfig))
    
    def test_mixed_storage_configs(self):
        """Test flow with mixed storage config types."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "file_data": {
                    "type": "file",
                    "path": "/data/input.json"
                },
                "bigquery_data": {
                    "type": "bigquery",
                    "project_id": "my-project",
                    "dataset_id": "my_dataset"
                },
                "plain_data": {
                    "type": "plaintext",
                    "path": "/data/text"
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Verify each storage config has correct type
        assert isinstance(orchestrator_config.storage["file_data"], FileStorageConfig)
        assert isinstance(orchestrator_config.storage["bigquery_data"], BigQueryStorageConfig)
        assert isinstance(orchestrator_config.storage["plain_data"], FileStorageConfig)  # plaintext uses FileStorageConfig
    
    def test_invalid_storage_type_fallback(self):
        """Test that configs with storage type bq fall back to DataSourceConfig when not supported by StorageFactory."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "bq_legacy": {
                    # Use 'bq' which DataSourceConfig supports but might not be in StorageFactory
                    "type": "bq",
                    "path": "/data/something"
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Verify it created appropriate config (should be BigQueryStorageConfig if supported)
        assert "bq_legacy" in orchestrator_config.storage
        bq_config = orchestrator_config.storage["bq_legacy"]
        # It should be converted to DataSourceConfig or BigQueryStorageConfig
        assert isinstance(bq_config, (DataSourceConfig, BigQueryStorageConfig))
    
    def test_storage_factory_direct(self):
        """Test StorageFactory.create_config directly."""
        # Test file storage
        file_config = StorageFactory.create_config({
            "type": "file",
            "path": "/test/path.json",
            "index": ["test"]  # index should be a list
        })
        assert isinstance(file_config, FileStorageConfig)
        assert file_config.path == "/test/path.json"
        
        # Test bigquery storage
        bq_config = StorageFactory.create_config({
            "type": "bigquery",
            "project_id": "test-project",
            "dataset_id": "test_dataset"
        })
        assert isinstance(bq_config, BigQueryStorageConfig)
        assert bq_config.project_id == "test-project"
        
        # Test missing type
        with pytest.raises(ValueError, match="Missing 'type' field"):
            StorageFactory.create_config({"path": "/test"})
    
    def test_storage_specific_fields_detection(self):
        """Test detection of storage-specific fields."""
        config_dict = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "auto_detect": {
                    # Has storage-specific field, should use StorageFactory
                    "auto_create": True,
                    "path": "/data/test.json",
                    "type": "file"  # Need to provide type for StorageFactory
                }
            }
        }
        
        # Create orchestrator protocol with config
        orchestrator_config = OrchestratorProtocol(**config_dict)
        
        # Since it has auto_create and type, it should use StorageFactory
        assert "auto_detect" in orchestrator_config.storage
        config = orchestrator_config.storage["auto_detect"]
        assert isinstance(config, FileStorageConfig)
        assert config.auto_create == True


class ConcreteOrchestrator(Orchestrator):
    """Concrete implementation for testing."""
    
    async def _setup(self) -> None:
        """Setup implementation."""
        pass
    
    async def _cleanup(self) -> None:
        """Cleanup implementation."""
        pass
    
    async def _run(self, run_request: Any) -> Any:
        """Run implementation."""
        return {}


class TestOrchestratorIntegration:
    """Test orchestrator integration with storage configs."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_with_storage_configs(self):
        """Test full orchestrator with storage configs."""
        config = {
            "orchestrator": "test",
            "name": "test_flow",
            "storage": {
                "input_file": {
                    "type": "file",
                    "path": "/data/input.jsonl"
                },
                "results_bq": {
                    "type": "bigquery",
                    "project_id": "my-project",
                    "dataset_id": "results"
                }
            },
            "agents": {},
            "observers": {},
            "parameters": {}
        }
        
        # Create orchestrator
        orchestrator = ConcreteOrchestrator(**config)
        
        # Verify storage configs
        assert isinstance(orchestrator.storage["input_file"], FileStorageConfig)
        assert isinstance(orchestrator.storage["results_bq"], BigQueryStorageConfig)
        
        # Verify attributes are accessible
        assert orchestrator.storage["input_file"].path == "/data/input.jsonl"
        assert orchestrator.storage["results_bq"].project_id == "my-project"