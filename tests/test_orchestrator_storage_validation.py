"""Test storage configuration validation in Orchestrator with Hydra."""

import pytest
from omegaconf import DictConfig, OmegaConf
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.storage_config import StorageConfig
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.types import RunRequest
from pydantic import ValidationError


class TestOrchestrator(Orchestrator):
    """Test implementation of Orchestrator."""
    
    async def _setup(self, request: RunRequest) -> None:
        pass
    
    async def _cleanup(self) -> None:
        pass
    
    async def _run(self, request: RunRequest) -> None:
        await self._setup(request)


class TestOrchestratorStorageValidation:
    """Test that Orchestrator properly handles storage configurations from Hydra."""
    
    def test_orchestrator_with_datasourceconfig_type_hint(self):
        """Test current behavior with DataSourceConfig type hint."""
        # Current type hint is Mapping[str, DataSourceConfig]
        # So Hydra will try to create DataSourceConfig objects
        config = {
            "test_storage": DataSourceConfig(
                type="file",
                path="/test/path.json"
            )
        }
        
        orchestrator = TestOrchestrator(
            orchestrator="test",
            name="test_orchestrator",
            storage=config
        )
        
        assert "test_storage" in orchestrator.storage
        assert isinstance(orchestrator.storage["test_storage"], DataSourceConfig)
    
    def test_issue_with_storageconfig_fields(self):
        """Test the actual issue - StorageConfig has extra='forbid' but DataSourceConfig doesn't have all fields."""
        # The issue is that if we try to create a StorageConfig with the current system,
        # it will fail because StorageConfig has fields that DataSourceConfig doesn't
        
        # This simulates what happens when YAML has StorageConfig-specific fields
        # but the type hint expects DataSourceConfig
        yaml_config = """
        storage:
          test_storage:
            type: bigquery
            project_id: test-project
            dataset_id: test_dataset
            table_id: test_table
            auto_create: true
            clustering_fields: ["record_id", "dataset_name"]
        """
        
        cfg = OmegaConf.create(yaml_config)
        
        # If we try to create DataSourceConfig from this, it should work because
        # DataSourceConfig has extra="ignore"
        storage_dict = OmegaConf.to_container(cfg.storage)
        test_storage = DataSourceConfig(**storage_dict["test_storage"])
        assert test_storage.type == "bigquery"
        # But auto_create and clustering_fields are ignored!
        assert not hasattr(test_storage, "auto_create")
        assert not hasattr(test_storage, "clustering_fields")
    
    def test_storageconfig_rejects_unknown_fields(self):
        """Test that StorageConfig with extra='forbid' rejects unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            StorageConfig(
                type="bigquery",
                project_id="test-project",
                unknown_field="should_fail"  # This should be rejected
            )
        
        assert "unknown_field" in str(exc_info.value)
    
    def test_desired_behavior_union_type(self):
        """Test what we want - support for both DataSourceConfig and StorageConfig."""
        # The fix should allow the orchestrator to accept either type
        # This might require:
        # 1. Changing the type hint to Union[DataSourceConfig, StorageConfig]
        # 2. Or having a discriminated union with proper validation
        
        # Test with StorageConfig directly
        config = {
            "test_storage": StorageConfig(
                type="bigquery",
                project_id="test-project",
                dataset_id="test_dataset",
                table_id="test_table",
                auto_create=True,
                clustering_fields=["record_id", "timestamp"]
            )
        }
        
        # This should work after the fix
        orchestrator = TestOrchestrator(
            orchestrator="test",
            name="test_orchestrator",
            storage=config
        )
        
        assert isinstance(orchestrator.storage["test_storage"], StorageConfig)
        assert orchestrator.storage["test_storage"].auto_create is True
        assert orchestrator.storage["test_storage"].clustering_fields == ["record_id", "timestamp"]
    
    def test_omegaconf_to_storageconfig_conversion(self):
        """Test converting OmegaConf DictConfig to StorageConfig."""
        # When Hydra loads YAML, it creates DictConfig objects
        yaml_config = """
        type: bigquery
        project_id: test-project
        dataset_id: test_dataset
        table_id: test_table
        auto_create: true
        """
        
        cfg = OmegaConf.create(yaml_config)
        
        # The field validator should handle DictConfig -> dict conversion
        # This is what convert_omegaconf_objects does
        storage = StorageConfig(**OmegaConf.to_container(cfg))
        assert storage.type == "bigquery"
        assert storage.auto_create is True
    
    def test_mixed_storage_configs(self):
        """Test orchestrator with mixed DataSourceConfig and StorageConfig."""
        # After fix, orchestrator should handle both types
        orchestrator = TestOrchestrator(
            orchestrator="test",
            name="test_orchestrator",
            storage={
                "legacy": DataSourceConfig(type="file", path="/old.json"),
                "modern": StorageConfig(
                    type="bigquery",
                    project_id="test-project",
                    dataset_id="test_dataset",
                    table_id="test_table"
                )
            }
        )
        
        assert isinstance(orchestrator.storage["legacy"], DataSourceConfig)
        assert isinstance(orchestrator.storage["modern"], StorageConfig)
    
    def test_yaml_dict_to_storageconfig_conversion(self):
        """Test that raw YAML dictionaries are converted to appropriate config types."""
        # This simulates what Hydra passes when loading YAML
        yaml_data = {
            "orchestrator": "test",
            "name": "test_orchestrator",
            "storage": {
                "bigquery_source": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "table_id": "test_table",
                    "auto_create": True,  # StorageConfig-specific field
                    "clustering_fields": ["record_id", "timestamp"]  # StorageConfig-specific field
                },
                "file_source": {
                    "type": "file",
                    "path": "/data/test.json",
                    "glob": "*.json"
                }
            }
        }
        
        # The model validator should convert these to proper config objects
        orchestrator = TestOrchestrator(**yaml_data)
        
        # BigQuery source should be converted to StorageConfig due to specific fields
        assert isinstance(orchestrator.storage["bigquery_source"], StorageConfig)
        assert orchestrator.storage["bigquery_source"].auto_create is True
        assert orchestrator.storage["bigquery_source"].clustering_fields == ["record_id", "timestamp"]
        
        # File source should be converted to DataSourceConfig for backward compatibility
        assert isinstance(orchestrator.storage["file_source"], DataSourceConfig)
        assert orchestrator.storage["file_source"].path == "/data/test.json"
    
    def test_storageconfig_validation_with_extra_forbid(self):
        """Test that StorageConfig validation works properly with extra='forbid'."""
        # This tests the core issue - StorageConfig has extra='forbid' but should still work
        yaml_data = {
            "orchestrator": "test",
            "name": "test_orchestrator", 
            "storage": {
                "test": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "table_id": "test_table",
                    "auto_create": True
                    # No invalid fields - should work fine
                }
            }
        }
        
        orchestrator = TestOrchestrator(**yaml_data)
        assert isinstance(orchestrator.storage["test"], StorageConfig)
        
        # Now test with an invalid field
        yaml_data_invalid = {
            "orchestrator": "test",
            "name": "test_orchestrator",
            "storage": {
                "test": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "invalid_field": "should_fail"  # This should cause validation error
                }
            }
        }
        
        # The validator should catch this and fall back to DataSourceConfig
        orchestrator = TestOrchestrator(**yaml_data_invalid)
        # With our implementation, invalid StorageConfig falls back to DataSourceConfig
        assert isinstance(orchestrator.storage["test"], DataSourceConfig)