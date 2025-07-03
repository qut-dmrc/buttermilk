"""Integration tests for Issue #79 - Storage config validation in Orchestrator.

These tests verify that the Orchestrator properly validates and converts storage
configurations, handling both legacy DataSourceConfig and modern StorageConfig
objects, with proper discrimination based on field presence.
"""

import asyncio
import json
import tempfile
from pathlib import Path
import pytest

from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.storage_config import StorageConfig
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.types import RunRequest

pytestmark = pytest.mark.anyio


class TestOrchestrator(Orchestrator):
    """Test orchestrator for integration tests."""
    
    setup_called: bool = False
    cleanup_called: bool = False
    storage_validation_results: dict = {}
    
    async def _setup(self, request: RunRequest) -> None:
        self.setup_called = True
        
        # Record the types of storage configs that were created
        for name, config in self.storage.items():
            self.storage_validation_results[name] = {
                'type': type(config).__name__,
                'config': config
            }
    
    async def _cleanup(self) -> None:
        self.cleanup_called = True
    
    async def _run(self, request: RunRequest) -> None:
        await self._setup(request)


class TestIssue79Integration:
    """Integration tests for Issue #79 - Storage config validation fix."""
    
    async def test_yaml_config_simulation(self):
        """Test that YAML-like dictionaries are properly converted to config objects."""
        # Create test data
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test JSONL file
            test_file = Path(tmp_dir) / "test_data.jsonl"
            test_records = [
                {"record_id": "demo1", "content": "First demo record"},
                {"record_id": "demo2", "content": "Second demo record"},
            ]
            
            with open(test_file, 'w') as f:
                for record in test_records:
                    f.write(json.dumps(record) + '\n')
            
            # This simulates what Hydra would pass when loading YAML
            yaml_simulation = {
                "orchestrator": "test",
                "name": "issue_79_test", 
                "description": "Testing fixed storage config validation",
                "storage": {
                    # Has StorageConfig-specific fields
                    "bigquery_storage": {
                        "type": "bigquery",
                        "project_id": "test-project", 
                        "dataset_id": "test_dataset",
                        "table_id": "test_table",
                        "auto_create": True,  # StorageConfig-specific
                        "clustering_fields": ["record_id", "timestamp"],  # StorageConfig-specific
                        "batch_size": 500
                    },
                    # Legacy file config
                    "file_storage": {
                        "type": "file",
                        "path": str(test_file),
                        "glob": "*.jsonl"
                    }
                }
            }
            
            # Verify input is raw dictionaries
            assert isinstance(yaml_simulation['storage']['bigquery_storage'], dict)
            assert isinstance(yaml_simulation['storage']['file_storage'], dict)
            
            # Create orchestrator - this triggers validation
            orchestrator = TestOrchestrator(**yaml_simulation)
            
            # Verify conversion happened correctly
            bq_config = orchestrator.storage["bigquery_storage"]
            file_config = orchestrator.storage["file_storage"]
            
            # BigQuery config should become StorageConfig due to specific fields
            assert isinstance(bq_config, StorageConfig)
            assert bq_config.auto_create is True
            assert bq_config.clustering_fields == ["record_id", "timestamp"]
            assert bq_config.batch_size == 500
            
            # File config should become DataSourceConfig for backward compatibility
            assert isinstance(file_config, DataSourceConfig)
            assert file_config.path == str(test_file)
            assert file_config.glob == "*.jsonl"


    async def test_mixed_config_objects(self):
        """Test that orchestrator works with pre-created config objects of both types."""
        # Create orchestrator with pre-created config objects (not dicts)
        orchestrator = TestOrchestrator(
            orchestrator="test",
            name="mixed_config_test",
            storage={
                "legacy": DataSourceConfig(
                    type="file",
                    path="/nonexistent/legacy.json"
                ),
                "modern": StorageConfig(
                    type="bigquery",
                    project_id="test-project",
                    dataset_id="test_dataset", 
                    table_id="test_table",
                    auto_create=False,
                    clustering_fields=["custom_field"]
                )
            }
        )
        
        # Verify both types are preserved
        assert isinstance(orchestrator.storage["legacy"], DataSourceConfig)
        assert isinstance(orchestrator.storage["modern"], StorageConfig)
        
        # Verify specific fields
        legacy_config = orchestrator.storage["legacy"]
        modern_config = orchestrator.storage["modern"]
        
        assert legacy_config.type == "file"
        assert legacy_config.path == "/nonexistent/legacy.json"
        
        assert modern_config.type == "bigquery"
        assert modern_config.auto_create is False
        assert modern_config.clustering_fields == ["custom_field"]

    async def test_validation_with_invalid_fields(self):
        """Test graceful fallback when StorageConfig validation fails."""
        # Config with invalid StorageConfig field - should fall back to DataSourceConfig
        yaml_with_invalid = {
            "orchestrator": "test",
            "name": "validation_test",
            "storage": {
                "test_storage": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "auto_create": True,  # This triggers StorageConfig attempt
                    "invalid_field_name": "this should cause fallback"  # Invalid field
                }
            }
        }
        
        # Should not raise exception, should fall back gracefully
        orchestrator = TestOrchestrator(**yaml_with_invalid)
        
        config = orchestrator.storage["test_storage"]
        # Should fallback to DataSourceConfig when StorageConfig validation fails
        assert isinstance(config, DataSourceConfig)
        assert config.type == "bigquery"
        assert config.project_id == "test-project"
        # invalid_field_name should be ignored due to extra="ignore" in DataSourceConfig

    async def test_discrimination_logic(self):
        """Test the discrimination logic that chooses between config types."""
        test_cases = [
            # Should become StorageConfig due to auto_create
            {
                "name": "with_auto_create",
                "config": {
                    "type": "bigquery",
                    "auto_create": True
                },
                "expected_type": StorageConfig
            },
            # Should become StorageConfig due to clustering_fields
            {
                "name": "with_clustering_fields", 
                "config": {
                    "type": "bigquery",
                    "clustering_fields": ["id", "timestamp"]
                },
                "expected_type": StorageConfig
            },
            # Should become StorageConfig due to type="chromadb"
            {
                "name": "chromadb_type",
                "config": {
                    "type": "chromadb",
                    "collection_name": "test"
                },
                "expected_type": StorageConfig
            },
            # Should become DataSourceConfig (no special fields)
            {
                "name": "basic_file",
                "config": {
                    "type": "file",
                    "path": "/test.json"
                },
                "expected_type": DataSourceConfig
            }
        ]
        
        for case in test_cases:
            orchestrator = TestOrchestrator(
                orchestrator="test",
                name=f"discrimination_test_{case['name']}",
                storage={
                    "test": case["config"]
                }
            )
            
            config = orchestrator.storage["test"]
            assert isinstance(config, case["expected_type"]), \
                f"Case {case['name']}: expected {case['expected_type'].__name__}, got {type(config).__name__}"

    async def test_protocol_validation(self):
        """Test that OrchestratorProtocol validation works independently."""
        # Test creating protocol directly with mixed storage types
        yaml_config = {
            "orchestrator": "test",
            "name": "protocol_test",
            "storage": {
                "storage_config_type": {
                    "type": "bigquery",
                    "auto_create": True,
                    "clustering_fields": ["record_id"]
                },
                "datasource_config_type": {
                    "type": "file", 
                    "path": "/test.json",
                    "glob": "*.json"
                }
            }
        }
        
        protocol = OrchestratorProtocol(**yaml_config)
        
        # Verify discrimination worked
        assert isinstance(protocol.storage["storage_config_type"], StorageConfig)
        assert isinstance(protocol.storage["datasource_config_type"], DataSourceConfig)
        
        # Verify fields preserved
        storage_config = protocol.storage["storage_config_type"]
        datasource_config = protocol.storage["datasource_config_type"]
        
        assert storage_config.auto_create is True
        assert storage_config.clustering_fields == ["record_id"]
        assert datasource_config.path == "/test.json"
        assert datasource_config.glob == "*.json"