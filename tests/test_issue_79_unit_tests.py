"""Unit tests demonstrating that Issue #79 is fixed.

Shows that raw YAML dictionaries are now properly converted to StorageConfig
objects when they contain StorageConfig-specific fields.
"""

import pytest
from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk._core.storage_config import StorageConfig
from buttermilk._core.config import DataSourceConfig


class TestIssue79StorageDiscrimination:
    """Unit tests for Issue #79 storage config discrimination logic."""

    def test_issue_79_fix_demonstration(self):
        """Show the exact issue from #79 is now fixed."""
        # This simulates what Hydra would pass when loading YAML
        # Before the fix: these raw dicts would become DataSourceConfig objects
        # and StorageConfig-specific fields would be lost (silently ignored)
        
        yaml_like_config = {
            "orchestrator": "test",
            "name": "issue_79_test",
            "storage": {
                "problematic_config": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "dataset_id": "test_dataset", 
                    "table_id": "test_table",
                    # These fields were the problem - they exist in StorageConfig but not DataSourceConfig
                    "auto_create": True,           # StorageConfig-specific
                    "clustering_fields": ["record_id", "created_at"],  # StorageConfig-specific
                    "batch_size": 2000            # Different default than DataSourceConfig
                },
                "legacy_config": {
                    "type": "file",
                    "path": "/data/old_format.json",
                    "glob": "*.json"
                }
            }
        }
        
        # Verify input is raw dictionaries
        for name, config in yaml_like_config["storage"].items():
            assert isinstance(config, dict)
        
        # This is where the magic happens - our model validator converts the dicts
        protocol = OrchestratorProtocol(**yaml_like_config)
        
        # Verify conversion worked correctly
        problematic = protocol.storage["problematic_config"]
        legacy = protocol.storage["legacy_config"]
        
        # The problematic config should now be StorageConfig with fields preserved
        assert isinstance(problematic, StorageConfig)
        assert problematic.auto_create is True
        assert problematic.clustering_fields == ["record_id", "created_at"]
        assert problematic.batch_size == 2000
        
        # Legacy config should be DataSourceConfig for backward compatibility
        assert isinstance(legacy, DataSourceConfig)
        assert legacy.type == "file"
        assert legacy.path == "/data/old_format.json"
        
        # Success - no need to return anything in a test


    def test_error_handling(self):
        """Show that validation errors are handled gracefully."""
        # Config that would fail StorageConfig validation (invalid field)
        config_with_error = {
            "orchestrator": "test", 
            "name": "error_test",
            "storage": {
                "bad_config": {
                    "type": "bigquery",
                    "auto_create": True,  # This makes it try StorageConfig first
                    "invalid_field": "this should cause error"  # But this will fail validation
                }
            }
        }
        
        # Should not raise exception, should fall back gracefully
        protocol = OrchestratorProtocol(**config_with_error)
        result_config = protocol.storage["bad_config"]
        
        # Should fallback to DataSourceConfig when StorageConfig validation fails
        assert isinstance(result_config, DataSourceConfig)
        assert result_config.type == "bigquery"
        # invalid_field should be ignored due to extra="ignore" in DataSourceConfig

    def test_storage_type_discrimination(self):
        """Test that storage types are discriminated correctly."""
        test_cases = [
            # StorageConfig due to type
            ("chromadb", StorageConfig),
            ("vector", StorageConfig), 
            ("gcs", StorageConfig),
            ("s3", StorageConfig),
            # DataSourceConfig for basic types
            ("file", DataSourceConfig),
            ("plaintext", DataSourceConfig),
            ("huggingface", DataSourceConfig),
        ]
        
        for storage_type, expected_class in test_cases:
            config = {
                "orchestrator": "test",
                "name": f"test_{storage_type}",
                "storage": {
                    "test": {
                        "type": storage_type,
                        "path": "/test/path"
                    }
                }
            }
            
            protocol = OrchestratorProtocol(**config)
            result = protocol.storage["test"]
            assert isinstance(result, expected_class), \
                f"Type '{storage_type}' should create {expected_class.__name__}, got {type(result).__name__}"

    def test_field_preservation(self):
        """Test that all fields are properly preserved in conversion."""
        config = {
            "orchestrator": "test",
            "name": "field_preservation_test",
            "storage": {
                "full_config": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "table_id": "test_table", 
                    "auto_create": False,
                    "clustering_fields": ["field1", "field2"],
                    "batch_size": 1500,
                    "randomize": False
                }
            }
        }
        
        protocol = OrchestratorProtocol(**config)
        result = protocol.storage["full_config"]
        
        assert isinstance(result, StorageConfig)
        assert result.type == "bigquery"
        assert result.project_id == "test-project"
        assert result.dataset_id == "test_dataset"
        assert result.table_id == "test_table"
        assert result.auto_create is False
        assert result.clustering_fields == ["field1", "field2"]
        assert result.batch_size == 1500
        assert result.randomize is False