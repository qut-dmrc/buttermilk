"""Compatibility layer for migration from old storage patterns to unified storage.

This module provides deprecated classes that delegate to the new unified storage
framework while maintaining backward compatibility.
"""

import warnings
from typing import TYPE_CHECKING, Iterator, Any

from buttermilk._core.types import Record
from .._core.storage_config import StorageConfig
from .bigquery import BigQueryStorage

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM


class BigQueryRecordLoader:
    """Compatibility wrapper for BigQueryStorage (DEPRECATED).
    
    This class maintains backward compatibility with the old BigQueryRecordLoader
    while delegating all operations to the new unified BigQueryStorage.
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize BigQuery loader (DEPRECATED).
        
        Args:
            config: Configuration object (deprecated pattern)
            **kwargs: Direct configuration parameters
        """
        warnings.warn(
            "BigQueryRecordLoader is deprecated. Use buttermilk.storage.BigQueryStorage instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert old configuration pattern to new StorageConfig
        storage_config = self._convert_config(config, kwargs)
        
        # Create unified storage instance
        # Note: BM instance not available here, so BigQueryStorage will create its own client
        self._storage = BigQueryStorage(storage_config, bm=None)
    
    def _convert_config(self, config, kwargs) -> StorageConfig:
        """Convert old configuration pattern to new StorageConfig."""
        config_data = {
            "type": "bigquery",
        }
        
        # Extract from config object if provided
        if config:
            config_data.update({
                "project_id": getattr(config, "project_id", None),
                "dataset_id": getattr(config, "dataset_id", None),
                "table_id": getattr(config, "table_id", None),
                "dataset_name": getattr(config, "dataset_name", None),
                "split_type": getattr(config, "split_type", None),
                "randomize": getattr(config, "randomize", True),
                "batch_size": getattr(config, "batch_size", 1000),
                "limit": getattr(config, "limit", None),
            })
        
        # Override with kwargs
        config_data.update(kwargs)
        
        # Ensure required fields have defaults
        if not config_data.get("dataset_id"):
            config_data["dataset_id"] = "buttermilk"
        if not config_data.get("table_id"):
            config_data["table_id"] = "records"
        
        return StorageConfig(**{k: v for k, v in config_data.items() if v is not None})
    
    def __iter__(self) -> Iterator[Record]:
        """Iterate over records from BigQuery."""
        return iter(self._storage)
    
    def __len__(self) -> int:
        """Return number of records if known, 0 if streaming/unknown."""
        return len(self._storage)
    
    def count(self) -> int:
        """Get the total count of records matching the criteria."""
        return self._storage.count()
    
    def create_table_if_not_exists(self) -> None:
        """Create the Records table if it doesn't exist."""
        self._storage.create()
    
    def insert_records(self, records: list[Record]) -> None:
        """Insert Record objects into the BigQuery table."""
        self._storage.save(records)


def create_bigquery_loader(config: dict[str, Any]) -> BigQueryRecordLoader:
    """Factory function to create BigQueryRecordLoader from config (DEPRECATED)."""
    warnings.warn(
        "create_bigquery_loader is deprecated. Use BM.get_storage() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return BigQueryRecordLoader(**config)


class DatasetMigrator:
    """Compatibility wrapper for dataset migration (DEPRECATED).
    
    This class maintains backward compatibility while using the new unified storage.
    """
    
    def __init__(self, project_id: str | None = None, dataset_id: str | None = None, table_id: str | None = None):
        """Initialize DatasetMigrator (DEPRECATED).
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID 
            table_id: BigQuery table ID
        """
        warnings.warn(
            "DatasetMigrator is deprecated. Use unified storage classes directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create storage config
        self.storage_config = StorageConfig(
            type="bigquery",
            project_id=project_id,
            dataset_id=dataset_id or "buttermilk",
            table_id=table_id or "records"
        )
        
        # For compatibility
        self.project_id = self.storage_config.project_id
        self.dataset_id = self.storage_config.dataset_id
        self.table_id = self.storage_config.table_id
        
        # Create client for backward compatibility
        from google.cloud import bigquery
        self.client = bigquery.Client(project=self.project_id)
    
    def migrate_jsonl_to_bigquery(
        self, 
        source_path: str, 
        dataset_name: str,
        split_type: str = "train",
        field_mapping: dict[str, str] | None = None,
        batch_size: int = 1000
    ) -> None:
        """Migrate JSONL file to BigQuery Records table (DEPRECATED)."""
        # Create source storage for reading
        source_config = StorageConfig(
            type="file",
            path=source_path,
            columns=field_mapping or {}
        )
        source_storage = FileStorage(source_config, bm=None)
        
        # Create target storage for writing
        target_config = self.storage_config.model_copy()
        target_config.dataset_name = dataset_name
        target_config.split_type = split_type
        target_config.batch_size = batch_size
        
        target_storage = BigQueryStorage(target_config, bm=None)
        
        # Migrate data
        batch = []
        total_processed = 0
        
        for record in source_storage:
            # Ensure record has required metadata
            record.metadata["dataset_name"] = dataset_name
            record.metadata["split_type"] = split_type
            record.metadata["migrated_from"] = source_path
            
            batch.append(record)
            
            if len(batch) >= batch_size:
                target_storage.save(batch)
                total_processed += len(batch)
                batch = []
        
        # Save remaining records
        if batch:
            target_storage.save(batch)
            total_processed += len(batch)
    
    def create_mapping_template(self, sample_jsonl_path: str, output_path: str | None = None) -> dict[str, str]:
        """Create field mapping template (DEPRECATED)."""
        # This method can remain largely unchanged as it's utility functionality
        import json
        sample_fields = set()
        
        with open(sample_jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                try:
                    record = json.loads(line)
                    sample_fields.update(record.keys())
                except json.JSONDecodeError:
                    continue
        
        record_fields = {
            "record_id", "content", "metadata", "alt_text", 
            "ground_truth", "uri", "mime"
        }
        
        mapping_template = {
            "# Field mappings from JSONL to Record format": "",
            "# Required fields": "",
            "record_id": "id",
            "content": "text",
            "# Optional fields": "",
        }
        
        for field in sample_fields:
            if field not in ["id", "text"] and field not in record_fields:
                mapping_template[f"# {field}"] = f"maps to metadata.{field}"
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(mapping_template, f, indent=2)
        
        return mapping_template


# Import FileStorage here to avoid circular imports
from .file import FileStorage