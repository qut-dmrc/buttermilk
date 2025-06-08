"""BigQuery record dataloader for Buttermilk (DEPRECATED).

This module is deprecated and maintained for backward compatibility only.
Use buttermilk.storage.BigQueryStorage instead.

Example migration:
    # Old way (deprecated)
    from buttermilk.data.bigquery_loader import BigQueryRecordLoader
    loader = BigQueryRecordLoader(dataset_name="test")
    
    # New way (recommended) 
    storage = bm.get_bigquery_storage("test")
    # or
    from buttermilk.storage import BigQueryStorage, StorageConfig
    config = StorageConfig(type="bigquery", dataset_name="test")
    storage = BigQueryStorage(config, bm)
"""

# Import the compatibility wrapper
from buttermilk.storage.compat import BigQueryRecordLoader, create_bigquery_loader

# Re-export for backward compatibility
__all__ = ["BigQueryRecordLoader", "create_bigquery_loader"]