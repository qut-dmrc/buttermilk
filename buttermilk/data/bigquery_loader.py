"""BigQuery record dataloader for Buttermilk (DEPRECATED).

This module is deprecated and maintained for backward compatibility only.
Use buttermilk.storage.BigQueryStorage via the BM singleton instead.

Example migration:
    # Old way (deprecated)
    from buttermilk.data.bigquery_loader import BigQueryRecordLoader
    loader = BigQueryRecordLoader(dataset_name="test")

    # New way (recommended) - use BM factory methods
    from buttermilk._core.dmrc import get_bm
    bm = get_bm()
    storage = bm.get_bigquery_storage("test")

    # For advanced configuration
    from buttermilk.storage import StorageConfig
    config = StorageConfig(type="bigquery", dataset_name="test", randomize=False)
    storage = bm.get_storage(config)
"""

# Import the compatibility wrapper
from buttermilk.storage.compat import BigQueryRecordLoader, create_bigquery_loader

# Re-export for backward compatibility
__all__ = ["BigQueryRecordLoader", "create_bigquery_loader"]
