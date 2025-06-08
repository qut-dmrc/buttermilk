"""Migration utility to convert JSONL datasets to BigQuery Records tables (DEPRECATED).

This utility is deprecated and maintained for backward compatibility only.
Use buttermilk.storage classes via the BM singleton for new implementations.

Example migration:
    # Old way (deprecated)
    from buttermilk.tools.migrate_to_bigquery import DatasetMigrator
    migrator = DatasetMigrator(project_id="my-project")
    
    # New way (recommended) - use BM factory methods
    from buttermilk._core.dmrc import get_bm
    from buttermilk.storage import StorageConfig
    
    bm = get_bm()
    
    # Source storage
    source_config = StorageConfig(type="file", path="data.jsonl")
    source = bm.get_storage(source_config)
    
    # Target storage - use convenience method
    target = bm.get_bigquery_storage("my_dataset")
    
    # Migrate
    records = list(source)
    target.save(records)
"""

import warnings

# Import the compatibility classes
from buttermilk.storage.compat import DatasetMigrator

# Import click for CLI compatibility
import click

# Re-export for backward compatibility
__all__ = ["DatasetMigrator", "cli"]


@click.group()
def cli():
    """Buttermilk dataset migration utilities (DEPRECATED)."""
    warnings.warn(
        "The migrate_to_bigquery CLI is deprecated. Use unified storage classes directly.",
        DeprecationWarning,
        stacklevel=2
    )


@cli.command()
@click.option("--source", required=True, help="Source JSONL file path (local or GCS)")
@click.option("--dataset-name", required=True, help="Dataset name (e.g., osb, tox, drag)")
@click.option("--project-id", help="Google Cloud Project ID (uses default from config if not provided)")
@click.option("--split-type", default="train", help="Split type (train/test/val)")
@click.option("--mapping-file", help="JSON file with field mappings")
@click.option("--batch-size", default=1000, help="Batch size for inserts")
def migrate_jsonl(source: str, dataset_name: str, project_id: str | None, split_type: str,
                  mapping_file: str | None, batch_size: int):
    """Migrate JSONL file to BigQuery Records table (DEPRECATED)."""
    warnings.warn(
        "migrate_jsonl command is deprecated. Use unified storage classes directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Load field mapping if provided
    field_mapping = None
    if mapping_file:
        import json
        with open(mapping_file, "r") as f:
            field_mapping = json.load(f)
    
    # Create migrator and run migration
    migrator = DatasetMigrator(project_id)
    migrator.migrate_jsonl_to_bigquery(
        source_path=source,
        dataset_name=dataset_name,
        split_type=split_type,
        field_mapping=field_mapping,
        batch_size=batch_size
    )


@cli.command()
@click.option("--source", required=True, help="Sample JSONL file to analyze")
@click.option("--output", help="Output path for mapping template")
def create_mapping(source: str, output: str | None):
    """Create field mapping template from sample JSONL file (DEPRECATED)."""
    warnings.warn(
        "create_mapping command is deprecated.",
        DeprecationWarning,
        stacklevel=2
    )
    
    migrator = DatasetMigrator()
    mapping = migrator.create_mapping_template(source, output)
    
    print("\\nField mapping template:")
    import json
    print(json.dumps(mapping, indent=2))


@cli.command()
@click.option("--project-id", help="Google Cloud Project ID (uses default from config if not provided)")
@click.option("--dataset-name", required=True, help="Dataset name to test loading")
@click.option("--limit", default=5, help="Number of records to load for testing")
def test_load(project_id: str | None, dataset_name: str, limit: int):
    """Test loading records from BigQuery table (DEPRECATED)."""
    warnings.warn(
        "test_load command is deprecated. Use storage classes directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from buttermilk.data.bigquery_loader import BigQueryRecordLoader
    
    loader = BigQueryRecordLoader(
        project_id=project_id,
        dataset_name=dataset_name,
        limit=limit
    )
    
    print(f"\\nLoading {limit} records from {dataset_name}:")
    for i, record in enumerate(loader):
        print(f"\\nRecord {i+1}:")
        print(f"  ID: {record.record_id}")
        print(f"  Content: {record.content[:100]}...")
        print(f"  Metadata keys: {list(record.metadata.keys())}")


if __name__ == "__main__":
    cli()