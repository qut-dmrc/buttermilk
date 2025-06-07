"""Migration utility to convert JSONL datasets to BigQuery Records tables.

This utility provides an easy interface for HASS scholars to map their datasets
to the Record format and upload to BigQuery.
"""

import json

import click
from google.cloud import bigquery

from buttermilk._core.config import BigQueryConfig
from buttermilk._core.log import logger
from buttermilk.data.bigquery_loader import BigQueryRecordLoader
from buttermilk.data.loaders import DataSourceConfig, create_data_loader


class DatasetMigrator:
    """Handles migration of datasets from various sources to BigQuery Records tables."""

    def __init__(self, project_id: str | None = None, dataset_id: str | None = None, table_id: str | None = None):
        """Initialize DatasetMigrator with BigQuery configuration.
        
        Args:
            project_id: Google Cloud project ID. If None, uses default from BigQueryConfig.
            dataset_id: BigQuery dataset ID. If None, uses default from BigQueryConfig.
            table_id: BigQuery table ID. If None, uses default from BigQueryConfig.
        """
        bq_defaults = BigQueryConfig()

        self.project_id = project_id or bq_defaults.project_id
        if not self.project_id:
            raise ValueError("project_id must be provided or available from environment/credentials")
        self.dataset_id = dataset_id or bq_defaults.dataset_id
        self.table_id = table_id or bq_defaults.table_id
        self.client = bigquery.Client(project=self.project_id)

    def migrate_jsonl_to_bigquery(
        self,
        source_path: str,
        dataset_name: str,
        split_type: str = "train",
        field_mapping: dict[str, str] | None = None,
        batch_size: int = 1000
    ) -> None:
        """Migrate JSONL file to BigQuery Records table.
        
        Args:
            source_path: Path to JSONL file (local or GCS)
            dataset_name: Name for the dataset (e.g., 'osb', 'tox', 'drag')
            split_type: Split type ('train', 'test', 'val')
            field_mapping: Optional mapping of JSONL fields to Record fields
            batch_size: Number of records to insert at once
        """
        logger.info(f"Starting migration of {source_path} to BigQuery")

        # Create BigQuery loader for target table
        bq_loader = BigQueryRecordLoader(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_id=self.table_id,
            dataset_name=dataset_name,
            split_type=split_type
        )

        # Ensure table exists
        bq_loader.create_table_if_not_exists()

        # Create source data loader
        source_config = DataSourceConfig(
            type="file",
            path=source_path,
            columns=field_mapping
        )
        source_loader = create_data_loader(source_config)

        # Process records in batches
        batch = []
        total_processed = 0

        for record in source_loader:
            # Ensure record has required metadata
            record.metadata["dataset_name"] = dataset_name
            record.metadata["split_type"] = split_type
            record.metadata["migrated_from"] = source_path

            batch.append(record)

            if len(batch) >= batch_size:
                bq_loader.insert_records(batch)
                total_processed += len(batch)
                logger.info(f"Processed {total_processed} records...")
                batch = []

        # Insert remaining records
        if batch:
            bq_loader.insert_records(batch)
            total_processed += len(batch)

        logger.info(f"Migration complete! Processed {total_processed} total records")

    def create_mapping_template(self, sample_jsonl_path: str, output_path: str | None = None) -> dict[str, str]:
        """Create a field mapping template by analyzing a sample JSONL file.
        
        Args:
            sample_jsonl_path: Path to sample JSONL file
            output_path: Optional path to save mapping template
            
        Returns:
            Dictionary template for field mapping
        """
        sample_fields = set()

        # Analyze first few records to identify fields
        with open(sample_jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Sample first 10 records
                    break
                try:
                    record = json.loads(line)
                    sample_fields.update(record.keys())
                except json.JSONDecodeError:
                    continue

        # Create mapping template
        record_fields = {
            "record_id", "content", "metadata", "alt_text",
            "ground_truth", "uri", "mime"
        }

        mapping_template = {
            "# Field mappings from JSONL to Record format": "",
            "# Required fields": "",
            "record_id": "id",  # Common field name
            "content": "text",  # Common field name
            "# Optional fields": "",
        }

        # Add suggestions for optional fields
        for field in sample_fields:
            if field not in ["id", "text"] and field not in record_fields:
                mapping_template[f"# {field}"] = f"maps to metadata.{field}"

        if output_path:
            with open(output_path, "w") as f:
                json.dump(mapping_template, f, indent=2)
            logger.info(f"Mapping template saved to {output_path}")

        return mapping_template


@click.group()
def cli():
    """Buttermilk dataset migration utilities."""
    pass


@cli.command()
@click.option("--source", required=True, help="Source JSONL file path (local or GCS)")
@click.option("--dataset-name", required=True, help="Dataset name (e.g., osb, tox, drag)")
@click.option("--project-id", help="Google Cloud Project ID (uses default from config if not provided)")
@click.option("--split-type", default="train", help="Split type (train/test/val)")
@click.option("--mapping-file", help="JSON file with field mappings")
@click.option("--batch-size", default=1000, help="Batch size for inserts")
def migrate_jsonl(source: str, dataset_name: str, project_id: str | None, split_type: str,
                  mapping_file: str | None, batch_size: int):
    """Migrate JSONL file to BigQuery Records table."""

    # Load field mapping if provided
    field_mapping = None
    if mapping_file:
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
    """Create field mapping template from sample JSONL file."""

    migrator = DatasetMigrator()  # Uses default config, project ID not needed for this operation
    mapping = migrator.create_mapping_template(source, output)

    print("\\nField mapping template:")
    print(json.dumps(mapping, indent=2))


@cli.command()
@click.option("--project-id", help="Google Cloud Project ID (uses default from config if not provided)")
@click.option("--dataset-name", required=True, help="Dataset name to test loading")
@click.option("--limit", default=5, help="Number of records to load for testing")
def test_load(project_id: str | None, dataset_name: str, limit: int):
    """Test loading records from BigQuery table."""

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
