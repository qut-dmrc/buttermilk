"""BigQuery record dataloader for Buttermilk.

This loader provides seamless integration with BigQuery Records tables that mirror
the Record object fields, with clustering by record_id and dataset name.
"""

import datetime
import json
from typing import Any, Iterator

from google.cloud import bigquery

from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk.data.loaders import DataLoader


class BigQueryRecordLoader(DataLoader):
    """BigQuery dataloader for Records table.
    
    Core functionality:
    1. Defaults to a generic `Records` table clustered by record_id and dataset name
    2. Seamless integration with other dataloader objects  
    3. YAML file configuration support
    4. Streaming randomized Record rows
    """

    def __init__(self, config=None, **kwargs):
        """Initialize BigQuery loader."""
        if config:
            super().__init__(config)
            # Extract BigQuery-specific config from the config object
            self.project_id = getattr(config, "project_id", kwargs.get("project_id"))
            self.dataset_id = getattr(config, "dataset_id", kwargs.get("dataset_id", "buttermilk"))
            self.table_id = getattr(config, "table_id", kwargs.get("table_id", "records"))
            self.dataset_name = getattr(config, "dataset_name", kwargs.get("dataset_name"))
            self.split_type = getattr(config, "split_type", kwargs.get("split_type", "train"))
            self.randomize = getattr(config, "randomize", kwargs.get("randomize", True))
            self.batch_size = getattr(config, "batch_size", kwargs.get("batch_size", 1000))
            self.limit = getattr(config, "limit", kwargs.get("limit"))
        else:
            # Direct initialization with kwargs
            self.project_id = kwargs["project_id"]
            self.dataset_id = kwargs.get("dataset_id", "buttermilk")
            self.table_id = kwargs.get("table_id", "records")
            self.dataset_name = kwargs["dataset_name"]
            self.split_type = kwargs.get("split_type", "train")
            self.randomize = kwargs.get("randomize", True)
            self.batch_size = kwargs.get("batch_size", 1000)
            self.limit = kwargs.get("limit")

        self.client = bigquery.Client(project=self.project_id)
        self.table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    def _build_query(self) -> str:
        """Build the BigQuery SQL query."""
        base_query = f"""
        SELECT 
            record_id,
            content,
            metadata,
            alt_text,
            ground_truth,
            uri,
            mime
        FROM `{self.table_ref}`
        WHERE dataset_name = @dataset_name
        """

        if self.split_type:
            base_query += " AND split_type = @split_type"

        if self.randomize:
            base_query += " ORDER BY RAND()"
        else:
            base_query += " ORDER BY record_id"

        if self.limit:
            base_query += f" LIMIT {self.limit}"

        return base_query

    def _parse_record(self, row: bigquery.Row) -> Record:
        """Parse a BigQuery row into a Record object."""
        try:
            # Parse JSON fields
            metadata = json.loads(row.metadata) if row.metadata else {}
            ground_truth = json.loads(row.ground_truth) if row.ground_truth else None

            # Create Record object
            record = Record(
                record_id=row.record_id,
                content=row.content,
                metadata=metadata,
                alt_text=row.alt_text,
                ground_truth=ground_truth,
                uri=row.uri,
                mime=row.mime or "text/plain"
            )

            return record

        except Exception as e:
            logger.warning(f"Error parsing BigQuery row {row.record_id}: {e}")
            # Return a minimal record on parse error
            return Record(
                record_id=row.record_id or "error",
                content=str(row.content) if row.content else "Error loading content",
                metadata={"parse_error": str(e)}
            )

    def __iter__(self) -> Iterator[Record]:
        """Iterate over records from BigQuery."""
        try:
            query = self._build_query()

            # Configure query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", self.dataset_name),
                    bigquery.ScalarQueryParameter("split_type", "STRING", self.split_type),
                ]
            )

            logger.info(f"Loading records from {self.table_ref} for dataset '{self.dataset_name}', split '{self.split_type}'")

            # Execute query
            query_job = self.client.query(query, job_config=job_config)

            # Stream results
            for row in query_job:
                yield self._parse_record(row)

        except Exception as e:
            logger.error(f"Error loading records from BigQuery: {e}")
            raise

    def count(self) -> int:
        """Get the total count of records matching the criteria."""
        try:
            query = f"""
            SELECT COUNT(*) as total
            FROM `{self.table_ref}`
            WHERE dataset_name = @dataset_name
            """

            if self.split_type:
                query += " AND split_type = @split_type"

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", self.dataset_name),
                    bigquery.ScalarQueryParameter("split_type", "STRING", self.split_type),
                ]
            )

            query_job = self.client.query(query, job_config=job_config)
            result = list(query_job)[0]
            return result.total

        except Exception as e:
            logger.warning(f"Error counting records: {e}")
            return 0

    def create_table_if_not_exists(self) -> None:
        """Create the Records table with proper schema and clustering if it doesn't exist."""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

            # Define schema matching Record object
            schema = [
                bigquery.SchemaField("record_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("dataset_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("split_type", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("alt_text", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("ground_truth", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("uri", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("mime", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
            ]

            # Create table with clustering
            table = bigquery.Table(table_id, schema=schema)
            table.clustering_fields = ["record_id", "dataset_name"]
            table.description = "Buttermilk Records table with clustering by record_id and dataset_name"

            # Attempt to create table
            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"Created/verified Records table: {table_id}")

        except Exception as e:
            logger.error(f"Error creating Records table: {e}")
            raise

    def insert_records(self, records: list[Record]) -> None:
        """Insert Record objects into the BigQuery table."""
        try:
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

            # Convert Record objects to BigQuery rows
            rows_to_insert = []
            for record in records:
                row = {
                    "record_id": record.record_id,
                    "dataset_name": self.dataset_name,
                    "split_type": self.split_type,
                    "content": record.content if isinstance(record.content, str) else json.dumps(record.content),
                    "metadata": json.dumps(record.metadata),
                    "alt_text": record.alt_text,
                    "ground_truth": json.dumps(record.ground_truth) if record.ground_truth else None,
                    "uri": record.uri,
                    "mime": record.mime,
                    "created_at": datetime.datetime.now(datetime.UTC),
                    "updated_at": datetime.datetime.now(datetime.UTC),
                }
                rows_to_insert.append(row)

            # Insert rows
            table = self.client.get_table(table_ref)
            errors = self.client.insert_rows_json(table, rows_to_insert)

            if errors:
                logger.error(f"Error inserting records: {errors}")
                raise Exception(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Successfully inserted {len(records)} records into {table_ref}")

        except Exception as e:
            logger.error(f"Error inserting records into BigQuery: {e}")
            raise


def create_bigquery_loader(config: dict[str, Any]) -> BigQueryRecordLoader:
    """Factory function to create BigQueryRecordLoader from config."""
    return BigQueryRecordLoader(**config)
