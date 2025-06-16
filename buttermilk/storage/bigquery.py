"""BigQuery storage implementation for unified storage operations."""

import datetime
import json
from typing import TYPE_CHECKING, Any, Iterator

from google.cloud import bigquery

from buttermilk._core.log import logger
from buttermilk._core.types import Record

from .base import Storage, StorageClient, StorageError

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM

    from .._core.storage_config import StorageConfig


class BigQueryStorage(Storage, StorageClient):
    """Unified BigQuery storage supporting both read and write operations.

    This class provides a single interface for BigQuery operations,
    replacing separate BigQueryRecordLoader and save functionality.
    """

    def __init__(self, config: "StorageConfig", bm: "BM | None" = None):
        """Initialize BigQuery storage.

        Args:
            config: Storage configuration with BigQuery settings
            bm: Buttermilk instance for BigQuery client access
        """
        super().__init__(config, bm)
        StorageClient.__init__(self, config, bm)

        if config.type != "bigquery":
            raise ValueError(f"BigQueryStorage requires type='bigquery', got '{config.type}'")

        if not config.dataset_name and not config.dataset_id:
            raise ValueError("BigQuery storage requires either dataset_name or dataset_id")

        # Validate that we have the required components for BigQuery table operations
        if not all([config.project_id, config.dataset_id, config.table_id]):
            missing_parts = []
            if not config.project_id:
                missing_parts.append("project_id")
            if not config.dataset_id:
                missing_parts.append("dataset_id")
            if not config.table_id:
                missing_parts.append("table_id")
            raise ValueError(f"BigQuery storage requires all table components: {', '.join(missing_parts)}")

        self._client = None
        self._table = None

    @property
    def client(self) -> bigquery.Client:
        """Get BigQuery client, creating it if necessary."""
        if self._client is None:
            if self.bm:
                self._client = self.get_bq_client()
            else:
                self._client = bigquery.Client(project=self.config.project_id)
        return self._client

    @property
    def table(self) -> bigquery.Table:
        """Get BigQuery table reference."""
        if self._table is None:
            table_ref = self.get_table_ref()
            self._table = self.client.get_table(table_ref)
        return self._table

    def __iter__(self) -> Iterator[Record]:
        """Iterate over records from BigQuery table.

        Yields:
            Record objects from the table
        """
        try:
            query = self._build_select_query()
            job_config = self._build_query_job_config()

            logger.info(f"Loading records from {self.get_table_ref()} for dataset '{self.config.dataset_name}'")

            query_job = self.client.query(query, job_config=job_config)

            for row in query_job:
                yield self._parse_record(row)

        except Exception as e:
            logger.error(f"Error loading records from BigQuery: {e}")
            raise StorageError(f"Failed to read from BigQuery: {e}") from e

    def save(self, records: list[Record] | Record) -> None:
        """Save records to BigQuery table.

        Uses the existing upload_rows pipeline for proper serialization and error handling.

        Args:
            records: Single record or list of records to save
        """
        if isinstance(records, Record):
            records = [records]

        if not records:
            logger.warning("No records to save")
            return

        try:
            # Ensure table exists
            if self.config.auto_create:
                self.create()

            # Convert records to list of dicts for upload_rows
            rows_to_insert = []
            for record in records:
                row = self._record_to_row(record)
                rows_to_insert.append(row)

            # Use the existing upload_rows function which handles proper serialization
            from buttermilk.utils.save import upload_rows
            from buttermilk.utils.schema_utils import get_record_bigquery_schema

            # Get the schema for proper data transformation
            schema = self.get_schema()
            if not schema:
                schema = get_record_bigquery_schema()

            # Use the proven upload_rows pipeline
            result = upload_rows(
                rows=rows_to_insert,
                schema=schema,
                dataset=self.get_table_ref()
            )

            if result:
                logger.info(f"Successfully saved {len(records)} records to {self.get_table_ref()}")
            else:
                raise StorageError("Upload failed - no result returned from upload_rows")

        except Exception as e:
            logger.error(f"Error saving records to BigQuery: {e}")
            raise StorageError(f"Failed to save to BigQuery: {e}") from e

    def count(self) -> int:
        """Count total records matching the criteria.

        Returns:
            Number of records in the table
        """
        try:
            query = f"""
            SELECT COUNT(*) as total
            FROM `{self.get_table_ref()}`
            WHERE dataset_name = @dataset_name
            """

            if self.config.split_type:
                query += " AND split_type = @split_type"

            job_config = self._build_query_job_config()
            query_job = self.client.query(query, job_config=job_config)
            result = list(query_job)[0]
            return result.total

        except Exception as e:
            logger.warning(f"Error counting records: {e}")
            return -1

    def exists(self) -> bool:
        """Check if the BigQuery table exists.

        Returns:
            True if table exists, False otherwise
        """
        try:
            self.client.get_table(self.get_table_ref())
            return True
        except Exception:
            return False

    def create(self) -> None:
        """Create the BigQuery table if it doesn't exist, or update schema if needed."""
        try:
            table_id = self.get_table_ref()

            # Use the schema from Pydantic model
            from buttermilk.utils.schema_utils import get_record_bigquery_schema

            expected_schema = get_record_bigquery_schema()

            # Use custom schema if provided in config
            if self.get_schema():
                expected_schema = self.get_schema()

            if self.exists():
                # Check if schema needs updating
                existing_table = self.client.get_table(table_id)
                existing_field_names = {field.name for field in existing_table.schema}
                expected_field_names = {field.name for field in expected_schema}
                
                missing_fields = expected_field_names - existing_field_names
                if missing_fields:
                    logger.info(f"Table {table_id} exists but missing fields: {missing_fields}")
                    logger.info(f"Updating table schema to add missing fields...")
                    
                    # Update the table schema
                    existing_table.schema = expected_schema
                    updated_table = self.client.update_table(existing_table, ["schema"])
                    logger.info(f"Updated BigQuery table schema: {table_id}")
                else:
                    logger.debug(f"Table {table_id} exists with correct schema")
                return

            # Create new table
            table = bigquery.Table(table_id, schema=expected_schema)
            table.clustering_fields = self.config.clustering_fields or ["dataset_name", "record_id"]
            table.description = f"Buttermilk Records table for dataset '{self.config.dataset_name}'"

            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"Created BigQuery table: {table_id}")

        except Exception as e:
            logger.error(f"Error creating/updating BigQuery table: {e}")
            raise StorageError(f"Failed to create/update table: {e}") from e

    def _build_select_query(self) -> str:
        """Build SQL query for selecting records."""
        base_query = f"""
        SELECT
            record_id,
            content,
            metadata,
            ground_truth,
            uri,
            mime
        FROM `{self.get_table_ref()}`
        WHERE dataset_name = @dataset_name
        """

        if self.config.split_type:
            base_query += " AND split_type = @split_type"

        # Apply additional filters
        for key, value in self.config.filter.items():
            if isinstance(value, str):
                base_query += f" AND {key} = '{value}'"
            else:
                base_query += f" AND {key} = {value}"

        # Ordering
        if self.config.randomize:
            base_query += " ORDER BY RAND()"
        else:
            base_query += " ORDER BY record_id"

        # Limit
        if self.config.limit:
            base_query += f" LIMIT {self.config.limit}"

        return base_query

    def _build_query_job_config(self) -> bigquery.QueryJobConfig:
        """Build BigQuery job configuration."""
        parameters = [
            bigquery.ScalarQueryParameter("dataset_name", "STRING", self.config.dataset_name),
        ]

        if self.config.split_type:
            parameters.append(
                bigquery.ScalarQueryParameter("split_type", "STRING", self.config.split_type)
            )

        return bigquery.QueryJobConfig(query_parameters=parameters)

    def _parse_record(self, row: bigquery.Row) -> Record:
        """Parse a BigQuery row into a Record object."""
        try:
            # Convert row to dictionary for easier column mapping
            row_dict = dict(row.items())
            
            # Apply column mapping if specified
            if self.config.columns:
                mapped_row = {}
                for new_name, old_name in self.config.columns.items():
                    if old_name in row_dict:
                        mapped_row[new_name] = row_dict[old_name]
                    elif hasattr(row, old_name):
                        mapped_row[new_name] = getattr(row, old_name)
                # Update row_dict with mapped values
                row_dict.update(mapped_row)
            
            # Parse JSON fields (handle both mapped and original names)
            metadata_field = row_dict.get('metadata', getattr(row, 'metadata', None))
            ground_truth_field = row_dict.get('ground_truth', getattr(row, 'ground_truth', None))
            
            metadata = json.loads(metadata_field) if metadata_field else {}
            ground_truth = json.loads(ground_truth_field) if ground_truth_field else None

            # Create Record object using mapped fields when available
            record = Record(
                record_id=row_dict.get('record_id', getattr(row, 'record_id', 'unknown')),
                content=row_dict.get('content', getattr(row, 'content', '')),
                metadata=metadata,
                ground_truth=ground_truth,
                uri=row_dict.get('uri', getattr(row, 'uri', None)),
                mime=row_dict.get('mime', getattr(row, 'mime', 'text/plain'))
            )

            return record

        except Exception as e:
            logger.warning(f"Error parsing BigQuery row {getattr(row, 'record_id', 'unknown')}: {e}")
            # Return a minimal record on parse error
            return Record(
                record_id=getattr(row, 'record_id', 'error'),
                content=str(getattr(row, 'content', 'Error loading content')),
                metadata={"parse_error": str(e)}
            )

    def _record_to_row(self, record: Record) -> dict[str, Any]:
        """Convert a Record object to a BigQuery row dict.
        
        Note: This returns raw Python objects. The upload_rows pipeline will handle
        proper JSON serialization and datetime formatting for BigQuery.
        """
        return {
            "record_id": record.record_id,
            "dataset_name": self.config.dataset_name,
            "split_type": self.config.split_type,
            "content": record.content,
            "metadata": record.metadata,
            "ground_truth": record.ground_truth,
            "uri": record.uri,
            "mime": record.mime,
            "created_at": datetime.datetime.now(datetime.UTC),
            "updated_at": datetime.datetime.now(datetime.UTC),
        }
