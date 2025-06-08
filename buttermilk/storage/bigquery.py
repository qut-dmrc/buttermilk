"""BigQuery storage implementation for unified storage operations."""

import datetime
import json
from typing import TYPE_CHECKING, Iterator, Any

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
            
            # Convert records to BigQuery rows
            rows_to_insert = []
            for record in records:
                row = self._record_to_row(record)
                rows_to_insert.append(row)
            
            # Insert rows
            table = self.table
            errors = self.client.insert_rows_json(table, rows_to_insert)
            
            if errors:
                logger.error(f"Error inserting records: {errors}")
                raise StorageError(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Successfully saved {len(records)} records to {self.get_table_ref()}")
                
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
        """Create the BigQuery table if it doesn't exist."""
        if self.exists():
            return
            
        try:
            table_id = self.get_table_ref()
            
            # Define schema for Records table
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
            
            # Use custom schema if provided
            if self.get_schema():
                schema = self.get_schema()
            
            # Create table with clustering
            table = bigquery.Table(table_id, schema=schema)
            table.clustering_fields = self.config.clustering_fields
            table.description = f"Buttermilk Records table for dataset '{self.config.dataset_name}'"
            
            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"Created/verified BigQuery table: {table_id}")
            
        except Exception as e:
            logger.error(f"Error creating BigQuery table: {e}")
            raise StorageError(f"Failed to create table: {e}") from e
    
    def _build_select_query(self) -> str:
        """Build SQL query for selecting records."""
        base_query = f"""
        SELECT 
            record_id,
            content,
            metadata,
            alt_text,
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
    
    def _record_to_row(self, record: Record) -> dict[str, Any]:
        """Convert a Record object to a BigQuery row dict."""
        return {
            "record_id": record.record_id,
            "dataset_name": self.config.dataset_name,
            "split_type": self.config.split_type,
            "content": record.content if isinstance(record.content, str) else json.dumps(record.content),
            "metadata": json.dumps(record.metadata),
            "alt_text": record.alt_text,
            "ground_truth": json.dumps(record.ground_truth) if record.ground_truth else None,
            "uri": record.uri,
            "mime": record.mime,
            "created_at": datetime.datetime.now(datetime.UTC),
            "updated_at": datetime.datetime.now(datetime.UTC),
        }