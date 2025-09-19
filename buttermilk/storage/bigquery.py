"""BigQuery storage implementation for unified storage operations."""

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypeVar

from google.cloud import bigquery
from pydantic import BaseModel

from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk.utils.save import upload_rows
from buttermilk.utils.utils import unwrap_numpy_arrow_types

from .base import Storage, StorageClient, StorageError

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM

    from .._core.storage_config import StorageConfig

# Generic type for any Pydantic model
T = TypeVar("T", bound=BaseModel)


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

        Raises:
            StorageError: If schema_path is missing or schema cannot be loaded

        """
        super().__init__(config, bm)
        StorageClient.__init__(self, config, bm)

        if config.type != "bigquery":
            raise ValueError(f"BigQueryStorage requires type='bigquery', got '{config.type}'")

        if not config.dataset_name and not config.dataset_id:
            raise ValueError("BigQuery storage requires either dataset_name or dataset_id")

        # CRITICAL: Require explicit schema - no implicit defaults allowed
        if not config.schema_path:
            raise StorageError(
                "BigQuery storage requires explicit schema_path in the configuration.",
            )

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
        self._schema_validated = False

    def _validate_schema(self) -> None:
        """Validate that schema can be loaded.

        Called lazily on first use to avoid requiring BM instance during init.
        """
        if not self._schema_validated:
            schema = self.get_schema()
            if not schema:
                raise StorageError(
                    f"Failed to load schema from {self.config.schema_path}. BigQuery storage requires a valid schema file.",
                )
            self._schema_validated = True

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

    def save(self, records: list[BaseModel | dict[str, Any]] | BaseModel | dict[str, Any]) -> None:
        """Save Pydantic models to BigQuery table.

        Uses the existing upload_rows pipeline for proper serialization and error handling.

        Args:
            records: Single Pydantic model, dict, or list of models/dicts to save

        Raises:
            StorageError: If save operation fails

        """
        # Validate schema on first use
        self._validate_schema()

        if isinstance(records, BaseModel):
            records = [records]

        if not records:
            logger.warning("No records to save")
            return

        try:
            # Ensure table exists
            if self.config.auto_create:
                self.create()

            # Convert Pydantic models to list of dicts for upload_rows
            rows_to_insert = []
            for record in records:
                if isinstance(record, dict):
                    row = record
                else:
                    # Assume it's a Pydantic model with model_dump
                    row = record.model_dump(mode="json")
                rows_to_insert.append(row)

            # Get the schema for proper data transformation
            schema = self.get_schema()
            if not schema:
                raise StorageError(
                    "Schema is required for BigQuery operations. Please ensure that a valid schema file is provided in the configuration.",
                )

            # Use the proven upload_rows pipeline
            result = upload_rows(
                rows=rows_to_insert,
                schema=schema,
                dataset=self.get_table_ref(),
            )

            if result:
                logger.debug(f"Successfully saved {len(records)} records to {self.get_table_ref()}")
            else:
                raise StorageError("Upload failed - no result returned from upload_rows")

        except Exception as e:
            logger.error(f"Error saving records to BigQuery: {e}")
            raise StorageError(f"Failed to save to BigQuery: {e}") from e

    def get_record_by_id(self, record_id: str) -> Record | None:
        """Get a single record by ID using a parameterized BigQuery query.

        Args:
            record_id: The unique identifier of the record to retrieve.

        Returns:
            Record if found, otherwise None.

        Raises:
            StorageError: If query fails or essential columns are missing.
        """
        if not record_id:
            raise ValueError("record_id must be a non-empty string")

        # Resolve columns and check availability
        record_col = self._resolve_column("record_id")
        dataset_col = self._resolve_column("dataset_name")
        split_col = self._resolve_column("split_type")
        available_cols = self._available_columns()

        if available_cols and record_col not in available_cols:
            raise StorageError(f"BigQuery table {self.get_table_ref()} has no column '{record_col}' required for record lookup.")

        # Compose query
        where_parts = [f"{record_col} = @record_id"]
        use_dataset = (dataset_col in available_cols) or (not available_cols)
        use_split = bool(self.config.split_type) and ((split_col in available_cols) or (not available_cols))

        if use_dataset:
            where_parts.append(f"{dataset_col} = @dataset_name")
        else:
            logger.warning(f"BigQuery table {self.get_table_ref()} has no column '{dataset_col}'. Skipping dataset filter in get_record_by_id().")

        if use_split:
            where_parts.append(f"{split_col} = @split_type")
        elif self.config.split_type and available_cols:
            logger.warning(f"BigQuery table {self.get_table_ref()} has no column '{split_col}'. Skipping split filter in get_record_by_id().")

        query = f"""
        SELECT *
        FROM `{self.get_table_ref()}`
        WHERE {" AND ".join(where_parts)}
        LIMIT 1
        """

        # Build parameters
        params: list[bigquery.ScalarQueryParameter] = [
            bigquery.ScalarQueryParameter("record_id", "STRING", record_id),
        ]
        if use_dataset:
            params.append(bigquery.ScalarQueryParameter("dataset_name", "STRING", self.config.dataset_name))
        if use_split:
            params.append(bigquery.ScalarQueryParameter("split_type", "STRING", self.config.split_type))

        job_config = bigquery.QueryJobConfig(query_parameters=params)

        try:
            query_job = self.client.query(query, job_config=job_config)
            for row in query_job:  # At most one due to LIMIT 1
                return self._parse_record(row)
            return None
        except Exception as e:
            logger.error(f"Error querying BigQuery for record_id {record_id}: {e}")
            raise StorageError(f"Failed to fetch record by id: {e}") from e

    def count(self) -> int:
        """Count total records matching the criteria.

        Returns:
            Number of records in the table

        """
        try:
            # Resolve logical->physical columns
            def _resolve_column(logical: str) -> str:
                return self.config.columns.get(logical, logical)

            dataset_col = _resolve_column("dataset_name")
            split_col = _resolve_column("split_type")

            try:
                available_cols = {field.name for field in self.table.schema}
            except Exception:
                available_cols = set()

            query = f"""
            SELECT COUNT(*) as total
            FROM `{self.get_table_ref()}`
            """

            where_clauses: list[str] = []
            if dataset_col in available_cols or not available_cols:
                where_clauses.append(f"{dataset_col} = @dataset_name")
            else:
                logger.warning(f"BigQuery table {self.get_table_ref()} has no column '{dataset_col}'. Skipping dataset filter in count().")

            if self.config.split_type:
                if split_col in available_cols or not available_cols:
                    where_clauses.append(f"{split_col} = @split_type")
                else:
                    logger.warning(f"BigQuery table {self.get_table_ref()} has no column '{split_col}'. Skipping split filter in count().")

            for key, value in self.config.filter.items():
                if isinstance(value, str):
                    where_clauses.append(f"{key} = '{value}'")
                else:
                    where_clauses.append(f"{key} = {value}")

            if where_clauses:
                query += "\nWHERE " + " AND ".join(where_clauses)

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
        """Create the BigQuery table if it doesn't exist.

        IMPORTANT: This method will NOT modify existing tables. If a table exists,
        it will log a message and return without making changes.

        Raises:
            StorageError: If table creation fails or schema is missing

        """
        # Validate schema on first use
        self._validate_schema()

        try:
            table_id = self.get_table_ref()

            # CRITICAL: Require explicit schema - no implicit defaults
            expected_schema = self.get_schema()
            if not expected_schema:
                raise StorageError(
                    "Schema is required for table creation. Configure schema_path in your storage config.",
                )

            if self.exists():
                # CRITICAL: Never modify existing tables
                logger.debug(f"Table {table_id} already exists. Skipping creation. BigQuery storage will not modify existing tables. If schema changes are needed, handle them manually.",
                )
                return

            # Create new table
            table = bigquery.Table(table_id, schema=expected_schema)
            table.clustering_fields = self.config.clustering_fields or ["dataset_name", "record_id"]
            table.description = f"Buttermilk table for dataset '{self.config.dataset_name}'"

            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"Created BigQuery table: {table_id}")

        except Exception as e:
            logger.error(f"Error creating/updating BigQuery table: {e}")
            raise StorageError(f"Failed to create/update table: {e}") from e

    def _build_select_query(self) -> str:
        """Build SQL query for selecting records."""
        order_col = self._resolve_column("record_id")
        available_cols = self._available_columns()

        query = f"""
        SELECT *
        FROM `{self.get_table_ref()}`
        """

        where_sql = self._compose_where_clause(available_cols)
        if where_sql:
            query += "\nWHERE " + where_sql + "\n"

        # Ordering
        if self.config.randomize:
            query += " ORDER BY RAND()"
        elif order_col in available_cols or not available_cols:
            query += f" ORDER BY {order_col}"
        else:
            logger.debug(f"BigQuery table {self.get_table_ref()} has no column '{order_col}'. Skipping ORDER BY.")

        # Limit
        if self.config.limit:
            query += f" LIMIT {self.config.limit}"

        return query

    def _resolve_column(self, logical: str) -> str:
        """Resolve logical Record field to physical column name using config.columns mapping."""
        return self.config.columns.get(logical, logical)

    def _available_columns(self) -> set[str]:
        """Return available BQ column names for the target table, or empty set if unknown."""
        try:
            return {field.name for field in self.table.schema}
        except Exception:
            return set()

    def _compose_where_clause(self, available_cols: set[str]) -> str:
        """Build WHERE clause (without the 'WHERE' keyword). Returns empty string if none.

        Avoids referencing columns that don't exist. Uses named parameters @dataset_name and @split_type.
        """
        clauses: list[str] = []

        dataset_col = self._resolve_column("dataset_name")
        split_col = self._resolve_column("split_type")

        # Dataset filter only if column exists
        if dataset_col in available_cols or not available_cols:
            clauses.append(f"{dataset_col} = @dataset_name")
        else:
            logger.warning(
                f"BigQuery table {self.get_table_ref()} has no column '{dataset_col}'. "
                "Skipping dataset filter; results may include multiple datasets. "
                "Configure columns mapping in storage config if your dataset column is named differently."
            )

        # Split filter only if requested and column exists
        if self.config.split_type and (split_col in available_cols or not available_cols):
            clauses.append(f"{split_col} = @split_type")
        elif self.config.split_type and available_cols:
            logger.warning(f"BigQuery table {self.get_table_ref()} has no column '{split_col}'. Skipping split filter.")

        # Additional literal filters (assumed to be physical column names)
        for key, value in self.config.filter.items():
            clauses.append(f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}")

        return " AND ".join(clauses)

    def _build_query_job_config(self) -> bigquery.QueryJobConfig:
        """Build BigQuery job configuration."""
        parameters = [
            bigquery.ScalarQueryParameter("dataset_name", "STRING", self.config.dataset_name),
        ]

        if self.config.split_type:
            parameters.append(
                bigquery.ScalarQueryParameter("split_type", "STRING", self.config.split_type),
            )

        return bigquery.QueryJobConfig(query_parameters=parameters)

    def _parse_record(self, row: bigquery.Row) -> Record:
        """Parse a BigQuery row into a Record object."""
        # Convert row to dictionary
        row_dict = dict(row.items())

        # Unwrap numpy/arrow types if present
        # This is necessary because BigQuery can return numpy/arrow types in the row data
        row_dict = unwrap_numpy_arrow_types(row_dict)

        # Apply column mapping if specified
        if self.config.columns:
            for new_name, old_name in self.config.columns.items():
                row_dict[new_name] = row_dict[old_name]

        # Parse JSON fields - metadata and ground_truth are stored as JSON strings in BigQuery
        metadata = json.loads(row_dict["metadata"]) if isinstance(row_dict["metadata"], str) else row_dict["metadata"]

        ground_truth = None
        if row_dict.get("ground_truth"):
            ground_truth = json.loads(row_dict["ground_truth"]) if isinstance(row_dict["ground_truth"], str) else row_dict["ground_truth"]

        # Create Record object from row data
        return Record(
            record_id=row_dict["record_id"],
            content=row_dict["content"],
            metadata=metadata,
            ground_truth=ground_truth,
            mime=row_dict.get("mime", "text/plain"),
        )
