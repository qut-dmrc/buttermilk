"""BigQuery storage implementation for unified storage operations."""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import shortuuid
from google.cloud import bigquery
from pydantic import BaseModel

from buttermilk._core.exceptions import StorageError
from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord, Record
from buttermilk.utils import scrub_serializable
from buttermilk.utils.save import upload_rows
from buttermilk.utils.utils import unwrap_numpy_arrow_types

from .base import Storage, StorageClient

if TYPE_CHECKING:
    from .._core.storage_config import StorageConfig

# Generic type for any Pydantic model
T = TypeVar("T", bound=BaseModel)


class BigQueryStorage(Storage, StorageClient):
    """Unified BigQuery storage supporting both read and write operations.

    This class provides a single interface for BigQuery operations,
    replacing separate BigQueryRecordLoader and save functionality.
    """

    def __init__(self, config: "StorageConfig"):
        """Initialize BigQuery storage.

        Args:
            config: Storage configuration with BigQuery settings
            bm: Buttermilk instance for BigQuery client access

        Raises:
            StorageError: If schema_path is missing or schema cannot be loaded

        """
        super().__init__(config)
        StorageClient.__init__(self, config)

        if config.type != "bigquery":
            raise ValueError(f"BigQueryStorage requires type='bigquery', got '{config.type}'")

        if not config.dataset_name and not config.dataset_id:
            raise ValueError("BigQuery storage requires either dataset_name or dataset_id")

        # Store read_only flag
        self.read_only = config.read_only

        # CRITICAL: Require explicit schema - no implicit defaults allowed
        # Exception: read_only mode doesn't need schema since it won't write
        if not self.read_only and not config.schema_path:
            raise StorageError(
                "BigQuery storage requires explicit schema_path in the configuration. " "Use read_only=True if you only need to read data.",
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
        if self.read_only:
            return
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
            self._client = self.get_bq_client()
        return self._client

    @property
    def table(self) -> bigquery.Table:
        """Get BigQuery table reference."""
        if self._table is None:
            table_ref = self.get_table_ref()
            self._table = self.client.get_table(table_ref)
        return self._table

    def __iter__(self) -> Iterator[BaseRecord]:
        """Iterate over records from BigQuery table.

        Yields:
            BaseRecord objects from the table (Record, Title, or other subclasses)

        """
        try:
            # Use custom query if provided
            if self.config.custom_query:
                query = self.config.custom_query.replace("{table}", f"{self.get_table_ref()}")
                job_config = None  # Custom query handles its own parameters
            else:
                query = self._build_select_query()
                job_config = self._build_query_job_config()

            logger.info(f"Loading records from {self.get_table_ref()} for dataset '{self.config.dataset_name}'")

            query_job = self.client.query(query, job_config=job_config)

            for row in query_job:
                yield self._parse_record(row)

        except Exception as e:
            logger.error(f"Error loading records from BigQuery: {e}")
            raise StorageError(f"Failed to read from BigQuery: {e}") from e

    def save(
        self,
        records: list[BaseRecord] | BaseRecord | list[dict[str, Any]] | dict[str, Any],
    ) -> None:
        """Save records to BigQuery table.

        Uses the existing upload_rows pipeline for proper serialization and error handling.

        Args:
            records: Single BaseRecord/dict or list of BaseRecord/dicts to save

        Raises:
            StorageError: If save operation fails

        """
        # Validate schema on first use
        self._validate_schema()

        # Normalize to list without mutating variable type for type-checkers
        # Normalize to a sequence without tripping type invariance
        items: Sequence[Any] = records if isinstance(records, list) else [records]

        if not records:
            logger.warning("No records to save")
            return

        try:
            # Ensure table exists
            if self.config.auto_create:
                self.create()

            # Convert Pydantic models to list of dicts for upload_rows
            rows_to_insert: list[dict[str, Any]] = []
            for record in items:
                if isinstance(record, dict):
                    rows_to_insert.append(record)
                elif hasattr(record, "model_dump"):
                    rows_to_insert.append(scrub_serializable(record.model_dump()))  # type: ignore[attr-defined]
                else:
                    # Last resort
                    rows_to_insert.append({"value": str(record)})

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
                logger.debug(f"Successfully saved {len(rows_to_insert)} records to {self.get_table_ref()}")
            else:
                raise StorageError("Upload failed - no result returned from upload_rows")

        except Exception as e:
            logger.error(
                "Failed to save records to BigQuery",
                extra={
                    "table": self.get_table_ref(),
                    "dataset_name": self.config.dataset_name,
                    "num_records": len(rows_to_insert),
                    "error": str(e),
                },
                exc_info=True,
            )
            raise StorageError(f"Failed to save to BigQuery table {self.get_table_ref()}: {e}") from e

    def get_record_by_id(self, record_id: str) -> BaseRecord | None:
        """Get a single record by ID using a parameterized BigQuery query.

        Args:
            record_id: The unique identifier of the record to retrieve.

        Returns:
            BaseRecord if found, otherwise None.

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

        table_id = self.get_table_ref()

        try:
            # CRITICAL: Require explicit schema - no implicit defaults
            expected_schema = self.get_schema()
            if not expected_schema:
                raise StorageError(
                    "Schema is required for table creation. Configure schema_path in your storage config.",
                )

            if self.exists():
                return

            # Create new table
            table = bigquery.Table(table_id, schema=expected_schema)

            # Determine clustering fields based on what's actually in the schema
            schema_field_names = {field.name for field in expected_schema}

            # Use configured clustering fields if provided, otherwise determine based on schema
            if self.config.clustering_fields:
                # Validate configured clustering fields exist in schema
                valid_clustering_fields = [field for field in self.config.clustering_fields if field in schema_field_names]
                if valid_clustering_fields != self.config.clustering_fields:
                    invalid_fields = set(self.config.clustering_fields) - set(valid_clustering_fields)
                    logger.warning(
                        "Some clustering fields not found in schema",
                        extra={
                            "table": table_id,
                            "requested_fields": self.config.clustering_fields,
                            "valid_fields": valid_clustering_fields,
                            "invalid_fields": list(invalid_fields),
                        },
                    )
                table.clustering_fields = valid_clustering_fields if valid_clustering_fields else None
            else:
                # Auto-determine clustering fields based on common fields in schema
                default_clustering = []
                for field in ["dataset_name", "record_id"]:
                    resolved_field = self._resolve_column(field)
                    if resolved_field in schema_field_names:
                        default_clustering.append(resolved_field)

                table.clustering_fields = default_clustering if default_clustering else None

            table.description = f"Buttermilk table for dataset '{self.config.dataset_name}'"

            table = self.client.create_table(table, exists_ok=True)
            logger.info(
                "Created BigQuery table",
                extra={
                    "table": table_id,
                    "dataset_name": self.config.dataset_name,
                    "clustering_fields": table.clustering_fields,
                    "num_fields": len(expected_schema),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to create BigQuery table",
                extra={
                    "table": table_id,
                    "dataset_name": self.config.dataset_name,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise StorageError(f"Failed to create table {table_id}: {e}") from e

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
        Includes custom_where clause if provided.
        """
        clauses: list[str] = []

        dataset_col = self._resolve_column("dataset_name")
        split_col = self._resolve_column("split_type")

        # Dataset filter only if column exists
        if self.config.dataset_name and (dataset_col in available_cols or not available_cols):
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

        # Add custom WHERE clause if provided
        if self.config.custom_where:
            # Wrap custom clause in parentheses for safety
            clauses.append(f"({self.config.custom_where})")

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

    def _parse_record(self, row: bigquery.Row) -> BaseRecord:
        """Parse a BigQuery row into a BaseRecord object.

        Leverages Pydantic's validation to handle JSON parsing and defaults.
        """
        # Convert row to dictionary
        row_dict = dict(row.items())

        # Unwrap numpy/arrow types if present
        row_dict = unwrap_numpy_arrow_types(row_dict)

        # Apply column mapping if specified
        if self.config.columns:
            for logical, physical in self.config.columns.items():
                if physical in row_dict and physical != logical:
                    row_dict[logical] = row_dict.pop(physical)

        # Add config defaults if not present in row
        if "dataset_name" not in row_dict:
            row_dict["dataset_name"] = self.config.dataset_name
        if "split_type" not in row_dict:
            row_dict["split_type"] = self.config.split_type

        # Let Pydantic handle all validation, JSON parsing, and type conversion
        try:
            return self._create_record(**row_dict)
        except Exception as e:
            # If record creation fails, create minimal valid Record for debugging
            logger.warning(f"Failed to create record from row data: {e}")
            return Record(
                record_id=str(row_dict.get("record_id", shortuuid.uuid())),
                dataset_name=str(row_dict.get("dataset_name", self.config.dataset_name or "default")),
                split_type=str(row_dict.get("split_type", self.config.split_type or "default")),
                metadata={"parse_error": str(e), "raw_data": str(row_dict)[:1000]},
                content=f"Failed to parse record: {e}",
            )
