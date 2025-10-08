"""DuckDB storage implementation for unified storage operations."""

from typing import TYPE_CHECKING, Iterator

from buttermilk._core.exceptions import StorageError
from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord

from .base import Storage

if TYPE_CHECKING:
    from .._core.storage_config import DuckDBStorageConfig


class DuckDBStorage(Storage):
    """Unified DuckDB storage supporting both read and write operations.

    Supports querying DuckDB databases with table names or custom SQL queries.
    """

    def __init__(self, config: DuckDBStorageConfig):
        """Initialize DuckDB storage.

        Args:
            config: DuckDB storage configuration

        Raises:
            StorageError: If database path is missing or connection fails
        """
        super().__init__(config)

        if not config.database:
            raise StorageError("DuckDB storage requires a database path")

        self.database = config.database
        self.table_name = config.table_name
        self.schema_name = config.schema_name or "main"
        self.custom_query = config.custom_query
        self.read_only = config.read_only

        # Validate configuration
        if not self.table_name and not self.custom_query:
            raise StorageError("DuckDB storage requires either table_name or custom_query")

        self._conn = None

    def _get_connection(self):
        """Get or create DuckDB connection."""
        if self._conn is None:
            try:
                import duckdb

                self._conn = duckdb.connect(self.database, read_only=self.read_only)
                # Load JSON extension for handling JSON columns
                self._conn.execute("INSTALL json; LOAD json;")
                logger.debug(f"Connected to DuckDB database: {self.database}")
            except Exception as e:
                raise StorageError(f"Failed to connect to DuckDB database {self.database}: {e}") from e
        return self._conn

    def _build_query(self) -> str:
        """Build SQL query based on configuration."""
        if self.custom_query:
            return self.custom_query

        # Build query from table name
        full_table = f"{self.schema_name}.{self.table_name}" if self.schema_name else self.table_name
        query = f"SELECT * FROM {full_table}"

        # Add dataset/split filters if specified
        filters = []
        if self.config.dataset_name:
            filters.append(f"dataset_name = '{self.config.dataset_name}'")
        if self.config.split_type:
            filters.append(f"split_type = '{self.config.split_type}'")

        if filters:
            query += " WHERE " + " AND ".join(filters)

        return query

    def __iter__(self) -> Iterator[BaseRecord]:
        """Iterate over records from DuckDB.

        Yields:
            BaseRecord objects from the database query
        """
        try:
            conn = self._get_connection()
            query = self._build_query()

            logger.debug(f"Executing DuckDB query: {query}")
            result = conn.execute(query)

            # Get column names
            columns = [desc[0] for desc in result.description] if result.description else []

            # Iterate over rows
            for row in result.fetchall():
                row_dict = dict(zip(columns, row))

                # Create record using configured record class
                try:
                    record = self._create_record(**row_dict)
                    yield record
                except Exception as e:
                    logger.warning(f"Failed to create record from row: {e}. Row data: {row_dict}")
                    continue

        except Exception as e:
            logger.error(f"Error iterating DuckDB storage: {e}")
            raise StorageError(f"Failed to iterate DuckDB storage: {e}") from e

    def save(self, records: list[BaseRecord] | BaseRecord | list[dict] | dict) -> None:
        """Save records to DuckDB.

        Args:
            records: Single BaseRecord or list of BaseRecord objects (or dicts)

        Raises:
            StorageError: If save operation fails
        """
        if not self.table_name:
            raise StorageError("Cannot save to DuckDB without table_name (custom_query not supported for writes)")

        # Normalize to list
        if not isinstance(records, list):
            records = [records]

        if not records:
            return

        try:
            conn = self._get_connection()

            # Convert records to dicts if they're BaseRecord objects
            rows = []
            for record in records:
                if isinstance(record, BaseRecord):
                    rows.append(record.model_dump())
                elif isinstance(record, dict):
                    rows.append(record)
                else:
                    raise ValueError(f"Expected BaseRecord or dict, got {type(record)}")

            if not rows:
                return

            # Build INSERT statement
            full_table = f"{self.schema_name}.{self.table_name}" if self.schema_name else self.table_name
            columns = list(rows[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            column_names = ", ".join(columns)

            insert_sql = f"INSERT INTO {full_table} ({column_names}) VALUES ({placeholders})"

            # Insert all rows
            for row in rows:
                values = [row.get(col) for col in columns]
                conn.execute(insert_sql, values)

            logger.info(f"Saved {len(rows)} records to DuckDB table {full_table}")

        except Exception as e:
            logger.error(f"Error saving to DuckDB: {e}")
            raise StorageError(f"Failed to save records to DuckDB: {e}") from e

    def count(self) -> int:
        """Count total records in storage.

        Returns:
            Number of records, or -1 if count fails
        """
        try:
            conn = self._get_connection()

            if self.custom_query:
                # Wrap custom query in COUNT
                count_query = f"SELECT COUNT(*) FROM ({self.custom_query}) AS subquery"
            else:
                full_table = f"{self.schema_name}.{self.table_name}" if self.schema_name else self.table_name
                count_query = f"SELECT COUNT(*) FROM {full_table}"

                # Add filters
                filters = []
                if self.config.dataset_name:
                    filters.append(f"dataset_name = '{self.config.dataset_name}'")
                if self.config.split_type:
                    filters.append(f"split_type = '{self.config.split_type}'")

                if filters:
                    count_query += " WHERE " + " AND ".join(filters)

            result = conn.execute(count_query).fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.warning(f"Failed to count records in DuckDB: {e}")
            return -1

    def exists(self) -> bool:
        """Check if database file exists.

        Returns:
            True if database file exists
        """
        from pathlib import Path
        return Path(self.database).exists()

    def close(self):
        """Close DuckDB connection."""
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
            self._conn = None
            logger.debug(f"Closed DuckDB connection to {self.database}")

    def __del__(self):
        """Cleanup connection on deletion."""
        # Only close if object was fully initialized
        if hasattr(self, '_conn'):
            self.close()
