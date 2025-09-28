"""Base storage classes for unified storage operations."""

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Iterator, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

from buttermilk._core.constants import BQ_SCHEMA_DIR
from buttermilk._core.exceptions import FatalError
from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord, Record

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM

    from .._core.storage_config import StorageConfig

# Generic type for any Pydantic model
T = TypeVar("T", bound=BaseModel)


class RecordFilter(Protocol):
    """Protocol for custom record filtering/sampling strategies.

    Implementations can provide any sampling logic - random sampling,
    field-based filtering, complex business rules, etc.
    """

    async def should_include(self, record: BaseRecord) -> bool:
        """Determine if a record should be included in the results.

        Args:
            record: The record to evaluate

        Returns:
            True if the record should be included, False to skip it
        """
        ...


class Storage(ABC):
    """Base class for unified storage operations (read and write).

    This abstract base class defines the interface for storage backends
    that support both reading and writing operations with the same configuration.
    """

    def __init__(self, config: "StorageConfig", bm: "BM | None" = None):
        """Initialize storage with configuration and BM instance.

        Args:
            config: Storage configuration
            bm: Buttermilk instance for accessing clients and defaults
        """
        self.config = config
        self.bm = bm
        self._record_class: Type[BaseRecord] | None = None
        self._async_iterator: Optional[AsyncGenerator[dict[str, Any], None]] = None

    @abstractmethod
    def __iter__(self) -> Iterator[BaseRecord]:
        """Iterate over items from storage.

        Returns:
            Iterator yielding BaseRecord objects (or subclasses like Record, Title, etc.)
        """
        pass

    @abstractmethod
    def save(self, records: list[BaseModel] | BaseModel) -> None:
        """Save Pydantic models to storage.

        Args:
            records: Single Pydantic model or list of models to save
        """
        pass

    def get_record_by_id(self, record_id: str) -> BaseRecord | None:
        """Get a single record by its ID.

        Default implementation: iterates through all records (assumes small datasets).
        Storage backends optimized for large datasets should override this method.

        Args:
            record_id: The unique identifier of the record to retrieve

        Returns:
            The record if found, None otherwise
        """
        for record in self:
            if hasattr(record, "record_id") and record.record_id == record_id:
                return record
        return None

    @abstractmethod
    def count(self) -> int:
        """Count total records in storage.

        Returns:
            Number of records, or -1 if unknown
        """
        pass

    def exists(self) -> bool:
        """Check if storage location exists.

        Returns:
            True if storage location exists
        """
        return True

    def create(self) -> None:
        """Create storage location if it doesn't exist.

        This is a no-op by default. Subclasses should override
        if they support creating storage locations.
        """
        pass

    def __len__(self) -> int:
        """Return number of records if known, 0 if streaming/unknown."""
        try:
            return self.count()
        except Exception:
            return 0

    async def iterate_async(
        self,
        batch_size: Optional[int] = None,
        filter: Optional[RecordFilter] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields record dictionaries from storage with optional filtering.

        This method enables Storage objects to be used directly as DataSource
        in simple pipelines by implementing the async generator protocol.

        Args:
            batch_size: Maximum number of records to yield (None = unlimited)
            filter: Optional filter to apply to records before yielding

        Yields:
            Dictionaries containing records that pass the filter (if provided)
            Format: {"record": BaseRecord}
        """
        count = 0
        for record in self:
            # Apply filter if provided
            if filter and not await filter.should_include(record):
                continue

            if batch_size is not None and count >= batch_size:
                break

            yield {"record": record}
            count += 1

    def __call__(
        self,
        batch_size: Optional[int] = None,
        filter: Optional[RecordFilter] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Make Storage objects callable as DataSource for pipelines.

        This allows Storage objects to be used directly in simple pipelines:
        ```python
        storage = bm.get_storage(config)

        # Without filter
        await run_simple_pipeline(storage(batch_size=100), processors)

        # With filter
        from buttermilk.storage.filters import YearRangeFilter
        filter = YearRangeFilter(2020, 2023)
        await run_simple_pipeline(storage(batch_size=100, filter=filter), processors)
        ```

        Args:
            batch_size: Maximum number of records to yield (None = unlimited)
            filter: Optional filter to apply to records

        Returns:
            Async generator of record dictionaries
        """
        return self.iterate_async(batch_size, filter)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Make Storage objects async iterable.

        Returns:
            Self as async iterator
        """
        # Reset/initialize the async iterator
        self._async_iterator = self.iterate_async()
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Get next item from async iterator.

        Returns:
            Next record dictionary from storage

        Raises:
            StopAsyncIteration: When no more records available
        """
        if self._async_iterator is None:
            raise StopAsyncIteration

        try:
            return await self._async_iterator.__anext__()
        except StopAsyncIteration:
            self._async_iterator = None
            raise

    def _get_record_class(self) -> Type[BaseRecord]:
        """Get the record class to use for instantiation.

        Resolves the class from the config's record_class field, with caching.
        Falls back to Record if not specified or on error.

        Returns:
            The class to use for creating record instances
        """
        # Return cached class if available
        if self._record_class is not None:
            return self._record_class

        # Default to Record class
        if not self.config.record_class:
            self._record_class = Record
            return self._record_class

        try:
            # Parse the class path
            module_path, class_name = self.config.record_class.rsplit(".", 1)

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            cls = getattr(module, class_name)

            # Verify it's a BaseRecord subclass
            if not issubclass(cls, BaseRecord):
                logger.warning(
                    f"Configured record_class '{self.config.record_class}' is not a BaseRecord subclass. "
                    f"Falling back to Record."
                )
                self._record_class = Record
            else:
                self._record_class = cls
                logger.debug(f"Using record class: {self.config.record_class}")

        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(
                f"Failed to import record_class '{self.config.record_class}': {e}. "
                f"Falling back to Record."
            )
            self._record_class = Record

        return self._record_class

    def _create_record(self, **kwargs: Any) -> BaseRecord:
        """Create a record instance using the configured class.

        Args:
            **kwargs: Fields to pass to the record constructor

        Returns:
            A new record instance of the configured type

        Raises:
            Exception: If record creation fails (will be caught by storage implementations)
        """
        record_class = self._get_record_class()
        return record_class(**kwargs)


class StorageClient:
    """Base utility class for managing storage clients and connections.

    This class provides common functionality for accessing cloud clients,
    schema handling, and configuration management.
    """

    def __init__(self, config: "StorageConfig", bm: "BM | None" = None):
        """Initialize storage client.

        Args:
            config: Storage configuration
            bm: Buttermilk instance for accessing clients
        """
        self.config = config
        self.bm = bm
        self._schema_cache = None

    def get_bq_client(self):
        """Get BigQuery client from BM instance."""
        if not self.bm:
            raise ValueError("BM instance required for BigQuery operations")
        return self.bm.bq

    def get_gcs_client(self):
        """Get Google Cloud Storage client from BM instance."""
        if not self.bm:
            raise ValueError("BM instance required for GCS operations")
        return self.bm.gcs

    def get_schema(self):
        """Load and cache schema from configuration."""
        if self._schema_cache is None and self.config.schema_path:
            try:
                bq_client = self.get_bq_client()
                if not Path(self.config.schema_path).exists():
                    if (BQ_SCHEMA_DIR / self.config.schema_path).exists():
                        self.config.schema_path = str(BQ_SCHEMA_DIR / self.config.schema_path)
                    else:
                        raise FatalError(f"Schema file not found: {self.config.schema_path}")
                self._schema_cache = bq_client.schema_from_json(self.config.schema_path)
            except Exception as e:
                self._schema_cache = None
                raise FatalError(f"Failed to load schema from {self.config.schema_path}: {e}") from e
        return self._schema_cache

    def get_table_ref(self) -> str:
        """Get full table reference for BigQuery operations.

        Returns the computed full_table_id from constituent parts.

        Returns:
            Full table reference in format 'project.dataset.table'

        Raises:
            ValueError: If any of project_id, dataset_id, or table_id is missing
        """
        if not self.config.full_table_id:
            missing_parts = []
            if not self.config.project_id:
                missing_parts.append("project_id")
            if not self.config.dataset_id:
                missing_parts.append("dataset_id")
            if not self.config.table_id:
                missing_parts.append("table_id")
            raise ValueError(f"Missing required fields for BigQuery operations: {', '.join(missing_parts)}")
        return self.config.full_table_id

