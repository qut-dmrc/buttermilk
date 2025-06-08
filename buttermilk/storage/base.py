"""Base storage classes for unified storage operations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator, Any

from buttermilk._core.log import logger
from buttermilk._core.types import Record

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM
    from .config import StorageConfig


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
        
    @abstractmethod
    def __iter__(self) -> Iterator[Record]:
        """Iterate over records from storage.
        
        Returns:
            Iterator yielding Record objects
        """
        pass
    
    @abstractmethod
    def save(self, records: list[Record] | Record) -> None:
        """Save records to storage.
        
        Args:
            records: Single record or list of records to save
        """
        pass
    
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
                self._schema_cache = bq_client.schema_from_json(self.config.schema_path)
            except Exception as e:
                logger.warning(f"Failed to load schema from {self.config.schema_path}: {e}")
                self._schema_cache = None
        return self._schema_cache
    
    def get_table_ref(self) -> str:
        """Get full table reference for BigQuery operations."""
        if not self.config.full_table_id:
            raise ValueError("Missing project_id, dataset_id, or table_id for BigQuery operations")
        return self.config.full_table_id


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class StorageConfigError(StorageError):
    """Exception raised for storage configuration errors.""" 
    pass


class StorageConnectionError(StorageError):
    """Exception raised for storage connection errors."""
    pass