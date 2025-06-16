"""Storage configuration classes for unified storage operations."""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from buttermilk._core.log import logger


class AdditionalFieldConfig(BaseModel):
    """Configuration for additional fields to embed in multi-field embedding."""
    
    source_field: str = Field(
        description="Name of the field in Record.metadata to embed"
    )
    chunk_type: str = Field(
        description="Type tag for this chunk (used for filtering searches)"
    )
    min_length: int = Field(
        default=10,
        description="Minimum character length required to embed this field"
    )


class MultiFieldEmbeddingConfig(BaseModel):
    """Configuration for embedding multiple fields from records."""
    
    content_field: str = Field(
        default="content",
        description="Main content field to chunk and embed (from Record.content)"
    )
    additional_fields: list[AdditionalFieldConfig] = Field(
        default_factory=list,
        description="Additional fields from Record.metadata to embed as single chunks"
    )
    chunk_size: int = Field(
        default=2000,
        description="Chunk size for main content field"
    )
    chunk_overlap: int = Field(
        default=500,
        description="Chunk overlap for main content field"
    )


class StorageConfig(BaseModel):
    """Unified configuration for storage operations (read and write).

    This class consolidates configuration for both data loading and saving operations,
    replacing the separate DataSourceConfig and SaveInfo classes with a single,
    consistent configuration approach.

    
    Attributes:
        type: The storage backend type
        project_id: Cloud project ID (auto-detected from environment if not provided)
        dataset_id: Dataset/database identifier
        table_id: Table/collection identifier
        path: File path or URI for file-based storage
        schema_path: Path to schema definition file
        dataset_name: Logical dataset name for filtering/grouping
        split_type: Data split type (train/test/val)
        randomize: Whether to randomize query results
        batch_size: Batch size for operations
        auto_create: Whether to auto-create storage if it doesn't exist
        clustering_fields: Fields to use for clustering (BigQuery)
        db: Database-specific configuration parameters
        filter: Filtering criteria for data operations
        columns: Column selection/transformation mapping
        limit: Maximum number of records to process
    """

    # Core identification
    type: Literal[
        "bigquery", "file", "gcs", "s3", "local",
        "chromadb", "vector", "plaintext", "generator",
        "huggingface", "job", "outputs"
    ] = Field(description="Storage backend type")

    # Cloud/database configuration
    project_id: str | None = Field(
        default=None,
        description="Cloud project ID (auto-detected from GOOGLE_CLOUD_PROJECT if not provided)"
    )
    dataset_id: str | None = Field(
        default=None,
        description="Dataset/database identifier"
    )
    table_id: str | None = Field(
        default=None,
        description="Table/collection identifier"
    )

    # Path-based storage
    path: str | None = Field(
        default=None,
        description="File path or URI for storage location"
    )

    # Schema and structure
    schema_path: str | None = Field(
        default=None,
        description="Path to schema definition file"
    )

    # Data organization
    dataset_name: str | None = Field(
        default=None,
        description="Logical dataset name for filtering/grouping"
    )
    split_type: str | None = Field(
        default=None,
        description="Data split type (train/test/val)"
    )

    # Operation configuration
    randomize: bool = Field(
        default=True,
        description="Whether to randomize query results"
    )
    batch_size: int = Field(
        default=1000,
        ge=1,
        description="Batch size for operations"
    )
    auto_create: bool = Field(
        default=True,
        description="Whether to auto-create storage if it doesn't exist"
    )
    clustering_fields: list[str] = Field(
        default=["record_id", "dataset_name"],
        description="Fields to use for clustering (BigQuery)"
    )

    # Provider-specific configuration
    db: dict[str, Any] = Field(
        default_factory=dict,
        description="Database-specific configuration parameters"
    )

    # Data filtering and selection
    filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtering criteria for data operations"
    )
    columns: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Column mapping for renaming data source fields to Record fields. "
            "Dictionary where keys are target Record field names and values are source field names. "
            "Can be empty ({}) if no field renaming is needed. "
            "Example: {'content': 'text', 'ground_truth': 'expected'}"
        )
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of records to process"
    )

    # Additional fields from DataSourceConfig for compatibility
    max_records_per_group: int = Field(
        default=-1,
        description="Maximum records to process per group if grouping is applied. -1 for no limit."
    )
    glob: str = Field(
        default="**/*",
        description="Glob pattern for matching files if type is 'file'."
    )
    join: dict[str, str] = Field(
        default_factory=dict,
        description="Configuration for joining with other data sources."
    )
    agg: bool = Field(
        default=False,
        description="Whether to aggregate results."
    )
    group: dict[str, str] = Field(
        default_factory=dict,
        description="Grouping configuration (new_group_col: original_col_or_expr)."
    )
    last_n_days: int = Field(
        default=7,
        description="For time-series data, retrieve from the last N days."
    )
    name: str = Field(
        default="",
        description="Name identifier for the storage configuration."
    )
    split: str = Field(
        default="train",
        description="Data split identifier (train/test/val)."
    )

    # File-based storage specific
    index: list[str] | None = Field(
        default=None,
        description="Columns to use as an index"
    )

    # ChromaDB/Vector storage specific
    persist_directory: str | None = Field(
        default=None,
        description="Directory for persisting data (for ChromaDB or file-based vector stores)"
    )
    collection_name: str | None = Field(
        default=None,
        description="Name of the collection for ChromaDB"
    )
    embedding_model: str | None = Field(
        default=None,
        description="Name or path of embedding model (for ChromaDB/vector search)"
    )
    dimensionality: int | None = Field(
        default=None,
        description="Dimensionality of embeddings, if applicable"
    )
    
    # Multi-field embedding configuration
    multi_field_embedding: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Configuration for embedding multiple fields from records. "
            "Format: {'content_field': 'content', 'additional_fields': [{'source_field': 'summary', 'chunk_type': 'summary', 'min_length': 50}]}"
        )
    )

    # Generic URI field for various storage types
    uri: str | None = Field(
        default=None,
        description="URI for data source (alternative to path for some storage types)"
    )

    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def set_project_id_from_env(self) -> "StorageConfig":
        """Set project_id from environment if not already set."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self

    @computed_field
    @property
    def full_table_id(self) -> str | None:
        """Compute full BigQuery table identifier from constituent parts.

        The full table ID is always computed from project_id, dataset_id, and table_id.
        This ensures consistency and prevents conflicts between direct full_table_id
        specification and constituent parts.

        Returns:
            Fully qualified table ID in format 'project.dataset.table' or None if any part is missing
        """
        if all([self.project_id, self.dataset_id, self.table_id]):
            return f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        return None

    def merge_defaults(self, defaults: "StorageConfig") -> "StorageConfig":
        """Merge this config with default values, prioritizing this config's values."""
        exclude_fields = {"full_table_id"}
        merged_data = defaults.model_dump(exclude=exclude_fields)
        merged_data.update(self.model_dump(exclude=exclude_fields))
        return StorageConfig(**merged_data)


class BigQueryDefaults(BaseModel):
    """Default configuration values specifically for BigQuery operations."""

    dataset_id: str | None = Field(default=None)
    table_id: str | None = Field(default=None)
    randomize: bool = Field(default=True)
    batch_size: int = Field(default=1000)
    auto_create: bool = Field(default=True)
    clustering_fields: list[str] = Field(default=["record_id", "dataset_name"])

    def to_storage_config(self) -> StorageConfig:
        """Convert to a StorageConfig object."""
        return StorageConfig(
            type="bigquery",
            **self.model_dump()
        )


class StorageFactory:
    """Factory for creating storage instances based on configuration."""
    
    @staticmethod
    def create_storage(config: StorageConfig, bm_instance=None):
        """Create storage instance based on configuration type.
        
        Args:
            config: StorageConfig instance (from OmegaConf/Hydra)
            bm_instance: BM instance for context (optional)
            
        Returns:
            Storage instance appropriate for the config type
        """
        from buttermilk.data.vector import ChromaDBEmbeddings
        
        # Ensure we have a proper StorageConfig instance
        if not isinstance(config, StorageConfig):
            raise ValueError(f"Expected StorageConfig instance, got {type(config)}")
        
        storage_type = config.type
        
        if storage_type in ["bigquery", "bq"]:
            from buttermilk.storage.bigquery import BigQueryStorage
            return BigQueryStorage(config, bm_instance)
        elif storage_type in ["file", "local", "gcs", "s3"]:
            from buttermilk.storage.file import FileStorage
            return FileStorage(config, bm_instance)
        elif storage_type == "chromadb":
            # Convert StorageConfig to ChromaDBEmbeddings parameters
            chromadb_params = {
                'collection_name': config.collection_name or 'default_collection',
                'persist_directory': config.persist_directory or './data/chromadb',
                'embedding_model': config.embedding_model or 'gemini-embedding-001',
                'dimensionality': config.dimensionality or 3072,
            }
            
            # Add multi-field embedding configuration if specified
            if config.multi_field_embedding:
                try:
                    # Parse multi-field config into proper Pydantic model
                    multi_field_config = MultiFieldEmbeddingConfig(**config.multi_field_embedding)
                    chromadb_params['multi_field_config'] = multi_field_config
                except Exception as e:
                    logger.warning(f"Invalid multi_field_embedding config, using default: {e}")
            
            # Add other ChromaDB-specific fields if present in config
            for field in ['concurrency', 'upsert_batch_size', 'embedding_batch_size', 'arrow_save_dir']:
                if hasattr(config, field) and getattr(config, field) is not None:
                    chromadb_params[field] = getattr(config, field)
            
            return ChromaDBEmbeddings(**chromadb_params)
        elif storage_type == "huggingface":
            from buttermilk.storage.huggingface import HuggingFaceStorage
            return HuggingFaceStorage(config, bm_instance)
        elif storage_type == "plaintext":
            # Use FileStorage with plaintext-specific configuration
            from buttermilk.storage.file import FileStorage
            # For plaintext, we typically use glob patterns
            if not config.glob or config.glob == "**/*":
                # Set default glob for text files
                config.glob = "**/*.txt"
            return FileStorage(config, bm_instance)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

