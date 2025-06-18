"""Storage configuration classes with type-specific schemas."""

import os
from typing import Any, Literal, Union, Annotated

from pydantic import BaseModel, Field, computed_field, model_validator, Discriminator

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


class BaseStorageConfig(BaseModel):
    """Base configuration for all storage operations.
    
    Contains common fields shared across all storage types.
    """

    # Core identification
    type: str = Field(description="Storage backend type")

    # Common fields across all storage types
    dataset_name: str | None = Field(
        default=None,
        description="Logical dataset name for filtering/grouping"
    )
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
    
    # Generic fields that some storage types may use
    name: str = Field(
        default="",
        description="Name identifier for the storage configuration."
    )
    schema_path: str | None = Field(
        default=None,
        description="Path to schema definition file"
    )
    uri: str | None = Field(
        default=None,
        description="URI for data source (alternative to path for some storage types)"
    )
    
    # Provider-specific configuration
    db: dict[str, Any] = Field(
        default_factory=dict,
        description="Database-specific configuration parameters"
    )

    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }
    
    def merge_defaults(self, defaults):
        """Merge this config with default values, prioritizing this config's values."""
        exclude_fields = set()
        merged_data = defaults.model_dump(exclude=exclude_fields)
        merged_data.update(self.model_dump(exclude=exclude_fields))
        # Return the same type as self
        return self.__class__(**merged_data)


# Type-specific storage configuration classes

class BigQueryStorageConfig(BaseStorageConfig):
    """Configuration for BigQuery storage operations."""
    
    type: Literal["bigquery"] = Field(default="bigquery", description="Storage backend type")
    
    # BigQuery-specific fields
    project_id: str | None = Field(
        default=None,
        description="Cloud project ID (auto-detected from GOOGLE_CLOUD_PROJECT if not provided)"
    )
    dataset_id: str | None = Field(
        default=None,
        description="Dataset identifier"
    )
    table_id: str | None = Field(
        default=None,
        description="Table identifier"
    )
    clustering_fields: list[str] = Field(
        default=["record_id", "dataset_name"],
        description="Fields to use for clustering"
    )
    
    # Data organization specific to BigQuery
    max_records_per_group: int = Field(
        default=-1,
        description="Maximum records to process per group. -1 for no limit."
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
    
    @model_validator(mode="after")
    def set_project_id_from_env(self) -> "BigQueryStorageConfig":
        """Set project_id from environment if not already set."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self

    @computed_field
    @property
    def full_table_id(self) -> str | None:
        """Compute full BigQuery table identifier from constituent parts."""
        if all([self.project_id, self.dataset_id, self.table_id]):
            return f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        return None


class FileStorageConfig(BaseStorageConfig):
    """Configuration for file-based storage operations."""
    
    type: Literal["file", "local", "gcs", "s3", "plaintext"] = Field(description="Storage backend type")
    
    # File-specific fields  
    path: str | None = Field(
        default=None,
        description="File path or URI for storage location"
    )
    glob: str = Field(
        default="**/*",
        description="Glob pattern for matching files."
    )
    max_records_per_group: int = Field(
        default=-1,
        description="Maximum records to process per group. -1 for no limit."
    )
    index: list[str] | None = Field(
        default=None,
        description="Columns to use as an index"
    )
    

class VectorStorageConfig(BaseStorageConfig):
    """Configuration for vector database storage operations."""
    
    type: Literal["chromadb", "vector"] = Field(description="Storage backend type")
    
    # Vector storage specific fields
    persist_directory: str | None = Field(
        default=None,
        description="Directory for persisting vector data"
    )
    collection_name: str | None = Field(
        default=None,
        description="Name of the collection"
    )
    embedding_model: str | None = Field(
        default=None,
        description="Name or path of embedding model"
    )
    dimensionality: int | None = Field(
        default=None,
        description="Dimensionality of embeddings"
    )
    
    # Multi-field embedding configuration
    multi_field_embedding: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Configuration for embedding multiple fields from records. "
            "Format: {'content_field': 'content', 'additional_fields': [{'source_field': 'summary', 'chunk_type': 'summary', 'min_length': 50}]}"
        )
    )


class HuggingFaceStorageConfig(BaseStorageConfig):
    """Configuration for HuggingFace dataset storage operations."""
    
    type: Literal["huggingface"] = Field(default="huggingface", description="Storage backend type")
    
    # HuggingFace specific fields
    dataset_id: str | None = Field(
        default=None,
        description="HuggingFace dataset identifier"
    )
    split: str = Field(
        default="train",
        description="Data split identifier (train/test/val)."
    )
    

class GeneratorStorageConfig(BaseStorageConfig):
    """Configuration for generator-based storage operations."""
    
    type: Literal["generator", "job", "outputs"] = Field(description="Storage backend type")


# Discriminated union for all storage config types
StorageConfig = Annotated[
    Union[
        BigQueryStorageConfig,
        FileStorageConfig, 
        VectorStorageConfig,
        HuggingFaceStorageConfig,
        GeneratorStorageConfig
    ],
    Field(discriminator="type")
]


# Legacy compatibility - keep BigQueryDefaults for existing code
class BigQueryDefaults(BaseModel):
    """Default configuration values specifically for BigQuery operations."""

    dataset_id: str | None = Field(default=None)
    table_id: str | None = Field(default=None)
    randomize: bool = Field(default=True)
    batch_size: int = Field(default=1000)
    auto_create: bool = Field(default=True)
    clustering_fields: list[str] = Field(default=["record_id", "dataset_name"])

    def to_storage_config(self) -> BigQueryStorageConfig:
        """Convert to a BigQueryStorageConfig object."""
        return BigQueryStorageConfig(
            **self.model_dump()
        )


class StorageFactory:
    """Factory for creating storage instances based on configuration."""
    
    @staticmethod
    def create_config(config_dict: dict) -> BaseStorageConfig:
        """Create appropriate config type based on the 'type' field in the dictionary.
        
        Args:
            config_dict: Dictionary with configuration values including 'type'
            
        Returns:
            Appropriate BaseStorageConfig subclass instance
            
        Raises:
            ValueError: If type is missing or not supported
        """
        if not isinstance(config_dict, dict):
            raise ValueError(f"Expected dict, got {type(config_dict)}")
            
        storage_type = config_dict.get('type')
        if not storage_type:
            raise ValueError("Missing 'type' field in storage configuration")
            
        if storage_type == 'bigquery':
            return BigQueryStorageConfig(**config_dict)
        elif storage_type in ['file', 'local', 'gcs', 's3', 'plaintext']:
            return FileStorageConfig(**config_dict)
        elif storage_type in ['chromadb', 'vector']:
            return VectorStorageConfig(**config_dict)
        elif storage_type == 'huggingface':
            return HuggingFaceStorageConfig(**config_dict)
        else:
            return GeneratorStorageConfig(**config_dict)
    
    @staticmethod
    def create_storage(config: Union[StorageConfig, BaseStorageConfig], bm_instance=None):
        """Create storage instance based on configuration type.
        
        Args:
            config: StorageConfig instance (from OmegaConf/Hydra)
            bm_instance: BM instance for context (optional)
            
        Returns:
            Storage instance appropriate for the config type
        """
        from buttermilk.data.vector import ChromaDBEmbeddings
        
        # Handle both new type-specific configs and legacy unified configs
        if not isinstance(config, (BaseStorageConfig, dict)):
            raise ValueError(f"Expected StorageConfig or dict instance, got {type(config)}")
        
        # Convert dict (from OmegaConf) to appropriate config type
        if isinstance(config, dict):
            config = StorageFactory.create_config(config)
            
        storage_type = config.type
        
        storage_type = config.type
        
        if storage_type in ["bigquery", "bq"]:
            from buttermilk.storage.bigquery import BigQueryStorage
            return BigQueryStorage(config, bm_instance)
        elif storage_type in ["file", "local", "gcs", "s3"]:
            from buttermilk.storage.file import FileStorage
            return FileStorage(config, bm_instance)
        elif storage_type == "chromadb":
            # Convert VectorStorageConfig to ChromaDBEmbeddings parameters
            chromadb_params = {
                'collection_name': getattr(config, 'collection_name', None) or 'default_collection',
                'persist_directory': getattr(config, 'persist_directory', None) or './data/chromadb',
                'embedding_model': getattr(config, 'embedding_model', None) or 'gemini-embedding-001',
                'dimensionality': getattr(config, 'dimensionality', None) or 3072,
            }
            
            # Add multi-field embedding configuration if specified
            multi_field_embedding = getattr(config, 'multi_field_embedding', None)
            if multi_field_embedding:
                try:
                    # Parse multi-field config into proper Pydantic model
                    multi_field_config = MultiFieldEmbeddingConfig(**multi_field_embedding)
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
            glob_pattern = getattr(config, 'glob', None)
            if not glob_pattern or glob_pattern == "**/*":
                # Set default glob for text files
                if hasattr(config, 'glob'):
                    config.glob = "**/*.txt"
            return FileStorage(config, bm_instance)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

