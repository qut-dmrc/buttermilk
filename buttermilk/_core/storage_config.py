"""Storage configuration classes with type-specific schemas."""

import os
from typing import Annotated, Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, model_validator


class AdditionalFieldConfig(BaseModel):
    """Configuration for additional fields to embed in multi-field embedding."""

    source_field: str = Field(
        description="Name of the field in Record.metadata to embed",
    )
    chunk_type: str = Field(
        description="Type tag for this chunk (used for filtering searches)",
    )
    min_length: int = Field(
        default=10,
        description="Minimum character length required to embed this field",
    )


class MultiFieldEmbeddingConfig(BaseModel):
    """Configuration for embedding multiple fields from records."""

    content_field: str = Field(
        default="content",
        description="Main content field to chunk and embed (from Record.content)",
    )
    additional_fields: list[AdditionalFieldConfig] = Field(
        default_factory=list,
        description="Additional fields from Record.metadata to embed as single chunks",
    )
    chunk_size: int = Field(
        default=2000,
        description="Chunk size for main content field",
    )
    chunk_overlap: int = Field(
        default=500,
        description="Chunk overlap for main content field",
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
        description="Logical dataset name for filtering/grouping",
    )
    split_type: str | None = Field(
        default=None,
        description="Data split type for datasets (e.g., 'train', 'test', 'validation').",
    )
    randomize: bool = Field(
        default=True,
        description="Whether to randomize query results",
    )
    batch_size: int = Field(
        default=1000,
        ge=1,
        description="Batch size for operations",
    )
    auto_create: bool = Field(
        default=True,
        description="Whether to auto-create storage if it doesn't exist",
    )

    # Data filtering and selection
    filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtering criteria for data operations",
    )
    columns: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Column mapping for renaming data source fields to Record fields. "
            "Dictionary where keys are target Record field names and values are source field names. "
            "Can be empty ({}) if no field renaming is needed. "
            "Example: {'content': 'text', 'ground_truth': 'expected'}"
        ),
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of records to process",
    )

    # Generic fields that some storage types may use
    name: str = Field(
        default="",
        description="Name identifier for the storage configuration.",
    )
    schema_path: str | None = Field(
        default=None,
        description="Path to schema definition file",
    )
    uri: str | None = Field(
        default=None,
        description="URI for data source (alternative to path for some storage types)",
    )
    read_only: bool = Field(default=False, description="Open the database in RO mode.")

    # Provider-specific configuration
    db: dict[str, Any] = Field(
        default_factory=dict,
        description="Database-specific configuration parameters",
    )

    # Record class configuration
    record_class: str | None = Field(
        default=None,
        description=(
            "Fully qualified class name for record instantiation. "
            "Example: 'buttermilk.tools.catalog_test.Title'. "
            "If not specified, defaults to 'buttermilk._core.types.BaseRecord'."
        ),
    )

    # File format configuration
    format: str | None = Field(
        default=None,
        description=(
<<<<<<< HEAD
            "File format for reading/writing. Supported formats: 'json', 'jsonl', 'csv'. If not specified, format is inferred from file extension."
=======
            "File format for reading/writing. "
            "Supported formats: 'json', 'jsonl', 'csv'. "
            "If not specified, format is inferred from file extension."
>>>>>>> origin/stable
        ),
    )

    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    def merge_defaults(self, defaults):
        """Merge this config with default values, prioritizing this config's values.

        None values in self do not override non-None values from defaults.
        """
        exclude_fields = set()
        merged_data = defaults.model_dump(exclude=exclude_fields)
        # Only update with non-None values from self
<<<<<<< HEAD
        self_data = {k: v for k, v in self.model_dump(exclude=exclude_fields).items() if v is not None}
=======
        self_data = {
            k: v
            for k, v in self.model_dump(exclude=exclude_fields).items()
            if v is not None
        }
>>>>>>> origin/stable
        merged_data.update(self_data)
        # Return the same type as self
        return self.__class__(**merged_data)


# Type-specific storage configuration classes


class BigQueryStorageConfig(BaseStorageConfig):
    """Configuration for BigQuery storage operations."""

<<<<<<< HEAD
    type: Literal["bigquery"] = Field(default="bigquery", description="Storage backend type")
=======
    type: Literal["bigquery"] = Field(
        default="bigquery", description="Storage backend type"
    )
>>>>>>> origin/stable

    # Custom SQL support for complex filtering
    custom_where: str | None = Field(
        default=None,
        description=(
            "Custom SQL WHERE clause for complex filtering. "
            "Example: 'year BETWEEN 2020 AND 2023 AND popularity > 50'. "
            "This is appended to the standard dataset/split filters."
        ),
    )
    custom_query: str | None = Field(
        default=None,
        description=(
            "Complete custom SQL query to override default query generation. "
            "Must return columns matching Record fields. "
            "Use {table} placeholder for table reference. "
            "Example: 'SELECT * FROM {table} WHERE complex_conditions'"
        ),
    )

    # BigQuery-specific fields
    project_id: str | None = Field(
        default=None,
        description="Cloud project ID (auto-detected from GOOGLE_CLOUD_PROJECT if not provided)",
    )
    dataset_id: str | None = Field(
        default=None,
        description="Dataset identifier",
    )
    table_id: str | None = Field(
        default=None,
        description="Table identifier",
    )
    clustering_fields: list[str] = Field(
        default=["record_id", "dataset_name", "split_type"],
        description="Fields to use for clustering",
    )

    # Data organization specific to BigQuery
    max_records_per_group: int = Field(
        default=-1,
        description="Maximum records to process per group. -1 for no limit.",
    )
    join: dict[str, str] = Field(
        default_factory=dict,
        description="Configuration for joining with other data sources.",
    )
    agg: bool = Field(
        default=False,
        description="Whether to aggregate results.",
    )
    group: dict[str, str] = Field(
        default_factory=dict,
        description="Grouping configuration (new_group_col: original_col_or_expr).",
    )
    last_n_days: int = Field(
        default=7,
        description="For time-series data, retrieve from the last N days.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_full_table_id(cls, data: Any) -> Any:
        """Parse full_table_id into component parts if provided."""
        if isinstance(data, dict) and "full_table_id" in data:
            full_table_id = data.pop("full_table_id")
            if full_table_id and isinstance(full_table_id, str):
                # Parse the full table ID into components
                parts = full_table_id.split(".")
                if len(parts) == 3:
                    # Only set if not already provided
                    if "project_id" not in data:
                        data["project_id"] = parts[0]
                    if "dataset_id" not in data:
                        data["dataset_id"] = parts[1]
                    if "table_id" not in data:
                        data["table_id"] = parts[2]
                else:
<<<<<<< HEAD
                    raise ValueError(f"Invalid full_table_id format: '{full_table_id}'. Expected 'project.dataset.table'")
=======
                    raise ValueError(
                        f"Invalid full_table_id format: '{full_table_id}'. Expected 'project.dataset.table'"
                    )
>>>>>>> origin/stable
        return data

    @model_validator(mode="after")
    def set_project_id_from_env(self) -> "BigQueryStorageConfig":
        """Set project_id from environment if not already set."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self

    @property
    def full_table_id(self) -> str | None:
        """Get full BigQuery table identifier from constituent parts.

        This is a regular property, not a computed field, so it won't be included
        in model dumps or cause validation errors.
        """
        if all([self.project_id, self.dataset_id, self.table_id]):
            return f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        return None


class FileStorageConfig(BaseStorageConfig):
    """Configuration for file-based storage operations."""

<<<<<<< HEAD
    type: Literal["file", "local", "gcs", "s3", "plaintext"] = Field(description="Storage backend type")
=======
    type: Literal["file", "local", "gcs", "s3", "plaintext"] = Field(
        description="Storage backend type"
    )
>>>>>>> origin/stable

    # File-specific fields
    path: str | None = Field(
        default=None,
        description="File path or URI for storage location",
    )
    glob: str = Field(
        default="**/*",
        description="Glob pattern for matching files.",
    )
    max_records_per_group: int = Field(
        default=-1,
        description="Maximum records to process per group. -1 for no limit.",
    )
    index: list[str] | None = Field(
        default=None,
        description="Columns to use as an index",
    )
    append: bool = Field(
        default=False,
        description="If True, append to existing file instead of overwriting. For JSONL files, new records are appended. For JSON files, existing content is merged.",
    )


class VectorStorageConfig(BaseStorageConfig):
    """Configuration for vector database storage operations."""

    type: Literal["chromadb", "vector"] = Field(description="Storage backend type")

    # Vector storage specific fields
    persist_directory: str | None = Field(
        default=None,
        description="Directory for persisting vector data",
    )
    collection_name: str | None = Field(
        default=None,
        description="Name of the collection",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Name or path of embedding model",
    )
    dimensionality: int | None = Field(
        default=None,
        description="Dimensionality of embeddings",
    )

    # Multi-field embedding configuration
    multi_field_embedding: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Configuration for embedding multiple fields from records. "
            "Format: {'content_field': 'content', 'additional_fields': [{'source_field': 'summary', 'chunk_type': 'summary', 'min_length': 50}]}"
        ),
    )


class HuggingFaceStorageConfig(BaseStorageConfig):
    """Configuration for HuggingFace dataset storage operations."""

<<<<<<< HEAD
    type: Literal["huggingface"] = Field(default="huggingface", description="Storage backend type")
=======
    type: Literal["huggingface"] = Field(
        default="huggingface", description="Storage backend type"
    )
>>>>>>> origin/stable

    # HuggingFace specific fields
    dataset_id: str | None = Field(
        default=None,
        description="HuggingFace dataset identifier",
    )
    split: str = Field(
        default="train",
        description="Data split identifier (train/test/val).",
    )


class GeneratorStorageConfig(BaseStorageConfig):
    """Configuration for generator-based storage operations."""

<<<<<<< HEAD
    type: Literal["generator", "job", "outputs"] = Field(description="Storage backend type")
=======
    type: Literal["generator", "job", "outputs"] = Field(
        description="Storage backend type"
    )
>>>>>>> origin/stable


class DuckDBStorageConfig(BaseStorageConfig):
    """Configuration for DuckDB storage operations."""

<<<<<<< HEAD
    type: Literal["duckdb"] = Field(default="duckdb", description="Storage backend type")
=======
    type: Literal["duckdb"] = Field(
        default="duckdb", description="Storage backend type"
    )
>>>>>>> origin/stable

    # DuckDB-specific fields
    database: str | None = Field(
        default=None,
        description="Path to DuckDB database file or ':memory:' for in-memory database",
    )
    table_name: str | None = Field(
        default=None,
        description="Name of the table to query/write",
    )
    schema_name: str | None = Field(
        default=None,
        description="Schema name (defaults to 'main' if not specified)",
    )
    custom_query: str | None = Field(
        default=None,
<<<<<<< HEAD
        description=("Complete custom SQL query to override default query generation. Cannot be used with write operations."),
=======
        description=(
            "Complete custom SQL query to override default query generation. Cannot be used with write operations."
        ),
>>>>>>> origin/stable
    )


# Discriminated union for all storage config types
StorageConfig = Annotated[
<<<<<<< HEAD
    BigQueryStorageConfig | FileStorageConfig | VectorStorageConfig | HuggingFaceStorageConfig | GeneratorStorageConfig | DuckDBStorageConfig,
=======
    BigQueryStorageConfig
    | FileStorageConfig
    | VectorStorageConfig
    | HuggingFaceStorageConfig
    | GeneratorStorageConfig
    | DuckDBStorageConfig,
>>>>>>> origin/stable
    Field(discriminator="type"),
]


class StorageFactory:
    """Factory for creating storage instances based on configuration."""

    @staticmethod
    def create_config(config_dict: dict) -> BaseStorageConfig:
        """Create appropriate config type based on the 'type' field in the dictionary.

        Uses the discriminated union to properly validate and create the correct subclass.

        Args:
            config_dict: Dictionary with configuration values including 'type'

        Returns:
            Appropriate BaseStorageConfig subclass instance

        Raises:
            ValueError: If type is missing or not supported

        """
        if not isinstance(config_dict, dict):
            raise ValueError(f"Expected dict, got {type(config_dict)}")

        storage_type = config_dict.get("type")
        if not storage_type:
            raise ValueError("Missing 'type' field in storage configuration")

        # Use the discriminated union for proper validation
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StorageConfig)
        return adapter.validate_python(config_dict)

    @staticmethod
    def create_storage(config: StorageConfig | BaseStorageConfig | dict | DictConfig):
        """Create storage instance based on configuration type.

        Args:
            config: StorageConfig instance (from OmegaConf/Hydra)

        Returns:
            Storage instance appropriate for the config type

        """
        from buttermilk.data.vector import ChromaDBEmbeddings

        if isinstance(config, DictConfig):
            # Convert to dict first
            config = OmegaConf.to_container(config, resolve=True)
        if isinstance(config, dict):
            config = StorageFactory.create_config(config)

        storage_type = config.type

        if storage_type in ["bigquery", "bq"]:
            from buttermilk.storage.bigquery import BigQueryStorage

            return BigQueryStorage(config)
        if storage_type in ["file", "local", "gcs", "s3"]:
            from buttermilk.storage.file import FileStorage

            return FileStorage(config)
        if storage_type == "chromadb":
            # Convert VectorStorageConfig to ChromaDBEmbeddings parameters
            chromadb_params = {
<<<<<<< HEAD
                "collection_name": getattr(config, "collection_name", None) or "default_collection",
                "persist_directory": getattr(config, "persist_directory", None) or "./data/chromadb",
                "embedding_model": getattr(config, "embedding_model", None) or "gemini-embedding-001",
=======
                "collection_name": getattr(config, "collection_name", None)
                or "default_collection",
                "persist_directory": getattr(config, "persist_directory", None)
                or "./data/chromadb",
                "embedding_model": getattr(config, "embedding_model", None)
                or "gemini-embedding-001",
>>>>>>> origin/stable
                "dimensionality": getattr(config, "dimensionality", None) or 3072,
            }

            # Add other ChromaDB-specific fields if present in config
            for field in [
                "concurrency",
                "upsert_batch_size",
                "embedding_batch_size",
                "arrow_save_dir",
            ]:
                if hasattr(config, field) and getattr(config, field) is not None:
                    chromadb_params[field] = getattr(config, field)

            return ChromaDBEmbeddings(**chromadb_params)
        if storage_type == "huggingface":
            from buttermilk.storage.huggingface import HuggingFaceStorage

            return HuggingFaceStorage(config)
        if storage_type == "plaintext":
            # Use FileStorage with plaintext-specific configuration
            from buttermilk.storage.file import FileStorage

            # For plaintext, we typically use glob patterns
            glob_pattern = getattr(config, "glob", None)
            if not glob_pattern or glob_pattern == "**/*":
                # Set default glob for text files
                if hasattr(config, "glob"):
                    config.glob = "**/*.txt"
            return FileStorage(config)
        if storage_type == "duckdb":
            from buttermilk.storage.duckdb import DuckDBStorage

            return DuckDBStorage(config)
        raise ValueError(f"Unsupported storage type: {storage_type}")
