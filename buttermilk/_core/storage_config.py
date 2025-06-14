"""Storage configuration classes for unified storage operations."""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


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

    # Generic URI field for various storage types
    uri: str | None = Field(
        default=None,
        description="URI for data source (alternative to path for some storage types)"
    )

    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
        "exclude_none": True,
        "exclude_unset": True,
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
        merged_data = defaults.model_dump(exclude_unset=True, exclude=exclude_fields)
        merged_data.update(self.model_dump(exclude_unset=True, exclude=exclude_fields))
        return StorageConfig(**merged_data)


class BigQueryDefaults(BaseModel):
    """Default configuration values specifically for BigQuery operations."""

    dataset_id: str = Field(default="buttermilk")
    table_id: str = Field(default="records")
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

