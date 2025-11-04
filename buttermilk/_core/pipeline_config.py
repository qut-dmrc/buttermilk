"""Type-safe configuration models for pipeline processing.

This module provides Pydantic models for pipeline configurations,
which orchestrate multi-stage data processing workflows.
"""

from typing import Any

from pydantic import BaseModel, Field

from buttermilk._core.storage_config import BaseStorageConfig


class TMDBProcessorConfig(BaseModel):
    """Configuration for TMDB metadata enrichment processor."""

    region: str = Field(default="US", description="Region code for TMDB data (e.g., 'US', 'GB')")
    language: str = Field(default="en-US", description="Language code for TMDB data")

    model_config = {
        "extra": "allow",  # Allow additional TMDB-specific parameters
    }


class PipelineConfig(BaseModel):
    """Configuration for pipeline processing mode.

    Pipelines orchestrate multiple processors (TMDB enrichment, uploading, etc.)
    to transform data from a source to an output destination.
    """

    # Source configuration
    source: dict[str, Any] | BaseStorageConfig | None = Field(default=None, description="Source storage configuration for input data")

    # Output configuration
    output: dict[str, Any] | BaseStorageConfig | None = Field(default=None, description="Output storage configuration for processed data")

    # Processor configurations
    tmdb: TMDBProcessorConfig | dict[str, Any] | bool | None = Field(
        default=None, description="TMDB processor configuration (True to enable with defaults, dict for custom config, None to disable)"
    )

    # Pipeline execution parameters
    concurrency: int = Field(default=1, ge=1, description="Number of concurrent processing tasks")
    max_records: int | None = Field(default=None, description="Maximum number of records to process (None for unlimited)")
    sample_size: int | None = Field(default=None, description="Number of samples to collect from source (for sampling pipelines)")

    # Uploader configuration
    buffer_size: int = Field(default=10, ge=1, description="Buffer size for batch uploading")
    flush_interval: int = Field(default=30, ge=1, description="Interval in seconds for flushing buffered data")

    model_config = {
        "extra": "allow",  # Allow additional processor configurations
        "arbitrary_types_allowed": True,  # Allow storage config objects
    }

    def get_source_config(self) -> BaseStorageConfig | None:
        """Get source storage configuration.

        Returns:
            StorageConfig instance or None if not configured
        """
        if self.source is None:
            return None
        if isinstance(self.source, BaseStorageConfig):
            return self.source
        if isinstance(self.source, dict):
            from buttermilk._core.storage_config import StorageFactory

            return StorageFactory.create_config(self.source)
        return None

    def get_output_config(self) -> BaseStorageConfig | None:
        """Get output storage configuration.

        Returns:
            StorageConfig instance or None if not configured
        """
        if self.output is None:
            return None
        if isinstance(self.output, BaseStorageConfig):
            return self.output
        if isinstance(self.output, dict):
            from buttermilk._core.storage_config import StorageFactory

            return StorageFactory.create_config(self.output)
        return None

    def get_tmdb_config(self) -> TMDBProcessorConfig | None:
        """Get TMDB processor configuration.

        Returns:
            TMDBProcessorConfig instance or None if TMDB processing is disabled
        """
        if self.tmdb is None or self.tmdb is False:
            return None
        if self.tmdb is True:
            return TMDBProcessorConfig()
        if isinstance(self.tmdb, TMDBProcessorConfig):
            return self.tmdb
        if isinstance(self.tmdb, dict):
            return TMDBProcessorConfig(**self.tmdb)
        return None
