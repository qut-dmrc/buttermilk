"""Type-safe configuration models for pipeline processing.

This module provides Pydantic models for pipeline configurations,
which orchestrate multi-stage data processing workflows.
"""

from typing import Any

import hydra
from pydantic import BaseModel, Field

from buttermilk._core.exceptions import FatalError
from buttermilk._core.log import logger
from buttermilk._core.storage_config import BaseStorageConfig
from buttermilk.utils.utils import clean_empty_values, expand_dict


class TMDBProcessorConfig(BaseModel):
    """Configuration for TMDB metadata enrichment processor."""

    region: str = Field(
        default="US", description="Region code for TMDB data (e.g., 'US', 'GB')"
    )
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
    source: dict[str, Any] | BaseStorageConfig | None = Field(
        default=None, description="Source storage configuration for input data"
    )

    # Output configuration
    output: dict[str, Any] | BaseStorageConfig | None = Field(
        default=None, description="Output storage configuration for processed data"
    )

    # Processor configurations
    tmdb: TMDBProcessorConfig | dict[str, Any] | bool | None = Field(
        default=None,
        description="TMDB processor configuration (True to enable with defaults, dict for custom config, None to disable)",
    )

    # Pipeline execution parameters
    concurrency: int = Field(
        default=1, ge=1, description="Number of concurrent processing tasks"
    )
    max_records: int | None = Field(
        default=None,
        description="Maximum number of records to process (None for unlimited)",
    )
    sample_size: int | None = Field(
        default=None,
        description="Number of samples to collect from source (for sampling pipelines)",
    )
    num_runs: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate each source record (for reliability studies)",
    )

    # Uploader configuration
    buffer_size: int = Field(
        default=10, ge=1, description="Buffer size for batch uploading"
    )
    flush_interval: int = Field(
        default=30, ge=1, description="Interval in seconds for flushing buffered data"
    )

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


class ProcessorVariants(BaseModel):
    """Factory for creating multiple processor instances with different configurations.

    Similar to AgentVariants but for processors. Generates multiple processor
    instances with different parameter combinations for A/B testing, ensemble
    methods, or quality comparison.

    NOTE: For repeated runs (num_runs), use ReplicatingSource at the pipeline
    source level instead of replicating processors. This prevents exponential
    API call multiplication.

    Attributes:
        processor_obj: Processor class path to instantiate (e.g., 'buttermilk.processors.LLMCore')
        variants: Parallel variant parameters (e.g., {'model': ['gpt-4', 'claude-3']})
        parameters: Base processor parameters merged with variant params

    Example:
        ```yaml
        - processor_obj: buttermilk.processors.LLMCore
          variants:
            model: ["gpt-4", "claude-3", "gemini-pro"]
            temperature: [0.7]
          parameters:
            prompt_template: "default"
        ```
    """

    processor_obj: str = Field(
        description="Processor class path to instantiate (e.g., 'buttermilk.processors.LLMCore')"
    )
    variants: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameters for parallel processor variations (e.g., {'model': ['gpt-4', 'claude-3']})",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Base processor parameters merged with variant params",
    )

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }

    def get_configs(
        self, flow_default_params: dict[str, Any] | None = None
    ) -> list[tuple[type[Any], dict[str, Any]]]:
        """Generate processor configurations from variants.

        Expands variant parameters to create multiple processor instances.
        Each variant gets its own parameter set combining base params,
        flow defaults, and variant-specific values.

        Args:
            flow_default_params: Optional default parameters from pipeline config

        Returns:
            List of (processor_class, config_dict) tuples

        Raises:
            ValueError: If processor_obj cannot be instantiated
            FatalError: If no processor configurations generated

        Example:
            >>> variants = ProcessorVariants(
            ...     processor_obj="buttermilk.processors.LLMCore",
            ...     variants={"model": ["gpt-4", "claude-3"]},
            ...     parameters={"temperature": 0.7}
            ... )
            >>> configs = variants.get_configs()
            >>> len(configs)
            2
        """
        if flow_default_params is None:
            flow_default_params = {}

        # Get processor class
        try:
            processor_class = hydra.utils.get_class(self.processor_obj)
        except Exception as e:
            raise ValueError(
                f"Failed to load processor class '{self.processor_obj}': {e}"
            ) from e

        # Expand variant combinations
        variant_combinations = (
            expand_dict(clean_empty_values(self.variants)) if self.variants else [{}]
        )

        generated_configs: list[tuple[type[Any], dict[str, Any]]] = []

        for variant_params in variant_combinations:
            # Merge parameters: flow defaults, then base, then variant-specific
            final_params = {
                **flow_default_params,
                **self.parameters,
                **variant_params,
            }

            generated_configs.append((processor_class, final_params))

        if not generated_configs:
            raise FatalError(
                f"No processor configurations generated for ProcessorVariants: {self.processor_obj}"
            )

        logger.debug(
            f"ProcessorVariants generated {len(generated_configs)} configs for {self.processor_obj}",
            processor_obj=self.processor_obj,
            variant_count=len(generated_configs),
        )

        return generated_configs
