"""Configuration for Pipelines in the Unified Architecture.

Defines the structure of a pipeline, which is essentially a sequence of processor configurations.
"""

import itertools
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from buttermilk._core.processor_config import ProcessorConfigUnion
from buttermilk.utils.validators import import_class_from_path


class ProcessorVariants(BaseModel):
    """Configuration for running a processor with multiple parameter variants.

    Expands parameters into cartesian product of all variant combinations.
    This enables A/B testing and parallel execution of different processor
    configurations within a single pipeline run.

    Attributes:
        processor_obj: Fully qualified class path of processor to instantiate
            (e.g., "buttermilk.processors.JMESPathTransform").
        variants: Parameter variations to expand. Each key is a parameter name,
            each value is a list of values to try. Creates cartesian product.
        parameters: Static parameters merged with all variant configs.
        num_runs: Number of times to run each variant (not used in get_configs,
            handled at source level instead to avoid exponential multiplication).

    Example:
        ```python
        cfg = ProcessorVariants(
            processor_obj="buttermilk.processors.LLMCore",
            variants={
                "model": ["gpt-4", "claude-3"],
                "temperature": [0.0, 0.7],
            },
            parameters={"prompt_template": "default"},
        )
        # Creates 4 configs: 2 models × 2 temperatures
        configs = cfg.get_configs()
        ```
    """

    processor_obj: str = Field(
        ...,
        description="Fully qualified class path of processor to instantiate.",
    )
    variants: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameter variants to expand into cartesian product.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Static parameters for all variants.",
    )
    num_runs: int = Field(
        default=1,
        description="Number of runs (handled at source level, not in get_configs).",
    )

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    def get_configs(self) -> list[tuple[type, dict[str, Any]]]:
        """Generate processor configs for each variant combination.

        Creates cartesian product of all variant parameters and merges with
        base parameters. Loads the processor class and returns tuples of
        (processor_class, config_dict).

        Returns:
            List of (processor_class, config_dict) tuples, one per variant.

        Raises:
            ValueError: If processor_obj class path is invalid or class cannot be loaded.
            ImportError: If module cannot be imported.
            AttributeError: If class cannot be found in module.

        Example:
            >>> cfg = ProcessorVariants(
            ...     processor_obj="buttermilk.processors.JMESPathTransform",
            ...     variants={"expression": ["content", "metadata"]},
            ...     parameters={"output_field": "result"},
            ... )
            >>> configs = cfg.get_configs()
            >>> len(configs)
            2
            >>> configs[0][1]["expression"]
            'content'
            >>> configs[0][1]["output_field"]
            'result'
        """
        # Load the processor class
        try:
            processor_class = import_class_from_path(self.processor_obj)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(
                f"Failed to load processor class from '{self.processor_obj}': {e}"
            ) from e

        # If no variants, return single config with base parameters
        if not self.variants:
            return [(processor_class, self.parameters.copy())]

        # Generate cartesian product of all variant values
        # Extract keys and values in consistent order
        variant_keys = list(self.variants.keys())
        variant_values = [self.variants[k] for k in variant_keys]

        configs = []
        for value_combination in itertools.product(*variant_values):
            # Create config dict by merging base parameters with variant values
            config = self.parameters.copy()
            for key, value in zip(variant_keys, value_combination):
                config[key] = value
            configs.append((processor_class, config))

        return configs


class PipelineConfig(BaseModel):
    """Configuration for a complete processing pipeline.

    Attributes:
        name: Unique name for the pipeline.
        processors: Ordered list of processor configurations.
        version: Version of the pipeline configuration.
    """
    name: str = Field(..., description="Name of the pipeline.")
    processors: list[ProcessorConfigUnion] = Field(..., description="Sequence of processors.")
    version: str = Field(default="1.0", description="Config version.")

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )