"""Batch expansion processor for multi-model image generation.

This module provides BatchExpansionProcessor which expands each input record
into multiple copies for batch generation across different models and repetitions.
"""

from collections.abc import AsyncGenerator
from typing import Any, Type

from pydantic import BaseModel, Field

from buttermilk._core.types import BaseRecord
from buttermilk.agents.imagegen import CHEAP_IMAGE_CLIENTS, TextToImageClient


class BatchExpansionProcessor(BaseModel):
    """Expands each record into multiple copies for batch generation.

    Creates repetitions × models copies of each input record, adding
    metadata for repetition number and model instance.

    Attributes:
        repetitions: Number of times to repeat each (record, model) combination.
        models: List of TextToImageClient classes or instances to use for generation.
    """

    repetitions: int = 3
<<<<<<< HEAD
    models: list[Type[TextToImageClient] | TextToImageClient] = Field(default_factory=lambda: [cls() for cls in CHEAP_IMAGE_CLIENTS])
=======
    models: list[Type[TextToImageClient] | TextToImageClient] = Field(
        default_factory=lambda: [cls() for cls in CHEAP_IMAGE_CLIENTS]
    )
>>>>>>> origin/stable

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Yield one record per (repetition, model) combination.

        For each input record, generates repetitions × len(models) output records,
        each with unique record_id and metadata indicating the repetition number
        and model class to use.

        Args:
            record: Input BaseRecord to expand
            processor_stage: Pipeline stage identifier (e.g., "expand")
            **kwargs: Additional arguments (unused)

        Yields:
            BaseRecord for each (repetition, model) combination with metadata:
            - repetition: 0-indexed repetition number
            - model_class: The TextToImageClient class object (not name)
            - model_prefix: Prefix string from the model instance
            - All original metadata preserved
        """
        for rep in range(self.repetitions):
            for model in self.models:
                # Handle both class types and instances
                if isinstance(model, type):
                    # It's a class, instantiate it to get prefix
                    model_class = model
                    model_instance = model()
                else:
                    # It's already an instance
                    model_class = type(model)
                    model_instance = model

                expanded = record.model_copy(
                    update={
                        "record_id": f"{record.record_id}_rep{rep}_{model_class.__name__}",
                        "metadata": {
                            **record.metadata,
                            "repetition": rep,
                            "model_class": model_class,  # Store class object, not name
                            "model_prefix": model_instance.prefix,
                        },
                    }
                )
                yield expanded
