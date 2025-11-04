"""Image generation processor for pipeline integration.

This processor adapts TextToImageClient for use in data pipelines by converting
between BaseRecord and ImageRecord formats.
"""

from typing import Any, AsyncGenerator, Optional, Type

from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.types import BaseRecord
from buttermilk.agents.imagegen import TextToImageClient


class ImageGenerationProcessor(BaseModel):
    """Processor that generates images from text prompts using TextToImageClient.

    This adapter allows any TextToImageClient to be used in a pipeline by:
    1. Reading model_class from record.metadata (set by BatchExpansionProcessor)
    2. Extracting prompt from BaseRecord.content
    3. Calling client.generate() to create image
    4. Converting ImageRecord result back to BaseRecord with image URI in metadata

    Preserves all original metadata fields and adds image generation metadata.

    Args:
        client_class: Optional default client class. If not specified, reads from record.metadata['model_class']
    """

    client_class: Optional[Type[TextToImageClient]] = Field(
        default=None, description="Optional default TextToImageClient class. If None, reads from record.metadata['model_class']"
    )

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by generating an image from its content.

        Args:
            record: BaseRecord with prompt in content field. Should have model_class in metadata if client_class not set.
            processor_stage: Pipeline stage identifier (e.g., "generate")
            **kwargs: Additional arguments (unused)

        Yields:
            BaseRecord with original data plus image_uri and model in metadata

        Raises:
            ValueError: If record.content is empty, None, or model_class cannot be determined
        """
        if not record.content:
            raise ValueError(f"Record {record.record_id} has empty content - cannot generate image")

        # Determine which client class to use
        if self.client_class is not None:
            # Use configured client class
            client_cls = self.client_class
        elif "model_class" in record.metadata:
            # Use model_class from record metadata (set by BatchExpansionProcessor)
            client_cls = record.metadata["model_class"]
        else:
            raise ValueError(f"Record {record.record_id} has no model_class in metadata and ImageGenerationProcessor has no default client_class")

        # Instantiate the client
        client = client_cls()

        logger.info(
            f"Generating image for record {record.record_id} using {client.model}",
            extra={"record_id": record.record_id, "model": client.model, "stage": processor_stage},
        )

        # Generate image
        image_record = await client.generate(text=record.content)

        # Convert ImageRecord back to BaseRecord with metadata
        result = record.model_copy(
            update={
                "metadata": {
                    **record.metadata,  # Preserve all original metadata
                    "image_uri": image_record.uri,
                    "model": client.model,
                    "model_class": client_cls.__name__,  # Store class name for tracking
                    "prompt": image_record.prompt,
                },
            }
        )

        yield result
