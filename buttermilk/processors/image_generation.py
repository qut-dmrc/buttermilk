"""Image generation processor for pipeline integration.

This processor adapts TextToImageClient for use in data pipelines by converting
between BaseRecord and ImageRecord formats.
"""

from typing import Any, AsyncGenerator, Type

from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.types import BaseRecord
from buttermilk.agents.imagegen import TextToImageClient


class ImageGenerationProcessor(BaseModel):
    """Processor that generates images from text prompts using TextToImageClient.

    This adapter allows any TextToImageClient to be used in a pipeline by:
    1. Extracting prompt from BaseRecord.content
    2. Calling client.generate() to create image
    3. Converting ImageRecord result back to BaseRecord with image URI in metadata

    Preserves all original metadata fields and adds image generation metadata.
    """

    client_class: Type[TextToImageClient] = Field(description="The TextToImageClient class to instantiate for generation")

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by generating an image from its content.

        Args:
            record: BaseRecord with prompt in content field
            processor_stage: Pipeline stage identifier (e.g., "generate")
            **kwargs: Additional arguments (unused)

        Yields:
            BaseRecord with original data plus image_uri and model in metadata

        Raises:
            ValueError: If record.content is empty or None
        """
        if not record.content:
            raise ValueError(f"Record {record.record_id} has empty content - cannot generate image")

        # Instantiate the client
        client = self.client_class()

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
                    "prompt": image_record.prompt,
                },
            }
        )

        yield result
