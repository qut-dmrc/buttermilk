"""Image generation processor for pipeline integration.

This processor adapts TextToImageClient for use in data pipelines by converting
between BaseRecord and ImageRecord formats.
"""

from typing import AsyncGenerator, Optional, Type

from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.agents.imagegen import ALL_IMAGE_CLIENTS, TextToImageClient


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
        default=None,
        description="Optional default TextToImageClient class. If None, reads from record.metadata['model_class']",
    )

    def _resolve_client_class(self, model_class: Type[TextToImageClient] | str) -> Type[TextToImageClient]:
        """Resolve model_class to actual class object.

        Args:
            model_class: Either a TextToImageClient class or string class name

        Returns:
            The resolved TextToImageClient class

        Raises:
            ValueError: If string class name not found in registry
        """
        # If already a class, return it
        if isinstance(model_class, type) and issubclass(model_class, TextToImageClient):
            return model_class

        # If string, look up in registry
        if isinstance(model_class, str):
            for client_cls in ALL_IMAGE_CLIENTS:
                if client_cls.__name__ == model_class:
                    return client_cls
            raise ValueError(f"Unknown model class name: {model_class}. Available: {[cls.__name__ for cls in ALL_IMAGE_CLIENTS]}")

        raise ValueError(f"model_class must be a TextToImageClient class or string class name, got: {type(model_class)}")

    async def process(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by generating an image from its content.

        Args:
            context: ProcessingContext with record containing prompt in content field.
                     Record should have model_class in metadata if client_class not set.

        Yields:
            BaseRecord with original data plus image_uri and model in metadata

        Raises:
            ValueError: If record.content is empty, None, or model_class cannot be determined
        """
        record = context.record
        if not record.content:
            raise ValueError(f"Record {record.record_id} has empty content - cannot generate image")

        # Determine which client class to use
        if self.client_class is not None:
            # Use configured client class
            client_cls = self._resolve_client_class(self.client_class)
        elif "model_class" in record.metadata:
            # Use model_class from record metadata (set by BatchExpansionProcessor)
            # This may be a class object or a string class name (if serialized)
            client_cls = self._resolve_client_class(record.metadata["model_class"])
        else:
            raise ValueError(f"Record {record.record_id} has no model_class in metadata and ImageGenerationProcessor has no default client_class")

        # Instantiate the client
        client = client_cls()

        logger.info(
            f"Generating image for record {record.record_id} using {client.model}",
            extra={
                "record_id": record.record_id,
                "model": client.model,
                "stage": context.session_id,
            },
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
