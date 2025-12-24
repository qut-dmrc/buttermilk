"""Base class for Unified Batch Processors.

This module implements the base `UnifiedBatchProcessor` class, which satisfies the
`BatchProcessor` protocol and handles common concerns like tracing and error handling
for batch operations.

Batch processors are Pydantic models that can be instantiated directly via Hydra's
`_target_` mechanism.
"""

from typing import AsyncGenerator

from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import BatchProcessor
from buttermilk._core.types import BaseRecord
from buttermilk import logger


class UnifiedBatchProcessor(BaseModel):
    """Base class for all unified batch processors.

    Batch processors are Pydantic models with field-based configuration.
    Hydra instantiates them directly via `_target_`.

    Common behavior provided:
    - Tracing (OpenTelemetry spans)
    - Error handling with structured logging
    - Batch size configuration
    - Enabled/disabled flag

    Subclasses define their own fields and implement `_process_batch()`.

    Example config:
        ```yaml
        processors:
          - _target_: buttermilk.processors.EmbeddingProcessor
            embedding_model: gemini-embedding-001
            batch_size: 32
        ```
    """

    name: str | None = Field(default=None, description="Optional processor instance name for tracing")
    enabled: bool = Field(default=True, description="Whether this processor is active")
    batch_size: int = Field(default=32, description="Number of records to batch together")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # For OmegaConf DictConfig and complex types
        extra="forbid",  # Strict validation - no unknown fields
    )

    @property
    def processor_type(self) -> str:
        """Return processor type name for tracing. Defaults to class name."""
        return self.__class__.__name__

    async def process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> AsyncGenerator[list[BaseRecord], None]:
        """Standard batch processing wrapper with tracing.

        Handles tracing setup and delegates to `_process_batch`.

        Args:
            contexts: List of ProcessingContext objects to process as a batch

        Yields:
            list[BaseRecord]: Batch of output records
        """
        tracer = trace.get_tracer("buttermilk.processor")

        # Create a span for this batch processor execution
        # Use the first context's span as parent if available
        parent_context = None
        if contexts and contexts[0].span:
            parent_context = trace.set_span_in_context(contexts[0].span)

        processor_name = self.name or self.processor_type

        with tracer.start_as_current_span(
            f"batch_processor.{self.processor_type}",
            context=parent_context,
            attributes={
                "processor.name": processor_name,
                "processor.type": self.processor_type,
                "batch.size": len(contexts),
                "batch.record_ids": [ctx.record.record_id for ctx in contexts],
            }
        ) as span:
            try:
                # Delegate to concrete implementation
                async for output_batch in self._process_batch(contexts):
                    span.set_attribute("batch.output_size", len(output_batch))
                    yield output_batch

            except Exception as e:
                span.record_exception(e)
                logger.error(
                    f"Batch processor {processor_name} failed: {e}",
                    processor=processor_name,
                    batch_size=len(contexts),
                    error=str(e)
                )
                raise

    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> AsyncGenerator[list[BaseRecord], None]:
        """Concrete implementation of batch processing logic.

        Must be implemented by subclasses.

        Args:
            contexts: List of ProcessingContext objects to process

        Yields:
            list[BaseRecord]: Batch of output records
        """
        raise NotImplementedError("Subclasses must implement _process_batch")
        yield  # Make it a generator

    async def finalize(self) -> None:
        """Optional cleanup or finalization logic.

        Default implementation does nothing. Subclasses can override for cleanup.
        """
        pass
