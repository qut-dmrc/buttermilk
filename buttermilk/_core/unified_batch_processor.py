"""Base class for Unified Batch Processors.

This module implements the base `UnifiedBatchProcessor` class, which satisfies the
`BatchProcessor` protocol and handles common concerns like tracing and error handling
for batch operations.
"""

from typing import AsyncGenerator

from opentelemetry import trace

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import BatchProcessorConfig
from buttermilk._core.protocols import BatchProcessor
from buttermilk._core.types import BaseRecord
from buttermilk import logger


class UnifiedBatchProcessor(BatchProcessor):
    """Base class for all unified batch processors.

    Wraps the raw BatchProcessor protocol with standard behavior:
    - Tracing (OpenTelemetry)
    - Error handling
    - Configuration management
    """

    def __init__(self, config: BatchProcessorConfig):
        self.config = config

    async def process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> AsyncGenerator[list[BaseRecord], None]:
        """Standard batch processing wrapper.

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

        with tracer.start_as_current_span(
            f"batch_processor.{self.config.type}",
            context=parent_context,
            attributes={
                "processor.name": self.config.name or self.config.type,
                "processor.type": self.config.type,
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
                    f"Batch processor {self.config.name} failed: {e}",
                    processor=self.config.name,
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
