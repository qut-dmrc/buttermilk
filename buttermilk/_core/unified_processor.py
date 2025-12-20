"""Base class for Unified Processors.

This module implements the base `UnifiedProcessor` class, which satisfies the
new `Processor` protocol and handles common concerns like tracing and error handling
using the `ProcessingContext`.
"""

from typing import AsyncGenerator

from opentelemetry import trace

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import ProcessorConfig
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord
from buttermilk import logger


class UnifiedProcessor(Processor):
    """Base class for all unified processors.
    
    Wraps the raw protocol with standard behavior:
    - Tracing (OpenTelemetry)
    - Error handling
    - Configuration management
    """

    def __init__(self, config: ProcessorConfig):
        self.config = config

    async def process(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Standard processing wrapper.
        
        Handles tracing setup and delegates to `_process_record`.
        """
        tracer = trace.get_tracer("buttermilk.processor")
        
        # Create a span for this processor execution
        # We attach it to the parent span from the context if available
        parent_context = trace.set_span_in_context(context.span) if context.span else None
        
        with tracer.start_as_current_span(
            f"processor.{self.config.type}",
            context=parent_context,
            attributes={
                "processor.name": self.config.name or self.config.type,
                "processor.type": self.config.type,
                "record.id": context.record.record_id,
            }
        ) as span:
            try:
                # Delegate to concrete implementation
                async for output in self._process_record(context):
                    yield output
                    
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    f"Processor {self.config.name} failed: {e}",
                    processor=self.config.name,
                    record_id=context.record.record_id,
                    error=str(e)
                )
                raise

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Concrete implementation of processing logic.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_record")
        yield  # Make it a generator
