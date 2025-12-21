"""Base class for Unified Processors.

This module implements the base `UnifiedProcessor` class, which satisfies the
`Processor` protocol and handles common concerns like tracing and error handling
using the `ProcessingContext`.

Processors are Pydantic models that can be instantiated directly via Hydra's
`_target_` mechanism. No registry or separate config classes needed.
"""

from typing import Any, AsyncGenerator

from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord
from buttermilk import logger


class UnifiedProcessor(BaseModel, Processor):
    """Base class for all unified processors.

    Processors are Pydantic models with field-based configuration.
    Hydra instantiates them directly via `_target_`.

    Common behavior provided:
    - Tracing (OpenTelemetry spans)
    - Error handling with structured logging
    - Enabled/disabled flag

    Subclasses define their own fields and implement `_process_record()`.

    Example config:
        ```yaml
        processors:
          - _target_: buttermilk.processors.GroupchatProcessor
            flow_name: trans
            flow_config: ${flows.trans}
        ```
    """

    name: str | None = Field(default=None, description="Optional processor instance name for tracing")
    enabled: bool = Field(default=True, description="Whether this processor is active")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # For OmegaConf DictConfig and complex types
        extra="forbid",  # Strict validation - no unknown fields
    )

    @property
    def processor_type(self) -> str:
        """Return processor type name for tracing. Defaults to class name."""
        return self.__class__.__name__

    async def process(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Standard processing wrapper with tracing.

        Handles OpenTelemetry span creation and delegates to `_process_record`.
        """
        tracer = trace.get_tracer("buttermilk.processor")

        # Create a span for this processor execution
        # Attach to parent span from context if available
        parent_context = trace.set_span_in_context(context.span) if context.span else None

        processor_name = self.name or self.processor_type

        with tracer.start_as_current_span(
            f"processor.{self.processor_type}",
            context=parent_context,
            attributes={
                "processor.name": processor_name,
                "processor.type": self.processor_type,
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
                    f"Processor {processor_name} failed: {e}",
                    processor=processor_name,
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

        Args:
            context: ProcessingContext containing the record and session state

        Yields:
            Zero or more BaseRecord objects (filter, transform, or expand)
        """
        raise NotImplementedError("Subclasses must implement _process_record")
        yield  # Make it a generator
