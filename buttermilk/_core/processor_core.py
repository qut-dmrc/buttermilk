"""Shared base class for all pipeline processors.

This module provides ProcessorCore, a minimal base class that encapsulates
common functionality shared between ClassifierCore, LLMCore, and
ToxicityClassifierCore.

The design enables:
- Consistent tracing across all processor types via TracingMixin
- Shared infrastructure (trace_writer, error handling)
- Unified Processor protocol implementation patterns
"""

from abc import abstractmethod
from typing import Any, AsyncGenerator, Optional

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.tracing_mixin import TracingMixin
from buttermilk._core.types import BaseRecord


class ProcessorCore(TracingMixin):
    """Minimal shared base for all pipeline processors.

    Inherits from TracingMixin to provide:
    - Lazy trace_writer property for BigQuery persistence
    - Common trace emission patterns (_emit_success_trace, _emit_error_trace)
    - Helper methods for building agent_info and metadata

    Subclasses (ClassifierCore, LLMCore) implement their specific processing
    logic while inheriting common infrastructure from this base and TracingMixin.

    Example:
        ```python
        class MyProcessor(ProcessorCore):
            async def process(
                self,
                record: BaseRecord,
                *,
                processor_stage: str,
                parent_trace_id: str | None = None,
                **kwargs,
            ) -> AsyncGenerator[BaseRecord, None]:
                start_time = time.time()
                try:
                    result = await self._do_processing(record)
                    duration_ms = (time.time() - start_time) * 1000

                    await self._emit_success_trace(
                        record=record,
                        outputs=result,
                        processor_stage=processor_stage,
                        parent_trace_id=parent_trace_id,
                        duration_ms=duration_ms,
                    )

                    yield enriched_record
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    await self._emit_error_trace(
                        record=record,
                        error=e,
                        processor_stage=processor_stage,
                        parent_trace_id=parent_trace_id,
                        duration_ms=duration_ms,
                    )
                    raise
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize processor with configuration.

        Args:
            **kwargs: Configuration stored in self.parameters
        """
        self.parameters: dict[str, Any] = kwargs
        self._trace_writer: Any = None

    @abstractmethod
    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord and yield zero or more output records.

        This is the Processor protocol method that subclasses must implement.

        Args:
            record: Input BaseRecord to process
            processor_stage: Unique stage identifier for tracing
            parent_trace_id: Optional trace ID for distributed tracing
            **kwargs: Additional arguments

        Yields:
            BaseRecord: Enriched record(s) with processing results

        Raises:
            ProcessingError: If processing fails
        """
        raise NotImplementedError("Subclasses must implement process()")
        yield  # Make this a generator
