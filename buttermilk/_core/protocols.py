"""Core protocols for the Unified Processor Architecture.

This module defines the standard interfaces for all processors in the Buttermilk framework.
It separates concerns into:
- Processor: Standard single-record processing (async generator).
- BatchProcessor: Efficient batch processing (e.g., for GPUs).

These protocols enable a unified execution model where "everything is a processor".

Typed Data Flow:
    Processors can yield any type, not just BaseRecord. This enables typed data flow
    where processors yield their natural output type (e.g., a Pydantic model for
    structured LLM output). The pipeline handles both BaseRecord and arbitrary types,
    adding metadata only to records that support it.
"""

from typing import Any, AsyncGenerator, Protocol, runtime_checkable

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord


@runtime_checkable
class Processor(Protocol):
    """Standard processor interface for single-record processing.

    Processors accept a ProcessingContext (which contains the input record)
    and yield zero or more output objects. Outputs can be BaseRecord instances
    or any other type (typed data flow).
    """

    async def process(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """Process a single record within a given context.

        Args:
            context: The ProcessingContext containing the input record, session state,
                     observability handles, and shared resources.

        Yields:
            Any: Output objects. Can be zero (filtering), one (1:1),
                 or multiple (1:N expansion). Typically BaseRecord, but can be
                 any type for typed data flow (e.g., structured LLM outputs).
        """
        ...

    async def flush(self) -> AsyncGenerator[Any, None]:
        """Flush any buffered records after source exhaustion.

        Called by the pipeline after the source is exhausted to allow
        processors that buffer (like BatchAccumulator) to process and
        yield any remaining records.

        Default implementation yields nothing. Override in buffering processors.

        Yields:
            Any: Any remaining buffered records/objects after processing.
        """
        ...


@runtime_checkable
class BatchProcessor(Protocol):
    """Processor interface for efficient batch operations.

    Designed for operations that benefit from batching, such as LLM calls,
    embeddings, or bulk API operations. Used inside BatchAccumulator.

    Contexts go in, records come out. Batch processors receive
    ProcessingContexts (which carry variant_params for configuration)
    and return BaseRecords.
    """

    async def process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Process a batch of contexts.

        Args:
            contexts: List of ProcessingContext objects containing records
                and variant_params for configuration.

        Returns:
            List of output records. Can be same length, shorter (filtering),
            or longer (expansion).
        """
        ...
