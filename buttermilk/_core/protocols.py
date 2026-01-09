"""Core protocols for the Unified Processor Architecture.

This module defines the standard interfaces for all processors in the Buttermilk framework.
It separates concerns into:
- Processor: Standard single-record processing (async generator).
- BatchProcessor: Efficient batch processing (e.g., for GPUs).

These protocols enable a unified execution model where "everything is a processor".
"""

from typing import AsyncGenerator, Protocol, runtime_checkable

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord


@runtime_checkable
class Processor(Protocol):
    """Standard processor interface for single-record processing.

    Processors accept a ProcessingContext (which contains the input record)
    and yield zero or more output BaseRecord objects.
    """

    async def process(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a single record within a given context.

        Args:
            context: The ProcessingContext containing the input record, session state,
                     observability handles, and shared resources.

        Yields:
            BaseRecord: Output records. Can be zero (filtering), one (1:1),
                        or multiple (1:N expansion).
        """
        ...

    async def flush(self) -> AsyncGenerator[BaseRecord, None]:
        """Flush any buffered records after source exhaustion.

        Called by the pipeline after the source is exhausted to allow
        processors that buffer (like BatchAccumulator) to process and
        yield any remaining records.

        Default implementation yields nothing. Override in buffering processors.

        Yields:
            BaseRecord: Any remaining buffered records after processing.
        """
        ...


@runtime_checkable
class BatchProcessor(Protocol):
    """Processor interface for efficient batch operations.

    Designed for operations that benefit from batching, such as LLM calls,
    embeddings, or bulk API operations. Used inside BatchAccumulator.

    This is the golden path for batch processing - simple list in, list out.
    """

    async def process_batch(
        self,
        records: list[BaseRecord],
    ) -> list[BaseRecord]:
        """Process a batch of records.

        Args:
            records: List of input records.

        Returns:
            List of output records. Can be same length, shorter (filtering),
            or longer (expansion).
        """
        ...
