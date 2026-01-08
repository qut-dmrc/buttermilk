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


@runtime_checkable
class BatchProcessor(Protocol):
    """Processor interface for efficient batch operations.

    Designed for operations that benefit from batching, such as GPU inference
    (LLMs, embeddings) or bulk API operations.
    """

    async def process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a batch of records.

        Args:
            contexts: A list of ProcessingContext objects, each containing a record.

        Yields:
            BaseRecord: Output records corresponding to the batch.
                        The processor handles mapping outputs to requests.

            Note: The executor is responsible for routing these records to the next stage.

            Note: The executor is responsible for mapping outputs back to inputs
            if order is preserved, or the processor must handle lineage.
        """
        ...

    async def finalize(self) -> None:
        """Optional cleanup or finalization logic."""
        ...
