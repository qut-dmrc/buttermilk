"""Simple async pipeline for processing records.

This module provides a minimal pipeline interface that chains async processors
together, following the pattern: data_source → processor1 → processor2 → ... → processorN.
"""

from typing import AsyncGenerator, Protocol

from buttermilk._core.types import BaseRecord


class Processor(Protocol):
    """Simple processor that transforms one record."""

    async def process(self, record: BaseRecord) -> BaseRecord | None:
        """Process a record. Return None to filter out."""
        ...


class MultiProcessor(Protocol):
    """Processor that can yield multiple output records for each input."""

    async def process(self, record: BaseRecord) -> AsyncGenerator[BaseRecord, None]:
        """Process a record and yield zero or more output records.

        Args:
            record: Input record to process

        Yields:
            Output records (can be different type than input)
        """
        ...


class DataSource(Protocol):
    """Simple data source that yields records."""

    def __call__(self) -> AsyncGenerator[BaseRecord, None]:
        """Return async generator of records."""
        ...


async def run_simple_pipeline(
    data_source: DataSource,
    processors: list[Processor | MultiProcessor]
) -> None:
    """Run simple pipeline: source → processors (in sequence).

    Supports both Processor (1:1) and MultiProcessor (1:N) types.
    When a MultiProcessor is encountered, its output records are each
    passed through the remaining processors in the chain.

    Args:
        data_source: Async generator that yields records
        processors: List of processors to apply in sequence (can be Processor or MultiProcessor)
    """
    async for record in data_source():
        await _process_record_chain([record], processors, 0)


async def _process_record_chain(
    records: list[BaseRecord],
    processors: list[Processor | MultiProcessor],
    processor_index: int
) -> None:
    """Process records through the remaining processor chain.

    Args:
        records: Current records to process
        processors: Full list of processors
        processor_index: Index of next processor to apply
    """
    if processor_index >= len(processors) or not records:
        return

    processor = processors[processor_index]
    next_records = []

    for record in records:
        # Check if this is a MultiProcessor by attempting to iterate its result
        result = processor.process(record)

        # Try to iterate - if it's an async generator (MultiProcessor), collect all outputs
        try:
            # If it's an async generator, iterate through it
            if hasattr(result, '__aiter__'):
                async for output_record in result:
                    if output_record is not None:
                        next_records.append(output_record)
            else:
                # It's a regular Processor, await the result
                output_record = await result
                if output_record is not None:
                    next_records.append(output_record)
        except TypeError:
            # Fallback: treat as regular Processor
            output_record = await result
            if output_record is not None:
                next_records.append(output_record)

    # Process next stage with all collected records
    if next_records:
        await _process_record_chain(next_records, processors, processor_index + 1)
