"""Simple async pipeline for processing records.

This module provides a minimal pipeline interface that chains async processors
together, following the pattern: data_source → processor1 → processor2 → ... → processorN.
"""

from typing import Protocol, AsyncGenerator, Optional
from buttermilk._core.types import BaseRecord


class Processor(Protocol):
    """Simple processor that transforms one record."""

    async def process(self, record: BaseRecord) -> BaseRecord | None:
        """Process a record. Return None to filter out."""
        ...


class DataSource(Protocol):
    """Simple data source that yields records."""

    def __call__(self) -> AsyncGenerator[BaseRecord, None]:
        """Return async generator of records."""
        ...


async def run_simple_pipeline(
    data_source: DataSource,
    processors: list[Processor]
) -> None:
    """Run simple pipeline: source → processors (in sequence).

    Args:
        data_source: Async generator that yields records
        processors: List of processors to apply in sequence (including any uploaders)
    """
    async for record in data_source():
        # Run through processor chain
        current_record = record
        for processor in processors:
            current_record = await processor.process(current_record)
            if current_record is None:  # Filtered out
                break