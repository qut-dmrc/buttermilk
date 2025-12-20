"""Unified Pipeline Executor.

This module implements the PipelineExecutor, responsible for orchestrating the execution
of a unified processing pipeline. It handles:
- Instantiating processors from configuration.
- Routing records through the processor chain.
- Managing batching for BatchProcessors.
- Handling lifecycle events (setup, teardown).
"""

import asyncio
from typing import AsyncGenerator, Any

from buttermilk._core.pipeline_config import PipelineConfig
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import ProcessorConfig
from buttermilk._core.processor_registry import create_processor
from buttermilk._core.protocols import Processor, BatchProcessor
from buttermilk._core.types import BaseRecord
from buttermilk._core.unified_processor import UnifiedProcessor
from buttermilk.processors import unified_processors  # Import to trigger registration
from buttermilk import logger

class PipelineExecutor:
    """Executes a pipeline of unified processors."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processors: list[Processor | BatchProcessor] = []
        self._batch_buffers: dict[int, list[ProcessingContext]] = {}
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Instantiate processors based on configuration."""
        for proc_config in self.config.processors:
            processor = self._create_processor(proc_config)
            self.processors.append(processor)

            # Initialize batch buffer for BatchProcessors
            if isinstance(processor, BatchProcessor):
                self._batch_buffers[len(self.processors) - 1] = []

    def _create_processor(self, config: ProcessorConfig) -> Processor | BatchProcessor:
        """Factory method to create processor instances.

        Uses the processor registry for dynamic loading.

        Args:
            config: Processor configuration

        Returns:
            Instantiated processor

        Raises:
            KeyError: If processor type is not registered
        """
        return create_processor(config)

    async def run(
        self,
        source: AsyncGenerator[BaseRecord, None],
        session_id: str
    ) -> AsyncGenerator[BaseRecord, None]:
        """Run the pipeline on a source of records.

        Args:
            source: Async generator yielding BaseRecord objects
            session_id: Session identifier for this pipeline run

        Yields:
            BaseRecord: Processed records from the pipeline
        """
        async for record in source:
            # Create context for this record
            context = ProcessingContext(session_id=session_id, record=record)

            # Process through the chain
            async for result in self._process_chain(context, 0):
                yield result

        # Flush any remaining buffered records in batch processors
        async for result in self._flush_all_batches(0):
            yield result

    async def _process_chain(
        self,
        context: ProcessingContext,
        processor_index: int
    ) -> AsyncGenerator[BaseRecord, None]:
        """Recursively process the chain.

        Args:
            context: Current processing context
            processor_index: Index of the current processor in the chain

        Yields:
            BaseRecord: Output records from the chain
        """
        if processor_index >= len(self.processors):
            yield context.record
            return

        processor = self.processors[processor_index]

        if isinstance(processor, Processor):
            async for output_record in processor.process(context):
                # Create new context for the next stage (preserving session info)
                next_context = ProcessingContext(
                    session_id=context.session_id,
                    record=output_record,
                    metadata=context.metadata.copy(),  # Copy to avoid mutation
                    resources=context.resources,
                    ui_callback=context.ui_callback
                )

                async for final_output in self._process_chain(next_context, processor_index + 1):
                    yield final_output

        elif isinstance(processor, BatchProcessor):
            # Buffer the context
            self._batch_buffers[processor_index].append(context)

            # Check if we've reached batch_size
            if len(self._batch_buffers[processor_index]) >= processor.config.batch_size:
                # Process the buffered batch
                async for result in self._flush_batch(processor_index):
                    yield result

    async def _flush_batch(self, processor_index: int) -> AsyncGenerator[BaseRecord, None]:
        """Flush a single batch processor's buffer.

        Args:
            processor_index: Index of the batch processor to flush

        Yields:
            BaseRecord: Output records from processing the batch
        """
        if processor_index not in self._batch_buffers:
            return

        buffer = self._batch_buffers[processor_index]
        if not buffer:
            return

        processor = self.processors[processor_index]

        # Process the batch
        async for output_batch in processor.process_batch(buffer):
            for output_record in output_batch:
                # Create new context for the next stage
                # Use the session_id from the first context in the batch
                next_context = ProcessingContext(
                    session_id=buffer[0].session_id,
                    record=output_record,
                    metadata={},  # Fresh metadata for batch output
                    resources=buffer[0].resources,
                    ui_callback=buffer[0].ui_callback
                )

                async for final_output in self._process_chain(next_context, processor_index + 1):
                    yield final_output

        # Clear the buffer after processing
        self._batch_buffers[processor_index] = []

    async def _flush_all_batches(self, start_index: int) -> AsyncGenerator[BaseRecord, None]:
        """Flush all remaining batches in the pipeline starting from start_index.

        Args:
            start_index: Starting processor index to flush from

        Yields:
            BaseRecord: Output records from flushing all batches
        """
        for processor_index in range(start_index, len(self.processors)):
            if processor_index in self._batch_buffers:
                async for result in self._flush_batch(processor_index):
                    yield result
