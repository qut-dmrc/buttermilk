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
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Instantiate processors based on configuration."""
        for proc_config in self.config.processors:
            processor = self._create_processor(proc_config)
            self.processors.append(processor)

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
        """Run the pipeline on a source of records."""
        
        # This is a simplified execution model (no batching support yet)
        async for record in source:
            # Create context for this record
            context = ProcessingContext(session_id=session_id, record=record)
            
            # Process through the chain
            async for result in self._process_chain(context, 0):
                yield result

    async def _process_chain(
        self, 
        context: ProcessingContext, 
        processor_index: int
    ) -> AsyncGenerator[BaseRecord, None]:
        """Recursively process the chain."""
        if processor_index >= len(self.processors):
            yield context.record
            return

        processor = self.processors[processor_index]
        
        if isinstance(processor, Processor):
            async for output_record in processor.process(context):
                # Create new context for the next stage (preserving session info)
                # Note: shallow copy might be safer, or explicitly copying fields
                next_context = ProcessingContext(
                    session_id=context.session_id,
                    record=output_record,
                    metadata=context.metadata, # Share metadata? Or copy?
                    resources=context.resources,
                    ui_callback=context.ui_callback
                )
                
                async for final_output in self._process_chain(next_context, processor_index + 1):
                    yield final_output
        
        elif isinstance(processor, BatchProcessor):
             # TODO: Implement batching logic
             # This requires buffering and is complex for a simple recursive generator
             logger.warning("BatchProcessor not yet supported in simple executor")
             pass
