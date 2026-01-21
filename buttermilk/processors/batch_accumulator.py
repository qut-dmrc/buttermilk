"""BatchAccumulator - Processor that accumulates records and runs batch operations.

This processor acts as a bridge between the streaming pipeline and batch-oriented
operations. It accumulates records up to a batch_size, runs internal SimpleBatchProcessors
on the batch, then yields individual records back to the streaming pipeline.

Usage:
    ```yaml
    processors:
      - _target_: buttermilk.processors.BatchAccumulator
        batch_size: 50
        batch_processors:
          - _target_: buttermilk.processors.BatchLLMProcessor
            model: gemini-2.5-flash
            template: judge
          - _target_: buttermilk.processors.BatchBQSaver
            table: results
    ```

The BatchAccumulator:
1. Accumulates records until batch_size is reached
2. Runs each SimpleBatchProcessor in sequence on the batch
3. Yields individual records (demux) back to the pipeline
4. On flush(), processes any remaining records in the buffer
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.protocols import BatchProcessor
from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import RecordBufferedException


class BatchAccumulator(ProcessorCore):
    """Accumulates records and processes them in batches.

    This processor bridges streaming and batch processing by:
    - Accumulating records until batch_size is reached
    - Running BatchProcessors on the accumulated batch
    - Yielding individual records back to the pipeline (demux)
    - Processing remaining records on flush()

    Attributes:
        batch_size: Number of records to accumulate before processing
        batch_processors: List of BatchProcessor instances to run on each batch
    """

    batch_size: int = Field(default=50, description="Records to accumulate before batch processing")
    batch_processors: list[Any] = Field(default_factory=list, description="BatchProcessor instances")

    # Internal buffer
    _buffer: list[ProcessingContext] = PrivateAttr(default_factory=list)
    _batch_count: int = PrivateAttr(default=0)
    _lock: asyncio.Lock = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Validate batch_processors implement BatchProcessor."""
        super().model_post_init(__context)
        # Initialize asyncio lock for thread-safe buffer access
        # Required because pipeline concurrency allows multiple tasks to access this processor
        self._lock = asyncio.Lock()
        for i, bp in enumerate(self.batch_processors):
            if not hasattr(bp, "process_batch"):
                raise TypeError(
                    f"batch_processors[{i}] must implement BatchProcessor protocol "
                    f"(missing process_batch method): {type(bp)}"
                )

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Accumulate record; process and yield when batch is full.

        Args:
            context: Processing context containing the record

        Yields:
            BaseRecord: Individual records after batch processing (when batch is full)
        """
        # Use lock to protect buffer operations from concurrent access
        # This prevents race conditions where multiple tasks see len >= batch_size
        # and process the same batch multiple times
        batch_to_process: list[ProcessingContext] | None = None

        async with self._lock:
            self._buffer.append(context)

            if len(self._buffer) >= self.batch_size:
                # Take ownership of buffer contents and reset
                # Other concurrent tasks will append to fresh buffer
                batch_to_process = self._buffer
                self._buffer = []

        if batch_to_process is not None:
            # Process outside lock to allow concurrent buffer appends
            async for record in self._process_batch_from_contexts(batch_to_process):
                yield record
        else:
            # Record is buffered, not filtered - raise distinct exception so pipeline
            # logs "buffered" instead of "skipped/filtered"
            raise RecordBufferedException(
                f"Record buffered in BatchAccumulator (buffer size: {len(self._buffer)}/{self.batch_size})"
            )

    async def _process_batch_from_contexts(
        self, contexts: list[ProcessingContext]
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a batch of contexts through batch processors and yield results.

        Args:
            contexts: List of ProcessingContext objects to process

        Yields:
            BaseRecord: Individual records after batch processing
        """
        if not contexts:
            return

        self._batch_count += 1
        batch_num = self._batch_count
        input_count = len(contexts)

        logger.info(
            f"BatchAccumulator processing batch {batch_num}",
            batch_size=input_count,
            processor_count=len(self.batch_processors),
        )

        # Extract records from contexts
        records = [ctx.record for ctx in contexts]

        # Run each batch processor in sequence
        for i, bp in enumerate(self.batch_processors):
            processor_name = getattr(bp, "name", None) or type(bp).__name__
            logger.debug(
                f"Running batch processor {i + 1}/{len(self.batch_processors)}: {processor_name}",
                input_count=len(records),
            )

            try:
                records = await bp.process_batch(records)
            except Exception as e:
                logger.error(
                    f"Batch processor {processor_name} failed",
                    batch_num=batch_num,
                    error=str(e),
                )
                raise

        logger.info(
            f"BatchAccumulator batch {batch_num} complete",
            input_count=input_count,
            output_count=len(records),
        )

        # Demux: yield individual records
        for record in records:
            yield record

    async def flush(self) -> AsyncGenerator[BaseRecord, None]:
        """Process any remaining buffered records.

        Called by the pipeline after source exhaustion.

        Yields:
            BaseRecord: Individual records from the remaining batch
        """
        # Take ownership of remaining buffer under lock
        batch_to_process: list[ProcessingContext] | None = None

        async with self._lock:
            if self._buffer:
                logger.info(
                    f"BatchAccumulator flushing remaining {len(self._buffer)} records",
                )
                batch_to_process = self._buffer
                self._buffer = []

        if batch_to_process:
            async for record in self._process_batch_from_contexts(batch_to_process):
                yield record

    async def finalize_processing(self) -> bool:
        """Finalize batch processors if they support it.

        Returns:
            bool: True if finalization succeeded
        """
        for bp in self.batch_processors:
            if hasattr(bp, "finalize"):
                try:
                    await bp.finalize()
                except Exception as e:
                    logger.error(f"Batch processor finalize failed: {e}")
                    return False
        return True
