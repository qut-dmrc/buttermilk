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
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore
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
            if not (hasattr(bp, "process_batch") or hasattr(bp, "process")):
                raise TypeError(f"batch_processors[{i}] must implement BatchProcessor (process_batch) or Processor (process) protocol: {type(bp)}")

    def _transfer_variant_params(self, context: ProcessingContext) -> None:
        """Transfer _variant_params from record.metadata to context.variant_params.

        Pops _variant_params from metadata to keep records clean.
        This bridges ParameterExpansionProcessor (which writes to metadata)
        with batch processors (which read from context.variant_params).
        """
        if context.record.metadata and "_variant_params" in context.record.metadata:
            variant_params = context.record.metadata.pop("_variant_params")
            if isinstance(variant_params, dict):
                context.variant_params = variant_params

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
        # Transfer variant params from record metadata to context
        self._transfer_variant_params(context)

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
            raise RecordBufferedException(f"Record buffered in BatchAccumulator (buffer size: {len(self._buffer)}/{self.batch_size})")

    async def _process_batch_from_contexts(self, contexts: list[ProcessingContext]) -> AsyncGenerator[BaseRecord, None]:
        """Process a batch of contexts through batch processors and yield results.

        The first batch processor receives the original contexts (with variant_params).
        Subsequent batch processors receive output records wrapped in minimal contexts.

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

        # Fan-out: each batch processor gets the ORIGINAL contexts independently.
        # This is correct for multi-model pipelines where each processor handles
        # the same records (e.g., deepseek-ai/deepseek-r1-0528-maas, meta/llama-4-maverick-17b-128e-instruct-maas, gpt-5-mini all
        # process the same expanded records with the same variant_params).
        all_output_records: list[BaseRecord] = []

        for i, bp in enumerate(self.batch_processors):
            processor_name = getattr(bp, "name", None) or type(bp).__name__
            is_batch = hasattr(bp, "process_batch")
            logger.debug(
                f"Running {'batch' if is_batch else 'live'} processor {i + 1}/{len(self.batch_processors)}: {processor_name}",
                input_count=len(contexts),
            )

            try:
                if is_batch:
                    output_records = await bp.process_batch(contexts)
                    all_output_records.extend(output_records)
                else:
                    # ProcessorCore: call process() per context concurrently
                    live_records = await self._run_live_processor(bp, contexts)
                    all_output_records.extend(live_records)
            except Exception as e:
                logger.error(
                    f"{'Batch' if is_batch else 'Live'} processor {processor_name} failed, continuing with remaining processors",
                    batch_num=batch_num,
                    error=str(e),
                )
                # Fan-out: continue with remaining processors even if one fails.
                # Each processor is independent (different model), so one failure
                # shouldn't block the others.

        logger.info(
            f"BatchAccumulator batch {batch_num} complete",
            input_count=input_count,
            output_count=len(all_output_records),
        )

        # Demux: yield individual records
        for record in all_output_records:
            yield record

    async def _run_live_processor(self, processor: Any, contexts: list[ProcessingContext]) -> list[BaseRecord]:
        """Run a live (ProcessorCore) processor across all contexts concurrently.

        Uses asyncio.gather for throughput — each context is processed independently.

        Args:
            processor: A ProcessorCore instance with a process() method
            contexts: List of ProcessingContext objects to process

        Returns:
            list[BaseRecord]: Collected output records from all contexts
        """

        async def _process_one(ctx: ProcessingContext) -> list[BaseRecord]:
            records: list[BaseRecord] = []
            async for record in processor.process(ctx):
                records.append(record)
            return records

        results = await asyncio.gather(
            *[_process_one(ctx) for ctx in contexts],
            return_exceptions=True,
        )

        output_records: list[BaseRecord] = []
        processor_name = getattr(processor, "name", None) or type(processor).__name__
        for ctx, result in zip(contexts, results):
            if isinstance(result, Exception):
                record_id = getattr(ctx.record, "record_id", "unknown")
                logger.error(
                    f"Live processor {processor_name} failed for record {record_id}",
                    error=str(result),
                )
                # Continue processing other records (fan-out semantics)
            else:
                output_records.extend(result)

        return output_records

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
