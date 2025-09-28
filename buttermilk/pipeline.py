"""Async pipeline orchestrator extracted and simplified from vector.py.

This module provides a concurrent async pipeline orchestrator that processes
records through stages, tracking metadata and errors without complex result objects.
"""

import asyncio
import time
from typing import Any, AsyncGenerator, AsyncIterator, Mapping, Optional, Protocol, runtime_checkable

import hydra
import pydantic
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk import bm, logger


class RecordSkippedException(Exception):
    """Exception raised when a record is intentionally skipped/filtered."""
    pass


@runtime_checkable
class Processor(Protocol):
    """Standard processor interface - async generator that yields record dictionaries."""

    async def process(self, inputs: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Process inputs dictionary and yield zero or more output dictionaries.

        Args:
            inputs: Dictionary containing 'record' and potentially other fields

        Yields:
            Dictionary containing processed data, typically with 'record' key

        Yield nothing to filter out the record.
        Yield one dict for 1:1 transformation.
        Yield multiple dicts for 1:N transformation.
        """
        ...


class PipelineOrchestrator(BaseModel):
    """Concurrent async pipeline orchestrator for processing record dictionaries through stages.

    Processes dictionaries in the format {"record": BaseRecord, ...} through processor chains.
    Each stage appends its status to record.metadata[stage_name].
    """

    concurrency: int = Field(default=1, description="Max concurrent record processing")
    max_records: Optional[int] = Field(default=None, description="Maximum records to process")
    stage_name: str = Field(..., description="Name for this processing stage")
    force_reprocess: bool = Field(default=False, description="Ignore cache and reprocess")
    enable_record_cache: bool = Field(default=True, description="Enable per-stage Record caching")

    # Inputs configured after instantiation
    source: Optional[Any] = Field(default=None, exclude=True, description="Source config or AsyncIterator")
    processors: list[Any] = Field(default_factory=list, exclude=True)  # List of processors to chain

    # Internal state
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _record_cache: Any = PrivateAttr(default=None)
    _attempted: int = PrivateAttr(default=0)
    _processed: int = PrivateAttr(default=0)
    _skipped: int = PrivateAttr(default=0)
    _failed: int = PrivateAttr(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _instantiate_components(cls, values: dict) -> dict:
        """Automatically instantiate source and processors from config."""
        if "source" in values and isinstance(values["source"], Mapping):
            # Instantiate source if it's a DictConfig or dict
            values["source"] = bm.get_storage(values.get("source"))

        values["processors"] = [hydra.utils.instantiate(p) if isinstance(p, DictConfig) else p for p in values.get("processors", [])]
        return values

    @pydantic.model_validator(mode="after")
    def _init(self):
        """Initialize semaphore and record cache."""
        self._semaphore = asyncio.Semaphore(self.concurrency)

        # Initialize record cache
        if self.enable_record_cache:
            try:
                from buttermilk._core.record_cache import RecordCache
                self._record_cache = RecordCache()
            except Exception:
                logger.warning("Failed to initialize RecordCache, continuing without caching")
                self._record_cache = None

        return self

    async def _process_single_record(self, inputs: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Process a single inputs dict through the entire processor chain.

        Properly handles 1:N transformations where processors can yield multiple outputs.
        Each output flows through all remaining processors in the chain.

        Args:
            inputs: Input dictionary containing 'record' and other fields

        Yields:
            Final processed outputs dicts with metadata (can be multiple for 1:N transformations)

        Raises:
            RecordSkippedException: If record is filtered out by any processor
            Exception: If any processor fails
        """
        if not self.processors:
            raise ValueError(f"[{self.stage_name}] No processors configured")

        # Extract record for cache operations
        record = inputs.get("record")
        if not record or not hasattr(record, "record_id"):
            # Can't cache without a record_id, process normally
            async for result in self._process_without_cache(inputs):
                yield result
            return

        # Check cache first
        if self.enable_record_cache and self._record_cache and not self.force_reprocess:
            cached_record = self._record_cache.load(record.record_id, self.stage_name)
            if cached_record and self._validate_cached_record(cached_record):
                logger.debug(f"⚡ Cache hit for record {record.record_id} at stage '{self.stage_name}' – skipping processing")

                # Return cached record with updated metadata
                cached_inputs = inputs.copy()
                cached_inputs["record"] = cached_record
                yield cached_inputs
                return

        # Process normally and cache results
        processed_results = []
        async for processed_inputs in self._process_without_cache(inputs):
            processed_results.append(processed_inputs)
            yield processed_inputs

        # Cache the processed record(s) - for 1:N transformations, cache the last record
        if (self.enable_record_cache and self._record_cache and processed_results):
            try:
                # For 1:N transformations, cache the last processed record
                last_result = processed_results[-1]
                if "record" in last_result and last_result["record"]:
                    processed_record = last_result["record"]
                    # Determine if we should include chunks (for vector processing compatibility)
                    include_chunks = bool(getattr(processed_record, "chunks", None))
                    self._record_cache.save(processed_record, self.stage_name, include_chunks=include_chunks)
            except Exception as ce:
                logger.debug(f"Cache save failed {record.record_id} @ {self.stage_name}: {ce}")

    def _validate_cached_record(self, cached_record) -> bool:
        """Validate that a cached record is still usable.

        Override this method for custom validation logic.
        """
        return cached_record is not None

    async def _process_without_cache(self, inputs: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Process inputs through the processor chain without caching."""
        start_time = time.time()
        current_inputs = inputs

        # Log processing start
        record = inputs.get("record")
        if record and hasattr(record, "record_id"):
            record_title = getattr(record, "title", "Unknown")[:50] if hasattr(record, "title") else "Unknown"
            logger.info(f"🔷 [{self.stage_name}-{record.record_id}] Processing record '{record_title}'")

        try:
            # Start with the input as a single item in a processing queue
            processing_queue = [current_inputs]

            # Flow inputs through each processor in sequence
            for processor in self.processors:
                next_queue = []

                # Process each item in the current queue through this processor
                for item in processing_queue:
                    outputs = []
                    async for output_dict in processor.process(item):
                        outputs.append(output_dict)

                    if not outputs:
                        # This item was filtered out by this processor
                        record = item.get("record")
                        if record:
                            skipped_metadata = {
                                **record.metadata,
                                self.stage_name: {
                                    "status": "skipped",
                                    "timestamp": time.time(),
                                    "reason": f"filtered_by_{getattr(processor, '__name__', 'processor')}",
                                }
                            }
                            skipped_record = record.model_copy(update={"metadata": skipped_metadata})
                            item["record"] = skipped_record

                        # Raise exception to indicate this entire input was skipped
                        raise RecordSkippedException("Record was filtered out by processor")
                    else:
                        # Add all outputs to the next processing queue
                        next_queue.extend(outputs)

                # Move to next stage with all outputs from this processor
                processing_queue = next_queue

            # If we reach here, processing_queue contains all final outputs
            if not processing_queue:
                # Everything was filtered out (shouldn't happen due to exception above)
                raise RecordSkippedException("All outputs were filtered")

            # Add success metadata to all final outputs and yield them
            for final_inputs in processing_queue:
                if "record" in final_inputs:
                    record = final_inputs["record"]
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    final_metadata = {
                        **record.metadata,
                        self.stage_name: {
                            "status": "processed",
                            "timestamp": time.time(),
                            "processing_time_ms": processing_time_ms,
                        }
                    }
                    updated_record = record.model_copy(update={"metadata": final_metadata})
                    final_inputs["record"] = updated_record

                yield final_inputs

        except Exception as e:
            # Let exception bubble up - TaskGroup will handle error collection
            logger.error(f"Error processing record in stage {self.stage_name}: {e}")
            raise

    async def __call__(self) -> AsyncIterator[dict[str, Any]]:
        """Process inputs dicts from source with TaskGroup-based concurrency.

        Each inputs dict flows through the entire processor chain individually.
        Uses TaskGroup for natural error collection and concurrency management.
        """
        if self.source is None:
            logger.error(f"[{self.stage_name}] No source iterator configured")
            return

        log_interval = 15.0
        last_log = time.monotonic()

        def maybe_log_status(pending_count: int):
            nonlocal last_log
            now = time.monotonic()
            if now - last_log >= log_interval:
                logger.debug(
                    f"📊 Stage '{self.stage_name}': attempted={self._attempted} processed={self._processed} "
                    f"skipped={self._skipped} failed={self._failed} pending={pending_count}"
                )
                last_log = now

        try:
            # Ensure we have an async iterator
            source_iter = self.source if hasattr(self.source, "__anext__") else self.source.__aiter__()

            pending_tasks: set[asyncio.Task] = set()
            completed_inputs = asyncio.Queue()

            async def process_and_queue(inputs: dict[str, Any]):
                """Process inputs dict and put all results in queue."""
                try:
                    results_count = 0
                    async for processed_inputs in self._process_single_record(inputs):
                        await completed_inputs.put(("success", processed_inputs))
                        results_count += 1

                    # Only count as processed if we got at least one output
                    if results_count > 0:
                        self._processed += 1
                except RecordSkippedException as e:
                    self._skipped += 1
                    # Record was intentionally skipped, no error logging needed
                    record = inputs.get("record")
                    record_id = getattr(record, "record_id", "unknown") if record else "unknown"
                    logger.debug(f"Record {record_id} skipped in stage {self.stage_name}: {e}")

                    # The metadata was already added in _process_without_cache
                    # Just put the skipped record in queue
                    await completed_inputs.put(("skipped", inputs))

                except Exception as e:
                    self._failed += 1
                    # Log error but don't stop processing other records
                    record = inputs.get("record")
                    record_id = getattr(record, "record_id", "unknown") if record else "unknown"
                    logger.error(f"Error processing record {record_id} in stage {self.stage_name}: {e}")

                    # Add error metadata to record and put in queue
                    if record:
                        error_metadata = {
                            **record.metadata,
                            self.stage_name: {
                                "status": "failed",
                                "timestamp": time.time(),
                                "error": str(e),
                                "error_type": type(e).__name__,
                            }
                        }
                        failed_record = record.model_copy(update={"metadata": error_metadata})
                        failed_inputs = inputs.copy()
                        failed_inputs["record"] = failed_record
                        await completed_inputs.put(("error", failed_inputs))
                    # Continue processing other records - don't raise

            try:
                # Producer: Create tasks for incoming inputs dicts
                async def producer():
                    nonlocal pending_tasks
                    async for inputs in source_iter:
                        self._attempted += 1

                        # Check if we've hit max_records
                        if self.max_records is not None and self._processed >= self.max_records:
                            logger.info(f"🔚 Stage '{self.stage_name}' reached max_records ({self._processed}) – stopping")
                            break

                        # Maintain concurrency limit
                        while len(pending_tasks) >= self.concurrency:
                            await asyncio.sleep(0.01)  # Brief pause to allow task completion
                            # Clean up completed tasks
                            pending_tasks = {t for t in pending_tasks if not t.done()}

                        # Create and track task
                        task = asyncio.create_task(process_and_queue(inputs))
                        pending_tasks.add(task)

                        maybe_log_status(len(pending_tasks))

                    # Signal completion by putting None
                    await completed_inputs.put(None)

                # Consumer: Yield completed inputs dicts
                async def consumer():
                    while True:
                        result = await completed_inputs.get()
                        if result is None:  # End signal
                            break

                        status, inputs = result
                        if status == "success":
                            yield inputs
                        elif status == "skipped":
                            # Optionally yield skipped records too (for debugging/tracking)
                            # For now, we'll skip them and just log
                            pass
                        elif status == "error":
                            # Optionally yield failed records too (for debugging/recovery)
                            # For now, we'll skip them and just log
                            pass

                # Start producer
                producer_task = asyncio.create_task(producer())

                # Yield from consumer
                async for inputs in consumer():
                    yield inputs

                # Wait for producer to finish
                await producer_task

                # Wait for all remaining tasks to complete
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"Pipeline error in stage '{self.stage_name}': {e}")
                # Cancel all pending tasks
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                # Wait for cancellations to complete
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                raise

            logger.info(
                f"✅ Stage '{self.stage_name}' complete: attempted={self._attempted} "
                f"processed={self._processed} skipped={self._skipped} failed={self._failed}"
            )


def chain_stages(*stages: PipelineOrchestrator) -> AsyncIterator[dict[str, Any]]:
    """Chain multiple pipeline stages together.

    Each stage's output becomes the next stage's input.

    Args:
        *stages: Variable number of PipelineOrchestrator instances

    Returns:
        Async iterator of final processed inputs dicts

    Example:
        ```python
        source = storage(batch_size=100)

        stage1 = PipelineOrchestrator(
            stage_name="enrich",
            source=source,
            processor=enrich_func
        )

        stage2 = PipelineOrchestrator(
            stage_name="validate",
            source=stage1(),
            processor=validate_func
        )

        async for record in stage2():
            print(record)
        ```
    """
    if not stages:
        raise ValueError("At least one stage required")

    # Chain stages by connecting outputs to inputs
    for i in range(1, len(stages)):
        stages[i].source = stages[i - 1]()

    # Return the final stage's iterator
    return stages[-1]()
