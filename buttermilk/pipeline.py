"""Async pipeline orchestrator.

This module provides a concurrent async pipeline orchestrator that processes
records through stages with caching and tracking metadata.

PIPELINE DESIGN

1. **Processor Agnosticism**: The pipeline orchestrator must not have special
    knowledge of any processor type. This enables extensibility without
    modifying core pipeline code.

    **Implications**:
    - No special loading paths for specific processors (e.g., ChromaDB)
    - All processor-specific logic lives in the processor itself
    - Pipeline only knows about the Processor Protocol

    **Anti-patterns**:
    ❌ if isinstance(processor, ChromaDBUploader): special_logic()
    ❌ Cache layer knowing about ChunkedDocument types
    ✅ Processors handle their own type conversions


2. Processors must acccept a BaseRecord and yield zero or more BaseRecord objects.

    **Error handling**: Processors must raise errors for any failures. The pipeline
    will catch, count, and log errors but continue processing other records.


3. Sources are async iterators that yield BaseRecord objects to be processed by the pipeline.

    **Requirements**:
    - Must implement `__aiter__()` returning an AsyncIterator[BaseRecord]
    - Must yield BaseRecord objects (or subclasses like Record)
    - Should handle errors internally and either:
    - Silently skip failed records (for optional/best-effort sources)
    - Yield error records with metadata indicating failure (for auditable sources)
    - Raise exceptions to halt pipeline (for critical failures)

    **Filtering Pattern**:
    Sources should use the `RecordFilter` protocol from `buttermilk.storage.base` for
    consistency. This allows reusable filters like existence checks, date ranges, etc.

    **Best Practices**:
    - Keep sources simple - they should only fetch and yield records
    - Use processors for expensive operations (downloads, extraction, transformation)
    - Apply filters at source level to minimize unnecessary processing
    - Return records in a deterministic order for incremental sync support


NON-GOALS
- Pipeline will NOT validate processor-specific data shapes
- Pipeline will NOT handle type conversions for processors
- Pipeline will NOT have processor-specific optimizations

"""

import asyncio
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Mapping,
    Optional,
)

import hydra
import pydantic

# weave import removed
from omegaconf import DictConfig
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk import bm, logger
from buttermilk._core.hashing import compute_processor_config_hash
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord


class RecordSkippedException(Exception):
    """Exception raised when a record is intentionally skipped/filtered."""

    pass


class PipelineOrchestrator(BaseModel):
    """Concurrent async pipeline orchestrator for processing records through processor chains.

    Processes BaseRecord objects through a sequence of processors.
    Each pipeline appends its status to record.metadata[pipeline_name].

    Terminology:
    - Pipeline: The entire processing flow (e.g., 'osb_vectorstore_pipeline')
    - Processor: Individual processing step (e.g., LLMCore, SemanticSplitter)
    - Processor ID: Unique identifier within pipeline (e.g., 'osb_vectorstore_pipeline.03.ChromaDBUploader')
    """

    concurrency: int = Field(default=1, description="Max concurrent record processing")
    api_concurrency: int = Field(
        default=10,
        description="Max concurrent API calls across all processors (limits nested parallelism)",
    )
    limit: Optional[int] | None = Field(
        default=None, description="Maximum records to process"
    )
    pipeline_name: str = Field(..., description="Name for this processing pipeline")
    force_reprocess: bool = Field(
        default=False, description="Ignore cache and reprocess"
    )
    enable_record_cache: bool = Field(
        default=True, description="Enable per-processor Record caching"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Base directory for record cache (defaults to ~/.cache/buttermilk)",
    )
    num_runs: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate each source record (for reliability studies)",
    )

    # Inputs configured after instantiation
    source: Optional[Any] = Field(
        default=None, exclude=True, description="Source config or AsyncIterator"
    )
    processors: list[Any] = Field(
        default_factory=list, exclude=True
    )  # List of processors to chain

    # Internal state
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _api_semaphore: asyncio.Semaphore = PrivateAttr()
    _record_cache: Any = PrivateAttr(default=None)
    _summary: Any = PrivateAttr(default=None)  # ProcessingSummary instance

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _instantiate_components(cls, values: dict) -> dict:
        """Automatically instantiate source and processors from config."""
        if "source" in values and isinstance(values["source"], Mapping):
            # Instantiate source if it's a DictConfig or dict
            values["source"] = bm.get_storage(values.get("source"))

        values["processors"] = [
            hydra.utils.instantiate(p) if isinstance(p, DictConfig) else p
            for p in values.get("processors", [])
        ]
        return values

    @pydantic.model_validator(mode="after")
    def _init(self):
        """Initialize semaphores, record cache, and processing summary."""
        from buttermilk._core.context import set_api_semaphore
        from buttermilk._core.types import ProcessingSummary

        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._api_semaphore = asyncio.Semaphore(self.api_concurrency)
        set_api_semaphore(self._api_semaphore)
        self._summary = ProcessingSummary()

        # Initialize record cache (lazy base_dir resolution happens in RecordCache)
        if self.enable_record_cache:
            try:
                from buttermilk._core.record_cache import RecordCache

                # RecordCache will lazily resolve base_dir from bm.session_info.cache_dir if not provided
                self._record_cache = RecordCache(base_dir=self.cache_dir)
                logger.info(
                    "🗂️  RecordCache created",
                    pipeline_name=self.pipeline_name,
                    cache_dir_param=self.cache_dir,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize RecordCache, continuing without caching",
                    error=str(e),
                )
                self._record_cache = None
        else:
            logger.info(
                "🚫 Record cache disabled for pipeline stage",
                pipeline_name=self.pipeline_name,
            )

        # Wrap source with LimitingSource FIRST (before replication)
        # This ensures limit applies to original records, not replicated ones
        if self.limit is not None and self.source is not None:
            from buttermilk.storage.limiting_source import LimitingSource

            self.source = LimitingSource(self.source, limit=self.limit)
            logger.info(
                f"🔢 Source wrapped with LimitingSource (limit={self.limit})",
                pipeline_name=self.pipeline_name,
                limit=self.limit,
            )

        # Wrap source with ReplicatingSource if num_runs > 1 (AFTER limiting)
        if self.num_runs > 1 and self.source is not None:
            from buttermilk.storage.replicating_source import ReplicatingSource

            self.source = ReplicatingSource(self.source, num_runs=self.num_runs)
            logger.info(
                f"🔄 Source wrapped with ReplicatingSource (num_runs={self.num_runs})",
                pipeline_name=self.pipeline_name,
                num_runs=self.num_runs,
            )

        return self

    async def _process_single_record(
        self, record: BaseRecord
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a single BaseRecord through the entire processor chain.

        Properly handles 1:N transformations where processors can yield multiple outputs.
        Each output flows through all remaining processors in the chain.

        Args:
            record: BaseRecord to process

        Yields:
            Final processed BaseRecord objects (can be multiple for 1:N transformations)

        Raises:
            RecordSkippedException: If record is filtered out by any processor
            Exception: If any processor fails
        """
        if not self.processors:
            raise ValueError(f"[{self.pipeline_name}] No processors configured")

        tracer = trace.get_tracer("buttermilk.pipeline")
        record_id = getattr(record, "record_id", "unknown")
        title = getattr(record, "title", None)
        record_title = (
            (title[:50] if title else "Unknown")
            if hasattr(record, "title")
            else "Unknown"
        )

        # Build span attributes for record processing
        span_attributes = {
            "record.id": record_id,
            "record.title": record_title,
            "stage.name": self.pipeline_name,
            "stage.processor_count": len(self.processors),
        }

        with tracer.start_as_current_span(
            "pipeline.process_record", attributes=span_attributes
        ) as span:
            try:
                # Per-processor caching is now handled in _process_without_cache

                # Process using per-processor caching
                span.set_attribute("cache.hit", False)

                # Process through processor chain with per-processor caching
                processed_results = []
                async for processed_record in self._process_without_cache(record):
                    processed_results.append(processed_record)
                    yield processed_record

                # Set final span attributes
                span.set_attribute("outputs.count", len(processed_results))
                span.set_status(trace.Status(trace.StatusCode.OK))

            except GeneratorExit:
                # Handle early generator termination gracefully
                span.set_status(trace.Status(trace.StatusCode.OK))
                raise
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def _validate_cached_record(self, cached_record) -> bool:
        """Validate that a cached record is still usable.

        Override this method for custom validation logic.
        """
        return cached_record is not None

    async def _process_without_cache(
        self, record: BaseRecord
    ) -> AsyncGenerator[BaseRecord, None]:  # noqa: PLR0912
        """Process BaseRecord through the processor chain without caching."""
        start_time = time.time()
        tracer = trace.get_tracer("buttermilk.pipeline")
        record_id = getattr(record, "record_id", "unknown")
        title = getattr(record, "title", None)
        record_title = (
            (title[:50] if title else "Unknown")
            if hasattr(record, "title")
            else "Unknown"
        )

        # Build span attributes for processor chain
        span_attributes = {
            "record.id": record_id,
            "pipeline.name": self.pipeline_name,
            "processor.count": len(self.processors),
            "record.title": record_title,
        }

        with tracer.start_as_current_span(
            "pipeline.process_chain", attributes=span_attributes
        ) as chain_span:
            # Log processing start
            logger.debug(
                f"🔷 {self.pipeline_name} Processing record ID: {record.record_id} - '{record_title}'",
                pipeline_name=self.pipeline_name,
                record_id=record.record_id,
                record_title=record_title,
            )
            try:
                # Start with the record as a single item in a processing queue
                processing_queue = [record]

                # Flow records through each processor in sequence
                for processor_index, processor in enumerate(self.processors):
                    # Create span for this processor
                    processor_class = type(processor).__name__

                    # Extract processor configuration for cache key
                    # This ensures cache invalidation when processor params change
                    try:
                        if hasattr(processor, "model_dump"):
                            # Pydantic model - use model_dump()
                            processor_config = processor.model_dump()
                        elif hasattr(processor, "__dict__"):
                            # Regular object - use __dict__
                            processor_config = processor.__dict__.copy()
                        else:
                            # Fallback to empty dict
                            processor_config = {}

                        # Compute parameter hash for cache invalidation
                        param_hash = compute_processor_config_hash(processor_config)
                    except Exception as e:
                        # If hashing fails, use a default hash to avoid breaking the pipeline
                        logger.warning(
                            "Failed to compute processor config hash, using default",
                            processor_class=processor_class,
                            error=str(e),
                        )
                        param_hash = "00000000"

                    # Create unique processor ID for this processor (for caching and process() calls)
                    # Format: {pipeline_name}/{index:02d}.{processor_class}/{param_hash}
                    # The param_hash ensures cache invalidation when processor configuration changes
                    # TODO: rename to processor_id
                    processor_stage_name = f"{self.pipeline_name}/{processor_index:02d}.{processor_class}/{param_hash}"
                    processor_span_attributes = {
                        "processor.index": processor_index,
                        "processor.class": processor_class,
                        "processor.id": processor_stage_name,  # Full processor identifier within pipeline
                        "processor.param_hash": param_hash,
                        "inputs.count": len(processing_queue),
                    }

                    with tracer.start_as_current_span(
                        f"pipeline.processor.{processor_index}",
                        attributes=processor_span_attributes,
                    ) as processor_span:
                        next_queue = []

                        # Process each record in the current queue through this processor
                        for current_record in processing_queue:
                            # Trace record state BEFORE processor
                            trace_before = self._trace_record_state(
                                current_record, "before_processor", processor_stage_name
                            )
                            logger.debug(
                                "📋 Record state before processor", **trace_before
                            )

                            # Check processor-specific cache first (unless processor opts out)
                            cached_outputs = None
                            if getattr(processor, "skip_cache", False):
                                logger.debug(
                                    f"🚫 Skipping cache for {processor_class} (skip_cache=True)",
                                    processor_class=processor_class,
                                )
                            else:
                                cached_outputs = await self._check_processor_cache(
                                    current_record, processor_stage_name
                                )

                            if cached_outputs:
                                logger.debug(
                                    f"⚡ Processor {processor_stage_name} cache hit",
                                    record_id=getattr(
                                        current_record, "record_id", "unknown"
                                    ),
                                    processor_stage=processor_stage_name,
                                    processor_stage_name=processor_stage_name,
                                    cached_outputs_count=len(cached_outputs),
                                )
                                # Trace cached outputs
                                for i, cached_output in enumerate(cached_outputs):
                                    trace_cached = self._trace_record_state(
                                        cached_output,
                                        "cached_output",
                                        processor_stage_name,
                                    )
                                    logger.debug(
                                        f"📋 Cached output {i} state", **trace_cached
                                    )
                                next_queue.extend(cached_outputs)
                                continue

                            outputs = []
                            try:
                                # Extract parent_call_id from record for trace lineage
                                # This links processor operations back to their source records
                                parent_trace_id = getattr(
                                    current_record, "parent_call_id", None
                                )

                                # All processors now use unified ProcessingContext interface
                                context = ProcessingContext(
                                    session_id=parent_trace_id or processor_stage_name,
                                    record=current_record,
                                    batch_id=self.pipeline_name,
                                    span=processor_span,
                                )
                                async for output_record in processor.process(context):
                                    outputs.append(output_record)
                            except Exception as e:
                                # Don't log here - let the task wrapper handle error logging
                                # to avoid duplicate error messages
                                processor_span.set_status(
                                    trace.Status(trace.StatusCode.ERROR, str(e))
                                )
                                raise

                            if not outputs:
                                # This record was filtered out by this processor
                                processor_span.set_attribute("filtered", True)
                                raise RecordSkippedException(
                                    f"Record was filtered out by processor in {processor_stage_name}"
                                )
                            else:
                                # Trace record state AFTER processor
                                for i, output_record in enumerate(outputs):
                                    trace_after = self._trace_record_state(
                                        output_record,
                                        "after_processor",
                                        processor_stage_name,
                                    )
                                    logger.debug(
                                        f"📋 Record state after processor (output {i})",
                                        **trace_after,
                                    )

                                # Cache the processor outputs (unless processor opts out)
                                if getattr(processor, "skip_cache", False):
                                    logger.debug(
                                        f"🚫 Skipping cache save for {processor_class} (skip_cache=True)",
                                        processor_class=processor_class,
                                    )
                                else:
                                    await self._save_processor_cache(
                                        current_record, outputs, processor_stage_name
                                    )
                                # Add all outputs to the next processing queue
                                next_queue.extend(outputs)

                        # Set processor span attributes for outputs
                        processor_span.set_attribute("outputs.count", len(next_queue))
                        processor_span.set_attribute("filtered", False)
                        processor_span.set_status(trace.Status(trace.StatusCode.OK))

                        # Move to next stage with all outputs from this processor
                        processing_queue = next_queue

                # If we reach here, processing_queue contains all final outputs
                if not processing_queue:
                    # Everything was filtered out (shouldn't happen due to exception above)
                    raise RecordSkippedException("All outputs were filtered in")

                # Set final chain span attributes
                total_outputs = len(processing_queue)
                processing_time_ms = int((time.time() - start_time) * 1000)
                chain_span.set_attribute("outputs.count", total_outputs)
                chain_span.set_attribute("processing.time_ms", processing_time_ms)
                chain_span.set_status(trace.Status(trace.StatusCode.OK))

                # Add success metadata to all final outputs and yield them
                for output_index, final_record in enumerate(processing_queue):
                    # Create stage metadata with output tracking for 1:N transformations
                    stage_metadata = {
                        "status": "processed",
                        "timestamp": time.time(),
                        "processing_time_ms": processing_time_ms,
                    }

                    # Add output tracking for 1:N transformations
                    if total_outputs > 1:
                        stage_metadata["output_index"] = output_index
                        stage_metadata["total_outputs"] = total_outputs

                    # Preserve existing metadata and add stage metadata
                    updated_metadata = (
                        final_record.metadata.copy() if final_record.metadata else {}
                    )
                    updated_metadata[self.pipeline_name] = stage_metadata

                    updated_record = final_record.model_copy(
                        update={"metadata": updated_metadata}
                    )
                    yield updated_record

            except GeneratorExit:
                # Handle early generator termination gracefully
                chain_span.set_status(trace.Status(trace.StatusCode.OK))
                raise
            except Exception as e:
                chain_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                # Let exception bubble up - task wrapper will handle error logging
                raise

    async def __call__(self) -> AsyncIterator[BaseRecord]:
        """Process BaseRecord objects from source with TaskGroup-based concurrency.

        Each BaseRecord flows through the entire processor chain individually.
        Uses TaskGroup for natural error collection and concurrency management.
        """
        if self.source is None:
            logger.error(
                f"[{self.pipeline_name}] No source iterator configured",
                pipeline_name=self.pipeline_name,
            )
            return

        tracer = trace.get_tracer("buttermilk.pipeline")
        start_time = time.time()

        # Build span attributes for stage orchestration
        span_attributes = {
            "stage.name": self.pipeline_name,
            "stage.concurrency": self.concurrency,
            "stage.limit": self.limit,
            "stage.processor_count": len(self.processors),
            "cache.enabled": self.enable_record_cache,
        }

        with tracer.start_as_current_span(
            f"pipeline.{self.pipeline_name}", attributes=span_attributes
        ) as stage_span:
            try:
                async for record in self._run_pipeline_with_tracing(
                    stage_span, start_time
                ):
                    yield record
            except GeneratorExit:
                # Handle early generator termination gracefully
                stage_span.set_status(trace.Status(trace.StatusCode.OK))
                raise
            except Exception as e:
                stage_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    async def _run_pipeline_with_tracing(
        self, stage_span, start_time, show_progress: bool = True
    ) -> AsyncIterator[BaseRecord]:
        """Internal method to run pipeline with tracing context.

        Args:
            stage_span: OpenTelemetry span for tracing
            start_time: Start time of the pipeline
            show_progress: Whether to display a progress bar (default: True)
        """
        from collections import deque

        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        tracer = trace.get_tracer("buttermilk.pipeline")
        pending_tasks: set[asyncio.Task] = set()

        # Track input record completion timing for rate calculation
        # Keep timestamps for last N completed inputs where N = concurrency * 5
        rate_window_size = max(self.concurrency * 5, 10)
        input_completion_timestamps: deque[float] = deque(maxlen=rate_window_size)

        def calculate_rate() -> str:
            """Calculate seconds per input record over the recent window."""
            if len(input_completion_timestamps) < 2:
                return "-- s/in"
            # Time span between oldest and newest in window
            time_span = input_completion_timestamps[-1] - input_completion_timestamps[0]
            # Number of intervals = number of records - 1
            num_intervals = len(input_completion_timestamps) - 1
            if num_intervals > 0 and time_span > 0:
                sec_per_record = time_span / num_intervals
                return f"{sec_per_record:.1f} s/in"
            return "-- s/in"

        # Set up progress bar
        # Shows total outputs yielded, skip/fail counts, rate, and elapsed time
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("Processed: {task.fields[output]}"),
            TextColumn("Skip: {task.fields[skipped]}"),
            TextColumn("Fail: {task.fields[failed]}"),
            TextColumn("•"),
            TextColumn("{task.fields[rate]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            disable=not show_progress,
        )

        try:
            log_interval = 15.0
            last_log = time.monotonic()

            def maybe_log_status(pending_count: int):
                nonlocal last_log
                now = time.monotonic()
                if now - last_log >= log_interval:
                    logger.debug(
                        f"📊 Pipeline '{self.pipeline_name}': attempted={self._summary.attempted} processed={self._summary.processed} "
                        f"skipped={self._summary.skipped} failed={self._summary.failed} pending={pending_count}"
                    )
                    last_log = now

            # Ensure we have an async iterator
            source_iter = (
                self.source
                if hasattr(self.source, "__anext__")
                else self.source.__aiter__()
            )

            completed_records = asyncio.Queue()

            async def process_and_queue(record: BaseRecord):
                """Process BaseRecord and put all results in queue, with semaphore-based concurrency control."""
                record_id = getattr(record, "record_id", "unknown")

                # Acquire semaphore slot before processing (enforces concurrency limit)
                async with self._semaphore:
                    # Create span for individual task processing
                    task_span_attributes = {
                        "record.id": record_id,
                        "pipeline.name": self.pipeline_name,
                    }

                    with tracer.start_as_current_span(
                        "pipeline.task.process", attributes=task_span_attributes
                    ) as task_span:
                        try:
                            results_count = 0
                            async for processed_record in self._process_single_record(
                                record
                            ):
                                await completed_records.put(("success", processed_record))
                                results_count += 1

                            # Only count as processed if we got at least one output
                            if results_count > 0:
                                self._summary.increment_processed()
                                input_completion_timestamps.append(time.monotonic())
                                task_span.set_attribute("outputs.count", results_count)
                                task_span.set_attribute("status", "processed")
                                task_span.set_status(trace.Status(trace.StatusCode.OK))
                            else:
                                task_span.set_attribute("status", "no_outputs")
                                task_span.set_status(trace.Status(trace.StatusCode.OK))

                        except RecordSkippedException as e:
                            self._summary.increment_skipped()
                            task_span.set_attribute("status", "skipped")
                            task_span.set_attribute("skip_reason", str(e))
                            task_span.set_status(trace.Status(trace.StatusCode.OK))
                            # Record was intentionally skipped, no error logging needed
                            logger.debug(
                                f"Record {record_id} skipped in stage {self.pipeline_name}: {e}",
                                record_id=record_id,
                                pipeline_name=self.pipeline_name,
                                error=str(e),
                            )
                            # For now we just drop skipped records, don't put them in queue

                        except Exception as e:
                            self._summary.increment_failed()
                            task_span.set_attribute("status", "failed")
                            task_span.set_attribute("error_type", type(e).__name__)
                            task_span.set_status(
                                trace.Status(trace.StatusCode.ERROR, str(e))
                            )
                            # Log failure but don't stop processing other records
                            logger.error(
                                f"❌ Failed to process record {record_id}: {type(e).__name__}",
                                record_id=record_id,
                                pipeline=self.pipeline_name,
                                error=str(e),
                                error_type=type(e).__name__,
                            )

                            # Add error metadata to record and put in queue
                            existing_metadata = getattr(record, "metadata", None) or {}
                            error_metadata = {
                                **existing_metadata,
                                self.pipeline_name: {
                                    "status": "failed",
                                    "timestamp": time.time(),
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                            }
                            failed_record = record.model_copy(
                                update={"metadata": error_metadata}
                            )
                            await completed_records.put(("error", failed_record))
                            # Continue processing other records - don't raise

            # Producer: Create tasks for incoming BaseRecord objects
            # Note: Record limiting is handled by LimitingSource wrapper applied in model_post_init
            # This ensures limit applies to original records BEFORE replication
            async def producer():
                nonlocal pending_tasks
                async for record in source_iter:
                    self._summary.increment_attempted()

                    # Create and track task (concurrency controlled by semaphore in process_and_queue)
                    task = asyncio.create_task(process_and_queue(record))
                    pending_tasks.add(task)

                    maybe_log_status(len(pending_tasks))

                # Wait for all tasks to complete before signaling completion
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)

                # Signal completion by putting None
                await completed_records.put(None)

            # Consumer: Yield completed BaseRecord objects
            async def consumer():
                try:
                    while True:
                        result = await completed_records.get()
                        if result is None:  # End signal
                            break

                        status, record = result
                        if status == "success":
                            yield record
                        elif status == "skipped":
                            # Optionally yield skipped records too (for debugging/tracking)
                            # For now, we'll skip them and just log
                            pass
                        elif status == "error":
                            # Optionally yield failed records too (for debugging/recovery)
                            # For now, we'll skip them and just log
                            pass
                except GeneratorExit:
                    # Handle early consumer termination gracefully
                    raise

            # Start producer
            producer_task = asyncio.create_task(producer())

            # Yield from consumer with progress bar
            with progress:
                # Create progress task with no total (count upwards)
                # We don't know how many outputs we'll get with 1:N transformations
                progress_task = progress.add_task(
                    f"Pipeline: {self.pipeline_name}",
                    total=None,  # Count upwards, don't show percentage
                    output=0,
                    skipped=0,
                    failed=0,
                    rate="-- s/in",
                )
                output_count = 0

                async for record in consumer():
                    output_count += 1
                    # Update progress with current stats
                    progress.update(
                        progress_task,
                        completed=output_count,
                        output=output_count,
                        skipped=self._summary.skipped,
                        failed=self._summary.failed,
                        rate=calculate_rate(),
                    )
                    yield record

                # Wait for producer to finish
                await producer_task

                # Final progress update
                progress.update(
                    progress_task,
                    completed=output_count,
                    output=output_count,
                    skipped=self._summary.skipped,
                    failed=self._summary.failed,
                    rate=calculate_rate(),
                )

            # Call finalize_processing on all processors
            await self._finalize_all_processors()

            # Set final stage span attributes
            stage_duration_ms = self._summary.duration_ms()
            stage_span.set_attribute("records.attempted", self._summary.attempted)
            stage_span.set_attribute("records.processed", self._summary.processed)
            stage_span.set_attribute("records.skipped", self._summary.skipped)
            stage_span.set_attribute("records.failed", self._summary.failed)
            stage_span.set_attribute("stage.duration_ms", stage_duration_ms)
            stage_span.set_status(trace.Status(trace.StatusCode.OK))

        except Exception as e:
            stage_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(
                f"Pipeline error in stage '{self.pipeline_name}': {e}",
                pipeline_name=self.pipeline_name,
                error=str(e),
            )
            # Cancel all pending tasks
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to complete
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            # Still try to finalize processors even on error
            try:
                await self._finalize_all_processors()
            except Exception as finalize_error:
                logger.error(
                    f"Error during finalization after pipeline error: {finalize_error}",
                    error=str(finalize_error),
                )
            raise
        finally:
            logger.info(
                f"Pipeline '{self.pipeline_name}': {self._summary.format_for_console()}"
            )

    async def _check_processor_cache(
        self, record: BaseRecord, processor_stage_name: str
    ) -> list[BaseRecord] | None:
        """Check cache for processor-specific outputs."""
        if (
            not self.enable_record_cache
            or not self._record_cache
            or self.force_reprocess
            or not hasattr(record, "record_id")
        ):
            return None

        logger.debug(
            "🔍 Checking processor cache",
            record_id=record.record_id,
            processor_stage=processor_stage_name,
        )

        # Try 1:1 cached result first
        cached_record = self._record_cache.load(record.record_id, processor_stage_name)
        if cached_record and self._validate_cached_record(cached_record):
            return [cached_record]

        # Try 1:N cached results
        cached_outputs = []
        output_index = 0
        while True:
            cache_key = f"{record.record_id}_output_{output_index}"
            cached_output = self._record_cache.load(cache_key, processor_stage_name)
            if not cached_output:
                break
            cached_outputs.append(cached_output)
            output_index += 1

        return cached_outputs if cached_outputs else None

    def _trace_record_state(
        self, record: BaseRecord, stage: str, processor_stage_name: str = ""
    ) -> dict:
        """Trace the current state of a record for debugging.

        Args:
            record: The record to trace
            stage: Stage description (e.g., "before_processor", "after_processor")
            processor_stage_name: Full processor stage name including param hash
                                 (e.g., "pipeline/00.LLMCore/a3f8b2c1")
        """
        record_id = getattr(record, "record_id", "unknown")

        # Basic record info
        trace_info = {
            "record_id": record_id,
            "stage": stage,
            "processor_stage_name": processor_stage_name,
        }

        # Add cache file paths for investigation
        if self._record_cache and processor_stage_name:
            cache_path = self._record_cache._record_path(
                processor_stage_name, record_id
            )
            trace_info["cache_file"] = str(cache_path)
            trace_info["cache_exists"] = (
                cache_path.exists() if hasattr(cache_path, "exists") else False
            )
            trace_info["cache_base_dir"] = str(self._record_cache.base_dir)

        # Get all non-private attributes of the record
        # Critical fields that should NEVER be truncated (needed for debugging)
        critical_fields = {"record_id", "record_hash", "record_class", "error", "dataset_name", "split_type"}
        # Fields that can be truncated (typically large content)
        truncatable_fields = {"content", "alt_text", "body", "text", "description"}

        record_fields = {}
        for attr_name in dir(record):
            if (
                not attr_name.startswith("_")
                and not callable(getattr(record, attr_name, None))
                and not attr_name.startswith("model_")
            ):
                try:
                    attr_value = getattr(record, attr_name, None)
                    if attr_value is not None:
                        # Handle different types of fields
                        if isinstance(attr_value, str):
                            if attr_name in critical_fields:
                                # Never truncate critical fields
                                record_fields[attr_name] = attr_value
                            elif attr_name in truncatable_fields and len(attr_value) > 20:
                                # Only truncate known large content fields
                                record_fields[attr_name] = attr_value[:20] + "..."
                            else:
                                # Other string fields: full value
                                record_fields[attr_name] = attr_value
                        elif isinstance(attr_value, list):
                            if attr_name == "error" and attr_value:
                                # Show full error list for debugging
                                record_fields["error"] = attr_value
                                record_fields["error_count"] = len(attr_value)
                            else:
                                record_fields[f"{attr_name}_count"] = len(attr_value)
                        elif isinstance(attr_value, dict):
                            record_fields[f"{attr_name}_keys"] = list(attr_value.keys())
                except Exception:
                    # Skip fields that can't be accessed
                    continue

        trace_info.update(record_fields)
        return trace_info

    async def _save_processor_cache(
        self,
        input_record: BaseRecord,
        outputs: list[BaseRecord],
        processor_stage_name: str,
    ) -> None:
        """Save processor outputs to cache."""
        if (
            not self.enable_record_cache
            or not self._record_cache
            or not outputs
            or not hasattr(input_record, "record_id")
        ):
            return

        logger.debug(
            "💾 Saving processor outputs to cache",
            record_id=input_record.record_id,
            processor_stage=processor_stage_name,
            outputs_count=len(outputs),
        )

        try:
            if len(outputs) == 1:
                # 1:1 transformation - use input record_id as cache key
                self._record_cache.save(outputs[0], processor_stage_name)
            else:
                # 1:N transformation - use indexed cache keys
                for output_index, output_record in enumerate(outputs):
                    cache_key = f"{input_record.record_id}_output_{output_index}"

                    # Preserve original record_id in metadata (only if not already set)
                    updated_metadata = (
                        output_record.metadata.copy() if output_record.metadata else {}
                    )
                    if "original_record_id" not in updated_metadata:
                        updated_metadata["original_record_id"] = input_record.record_id

                    cache_record = output_record.model_copy(
                        update={"record_id": cache_key, "metadata": updated_metadata}
                    )
                    self._record_cache.save(cache_record, processor_stage_name)
        except Exception as e:
            logger.debug(
                "💥 Failed to save processor cache",
                record_id=input_record.record_id,
                processor_stage=processor_stage_name,
                error=str(e),
            )

    async def _finalize_all_processors(self) -> None:
        """Call finalize_processing on all processors that support it.

        This ensures proper cleanup and final sync operations for processors
        like ChromaDBUploader and ChromaDBEmbeddings.
        """
        logger.debug(
            f"🔄 Finalizing {len(self.processors)} processors in stage '{self.pipeline_name}'",
            processor_count=len(self.processors),
            pipeline_name=self.pipeline_name,
        )

        for i, processor in enumerate(self.processors):
            if hasattr(processor, "finalize_processing"):
                try:
                    logger.debug(
                        f"🔄 Finalizing processor {i}: {type(processor).__name__}",
                        processor_index=i,
                        processor_name=type(processor).__name__,
                    )
                    result = await processor.finalize_processing()
                    if result:
                        logger.info(
                            f"✅ Successfully finalized processor {i}: {type(processor).__name__}",
                            processor_index=i,
                            processor_name=type(processor).__name__,
                        )
                    else:
                        logger.warning(
                            f"⚠️ Processor {i} finalization reported issues: {type(processor).__name__}",
                            processor_index=i,
                            processor_name=type(processor).__name__,
                        )
                except Exception as e:
                    logger.error(
                        f"❌ Failed to finalize processor {i} ({type(processor).__name__}): {e}",
                        processor_index=i,
                        processor_name=type(processor).__name__,
                        error=str(e),
                    )
            else:
                logger.debug(
                    f"⏭️ Processor {i} has no finalize_processing method: {type(processor).__name__}",
                    processor_index=i,
                    processor_name=type(processor).__name__,
                )

        logger.debug(
            f"✅ Completed finalization for stage '{self.pipeline_name}'",
            pipeline_name=self.pipeline_name,
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
            pipeline_name="demo",
            processor_stage_name="enrich",
            source=source,
            processor=enrich_func
        )

        stage2 = PipelineOrchestrator(
            pipeline_name="demo",
            processor_stage_name="validate",
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
