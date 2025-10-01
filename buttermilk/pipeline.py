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

NON-GOALS
- Pipeline will NOT validate processor-specific data shapes
- Pipeline will NOT handle type conversions for processors
- Pipeline will NOT have processor-specific optimizations

"""

import asyncio
import time
from typing import Any, AsyncGenerator, AsyncIterator, Mapping, Optional, Protocol, runtime_checkable
import weave

import hydra
import pydantic
from omegaconf import DictConfig
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk import bm, logger
from buttermilk._core.types import BaseRecord


class RecordSkippedException(Exception):
    """Exception raised when a record is intentionally skipped/filtered."""

    pass


@runtime_checkable
class Processor(Protocol):
    """Standard processor interface - async generator that accepts and yields BaseRecord objects.

    PROCESSOR GUIDELINES:

    1. **record_id Immutability**: NEVER modify the record_id field. It must remain
       unchanged through all transformations to preserve lineage to the original record.

    2. **Semantic Identifiers**: When creating 1:N transformations (splits), add your own
       meaningful identifier fields instead of modifying record_id:
       - Chunking processor: Add `chunk_id`, `chunk_index` fields
       - TMDB processor: Add `observation_id`, `provider_name` fields
       - LLM processor with multiple calls: Add `llm_call_index` field

    3. **Metadata Namespacing**: Store processor-specific metadata in record.metadata[stage_name].
       Each processor should use its own namespace to avoid conflicts.

    4. **Pipeline Metadata**: The pipeline will automatically add stage metadata with:
       - status: "processed"
       - timestamp: processing timestamp
       - processing_time_ms: time taken
       - output_index, total_outputs: for 1:N transformations

    5. **Filtering**: To filter out a record, simply yield nothing.
       The pipeline will handle the RecordSkippedException automatically.

    6. Processors MUST work with standard Python types. When loading from cache, objects
       will be deserialized into dicts/lists/primitives. Processors must handle their own
       own conversions internally if needed.
    """

    async def process(
        self,
        record: Any = BaseRecord,
        *,
        pipeline_stage: str,
        parent_trace_id: Optional[str] = None,
        component_name: str = "LLMCore",
        cancellation_token: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord and yield zero or more output BaseRecord objects.

        Args:
            record: BaseRecord object to process

        Yields:
            BaseRecord objects (can be zero for filtering, one for 1:1, multiple for 1:N)

        Examples:
            # 1:1 transformation preserving record_id
            updated_record = record.model_copy(update={
                output_col: LLMResult.content,
                "metadata": {
                    **record.metadata,
                    "summarizer": {"model": "gpt-4", "tokens": 500}
                }
            })
            yield updated_record

            # 1:N transformation with semantic IDs
            chunks = split_text(record.content)
            for i, chunk_text in enumerate(chunks):
                chunk_record = record.model_copy(update={
                    "content": chunk_text,
                    "chunk_id": f"{record.record_id}_chunk_{i}",
                    "chunk_index": i,
                    "metadata": {
                        **record.metadata,
                        "chunking": {"parent_id": record.record_id, "index": i}
                    }
                })
                yield chunk_record

            # Filtering (yield nothing)
            if should_filter(record):
                return  # Record is filtered out
            yield record  # Pass through unchanged
        """
        ...


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
    max_records: Optional[int] = Field(default=None, description="Maximum records to process")
    stage_name: str = Field(..., description="Name for this processing pipeline")  # TODO: rename to pipeline_name
    force_reprocess: bool = Field(default=False, description="Ignore cache and reprocess")
    enable_record_cache: bool = Field(default=True, description="Enable per-processor Record caching")
    cache_dir: Optional[str] = Field(default=None, description="Base directory for record cache (defaults to ~/.cache/buttermilk)")

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
        logger.info(
            "🗂️  Pipeline cache initialization",
            stage_name=self.stage_name,
            enable_record_cache=self.enable_record_cache,
            cache_dir_param=self.cache_dir,
            session_cache_dir=getattr(bm.session_info, "cache_dir", None),
        )

        if self.enable_record_cache:
            try:
                from buttermilk._core.record_cache import RecordCache

                # Use cache_dir from session_info if available, otherwise use parameter or let RecordCache use defaults
                cache_base_dir = self.cache_dir or bm.session_info.cache_dir

                logger.info("🗂️  Creating RecordCache", cache_base_dir=cache_base_dir, stage_name=self.stage_name)
                self._record_cache = RecordCache(base_dir=cache_base_dir)
            except Exception as e:
                logger.warning("Failed to initialize RecordCache, continuing without caching", error=str(e))
                self._record_cache = None
        else:
            logger.info("🚫 Record cache disabled for pipeline stage", stage_name=self.stage_name)

        return self

    @weave.op
    async def _process_single_record(self, record: BaseRecord) -> AsyncGenerator[BaseRecord, None]:
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
            raise ValueError(f"[{self.stage_name}] No processors configured")

        tracer = trace.get_tracer("buttermilk.pipeline")
        record_id = getattr(record, "record_id", "unknown")
        title = getattr(record, "title", None)
        record_title = (title[:50] if title else "Unknown") if hasattr(record, "title") else "Unknown"

        # Build span attributes for record processing
        span_attributes = {
            "record.id": record_id,
            "record.title": record_title,
            "stage.name": self.stage_name,
            "stage.processor_count": len(self.processors),
        }

        with tracer.start_as_current_span("pipeline.process_record", attributes=span_attributes) as span:
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

                # Note: Caching is now handled per-processor in _process_without_cache

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

    async def _process_without_cache(self, record: BaseRecord) -> AsyncGenerator[BaseRecord, None]:
        """Process BaseRecord through the processor chain without caching."""
        start_time = time.time()
        tracer = trace.get_tracer("buttermilk.pipeline")
        record_id = getattr(record, "record_id", "unknown")

        # Log processing start
        if hasattr(record, "record_id"):
            title = getattr(record, "title", None)
            record_title = (title[:50] if title else "Unknown") if hasattr(record, "title") else "Unknown"
            logger.info(f"🔷 [{self.stage_name}-{record.record_id}] Processing record '{record_title}'")

        # Build span attributes for processor chain
        span_attributes = {
            "record.id": record_id,
            "stage.name": self.stage_name,
            "processor.count": len(self.processors),
        }

        with tracer.start_as_current_span("pipeline.process_chain", attributes=span_attributes) as chain_span:
            try:
                # Start with the record as a single item in a processing queue
                processing_queue = [record]

                # Flow records through each processor in sequence
                for processor_index, processor in enumerate(self.processors):
                    # Create span for this processor
                    processor_class = type(processor).__name__
                    # Create unique processor ID for this processor (for caching and process() calls)
                    # Format: {pipeline_name}.{index:02d}.{processor_class}
                    processor_stage_name = f"{self.stage_name}.{processor_index:02d}.{processor_class}"  # TODO: rename to processor_id
                    processor_span_attributes = {
                        "processor.index": processor_index,
                        "processor.class": processor_class,
                        "processor.id": processor_stage_name,  # Full processor identifier within pipeline
                        "inputs.count": len(processing_queue),
                    }

                    with tracer.start_as_current_span(
                        f"pipeline.processor.{processor_index}", attributes=processor_span_attributes
                    ) as processor_span:
                        next_queue = []

                        # Process each record in the current queue through this processor
                        for current_record in processing_queue:
                            # Trace record state BEFORE processor
                            trace_before = self._trace_record_state(current_record, "before_processor", processor_class, processor_index)
                            logger.debug("📋 Record state before processor", **trace_before)

                            # Check processor-specific cache first (unless processor opts out)
                            cached_outputs = None
                            if getattr(processor, "skip_cache", False):
                                logger.debug(f"🚫 Skipping cache for {processor_class} (skip_cache=True)")
                            else:
                                cached_outputs = await self._check_processor_cache(current_record, processor_stage_name)

                            if cached_outputs:
                                logger.info(
                                    f"⚡ Processor {processor_stage_name} cache hit",
                                    record_id=getattr(current_record, "record_id", "unknown"),
                                    processor_stage=processor_stage_name,
                                    cached_outputs_count=len(cached_outputs),
                                )
                                # Trace cached outputs
                                for i, cached_output in enumerate(cached_outputs):
                                    trace_cached = self._trace_record_state(cached_output, "cached_output", processor_class, processor_index)
                                    logger.debug(f"📋 Cached output {i} state", **trace_cached)
                                next_queue.extend(cached_outputs)
                                continue

                            outputs = []
                            try:
                                async for output_record in processor.process(current_record, processor_stage=processor_stage_name):
                                    outputs.append(output_record)
                            except Exception as e:
                                # Log with processor-specific stage name using structured logging
                                record_id = getattr(current_record, "record_id", "unknown")
                                processor_type = type(processor).__name__
                                logger.error(
                                    f"Error processing record in pipeline processor {processor_stage_name}",
                                    record_id=record_id,
                                    stage_name=self.stage_name,
                                    processor_stage=processor_stage_name,
                                    processor_type=processor_type,
                                    processor_index=processor_index,
                                    error=str(e),
                                    error_type=type(e).__name__,
                                )
                                processor_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                                raise

                            if not outputs:
                                # This record was filtered out by this processor
                                processor_span.set_attribute("filtered", True)
                                raise RecordSkippedException(f"Record was filtered out by processor in {processor_stage_name}")
                            else:
                                # Trace record state AFTER processor
                                for i, output_record in enumerate(outputs):
                                    trace_after = self._trace_record_state(output_record, "after_processor", processor_class, processor_index)
                                    logger.debug(f"📋 Record state after processor (output {i})", **trace_after)

                                # Cache the processor outputs (unless processor opts out)
                                if getattr(processor, "skip_cache", False):
                                    logger.debug(f"🚫 Skipping cache save for {processor_class} (skip_cache=True)")
                                else:
                                    await self._save_processor_cache(current_record, outputs, processor_stage_name)
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
                    updated_metadata = final_record.metadata.copy() if final_record.metadata else {}
                    updated_metadata[self.stage_name] = stage_metadata

                    updated_record = final_record.model_copy(update={"metadata": updated_metadata})
                    yield updated_record

            except GeneratorExit:
                # Handle early generator termination gracefully
                chain_span.set_status(trace.Status(trace.StatusCode.OK))
                raise
            except Exception as e:
                chain_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                # Let exception bubble up - TaskGroup will handle error collection
                record_id = getattr(record, "record_id", "unknown")
                logger.error(
                    "Error processing record in pipeline chain",
                    record_id=record_id,
                    stage_name=self.stage_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    processor_count=len(self.processors),
                )
                raise

    async def __call__(self) -> AsyncIterator[BaseRecord]:
        """Process BaseRecord objects from source with TaskGroup-based concurrency.

        Each BaseRecord flows through the entire processor chain individually.
        Uses TaskGroup for natural error collection and concurrency management.
        """
        if self.source is None:
            logger.error(f"[{self.stage_name}] No source iterator configured")
            return

        tracer = trace.get_tracer("buttermilk.pipeline")
        start_time = time.time()

        # Build span attributes for stage orchestration
        span_attributes = {
            "stage.name": self.stage_name,
            "stage.concurrency": self.concurrency,
            "stage.max_records": self.max_records,
            "stage.processor_count": len(self.processors),
            "cache.enabled": self.enable_record_cache,
        }

        with tracer.start_as_current_span(f"pipeline.stage.{self.stage_name}", attributes=span_attributes) as stage_span:
            try:
                async for record in self._run_pipeline_with_tracing(stage_span, start_time):
                    yield record
            except GeneratorExit:
                # Handle early generator termination gracefully
                stage_span.set_status(trace.Status(trace.StatusCode.OK))
                raise
            except Exception as e:
                stage_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    async def _run_pipeline_with_tracing(self, stage_span, start_time) -> AsyncIterator[BaseRecord]:
        """Internal method to run pipeline with tracing context."""
        tracer = trace.get_tracer("buttermilk.pipeline")
        pending_tasks: set[asyncio.Task] = set()

        try:
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

            # Ensure we have an async iterator
            source_iter = self.source if hasattr(self.source, "__anext__") else self.source.__aiter__()

            completed_records = asyncio.Queue()

            async def process_and_queue(record: BaseRecord):
                """Process BaseRecord and put all results in queue."""
                record_id = getattr(record, "record_id", "unknown")

                # Create span for individual task processing
                task_span_attributes = {
                    "record.id": record_id,
                    "stage.name": self.stage_name,
                }

                with tracer.start_as_current_span("pipeline.task.process", attributes=task_span_attributes) as task_span:
                    try:
                        results_count = 0
                        async for processed_record in self._process_single_record(record):
                            await completed_records.put(("success", processed_record))
                            results_count += 1

                        # Only count as processed if we got at least one output
                        if results_count > 0:
                            self._processed += 1
                            task_span.set_attribute("outputs.count", results_count)
                            task_span.set_attribute("status", "processed")
                            task_span.set_status(trace.Status(trace.StatusCode.OK))
                        else:
                            task_span.set_attribute("status", "no_outputs")
                            task_span.set_status(trace.Status(trace.StatusCode.OK))

                    except RecordSkippedException as e:
                        self._skipped += 1
                        task_span.set_attribute("status", "skipped")
                        task_span.set_attribute("skip_reason", str(e))
                        task_span.set_status(trace.Status(trace.StatusCode.OK))
                        # Record was intentionally skipped, no error logging needed
                        logger.debug(f"Record {record_id} skipped in stage {self.stage_name}: {e}")
                        # For now we just drop skipped records, don't put them in queue

                    except Exception as e:
                        self._failed += 1
                        task_span.set_attribute("status", "failed")
                        task_span.set_attribute("error_type", type(e).__name__)
                        task_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        # Log error but don't stop processing other records
                        logger.error(f"Error processing record {record_id} in stage {self.stage_name}: {e}")

                        # Add error metadata to record and put in queue
                        existing_metadata = getattr(record, "metadata", None) or {}
                        error_metadata = {
                            **existing_metadata,
                            self.stage_name: {
                                "status": "failed",
                                "timestamp": time.time(),
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        }
                        failed_record = record.model_copy(update={"metadata": error_metadata})
                        await completed_records.put(("error", failed_record))
                        # Continue processing other records - don't raise

            # Producer: Create tasks for incoming BaseRecord objects
            async def producer():
                nonlocal pending_tasks
                async for record in source_iter:
                    self._attempted += 1

                    # Check if we've hit max_records
                    if self.max_records is not None and (self._processed + self._failed + self._skipped) >= self.max_records:
                        logger.info(f"🔚 Stage '{self.stage_name}' reached max_records ({self._attempted}/{self.max_records}) – stopping")
                        break

                    # Maintain concurrency limit
                    while len(pending_tasks) >= self.concurrency:
                        await asyncio.sleep(0.01)  # Brief pause to allow task completion
                        # Clean up completed tasks
                        pending_tasks = {t for t in pending_tasks if not t.done()}

                    # Create and track task
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

            # Yield from consumer
            async for record in consumer():
                yield record

            # Wait for producer to finish
            await producer_task

            # Set final stage span attributes
            stage_duration_ms = int((time.time() - start_time) * 1000)
            stage_span.set_attribute("records.attempted", self._attempted)
            stage_span.set_attribute("records.processed", self._processed)
            stage_span.set_attribute("records.skipped", self._skipped)
            stage_span.set_attribute("records.failed", self._failed)
            stage_span.set_attribute("stage.duration_ms", stage_duration_ms)
            stage_span.set_status(trace.Status(trace.StatusCode.OK))

        except Exception as e:
            stage_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(f"Pipeline error in stage '{self.stage_name}': {e}")
            # Cancel all pending tasks
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to complete
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            raise
        finally:
            logger.info(
                f"✅ Stage '{self.stage_name}' complete: attempted={self._attempted} "
                f"processed={self._processed} skipped={self._skipped} failed={self._failed}"
            )

    async def _check_processor_cache(self, record: BaseRecord, processor_stage_name: str) -> list[BaseRecord] | None:
        """Check cache for processor-specific outputs."""
        if not self.enable_record_cache or not self._record_cache or self.force_reprocess or not hasattr(record, "record_id"):
            return None

        logger.debug("🔍 Checking processor cache", record_id=record.record_id, processor_stage=processor_stage_name)

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
            # Restore original record_id
            restored_output = cached_output.model_copy(update={"record_id": record.record_id})
            cached_outputs.append(restored_output)
            output_index += 1

        return cached_outputs if cached_outputs else None

    def _trace_record_state(self, record: BaseRecord, stage: str, processor_class: str = "", processor_index: int = -1) -> dict:
        """Trace the current state of a record for debugging."""
        record_id = getattr(record, "record_id", "unknown")

        # Basic record info
        trace_info = {
            "record_id": record_id,
            "stage": stage,
            "processor_class": processor_class,
            "processor_index": processor_index,
        }

        # Add cache file paths for investigation
        if self._record_cache and processor_class:
            processor_stage_name = f"{self.stage_name}.{processor_index:02d}.{processor_class}"
            cache_path = self._record_cache._record_path(processor_stage_name, record_id)
            trace_info["cache_file"] = str(cache_path)
            trace_info["cache_exists"] = cache_path.exists() if hasattr(cache_path, 'exists') else False
            trace_info["cache_base_dir"] = str(self._record_cache.base_dir)
            trace_info["processor_stage_name"] = processor_stage_name

        # Get all non-private attributes of the record
        record_fields = {}
        for attr_name in dir(record):
            if not attr_name.startswith('_') and not callable(getattr(record, attr_name, None)) and not attr_name.startswith('model_'):
                try:
                    attr_value = getattr(record, attr_name, None)
                    if attr_value is not None:
                        # Handle different types of fields
                        if isinstance(attr_value, str):
                            record_fields[f"{attr_name}"] = attr_value[:20] + "..."
                        elif isinstance(attr_value, list):
                            record_fields[f"{attr_name}_count"] = len(attr_value)
                        elif isinstance(attr_value, dict):
                            record_fields[f"{attr_name}_keys"] = list(attr_value.keys())
                except Exception:
                    # Skip fields that can't be accessed
                    continue

        trace_info.update(record_fields)
        return trace_info

    async def _save_processor_cache(self, input_record: BaseRecord, outputs: list[BaseRecord], processor_stage_name: str) -> None:
        """Save processor outputs to cache."""
        if not self.enable_record_cache or not self._record_cache or not outputs or not hasattr(input_record, "record_id"):
            return

        logger.debug(
            "💾 Saving processor outputs to cache", record_id=input_record.record_id, processor_stage=processor_stage_name, outputs_count=len(outputs)
        )

        try:
            if len(outputs) == 1:
                # 1:1 transformation - use input record_id as cache key
                self._record_cache.save(outputs[0], processor_stage_name)
            else:
                # 1:N transformation - use indexed cache keys
                for output_index, output_record in enumerate(outputs):
                    cache_key = f"{input_record.record_id}_output_{output_index}"
                    cache_record = output_record.model_copy(update={"record_id": cache_key})
                    self._record_cache.save(cache_record, processor_stage_name)
        except Exception as e:
            logger.debug("💥 Failed to save processor cache", record_id=input_record.record_id, processor_stage=processor_stage_name, error=str(e))


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
