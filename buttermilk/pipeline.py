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
from buttermilk._core.types import BaseRecord


@runtime_checkable
class Processor(Protocol):
    """Standard processor interface - async generator that yields records."""

    async def process(self, record: BaseRecord) -> AsyncGenerator[BaseRecord, None]:
        """Process a record and yield zero or more output records.

        Yield nothing to filter out the record.
        Yield one record for 1:1 transformation.
        Yield multiple records for 1:N transformation.
        """
        ...


class PipelineOrchestrator(BaseModel):
    """Concurrent async pipeline orchestrator for processing records through stages.

    Simplified from DocProcessor to use BaseRecord metadata tracking instead of ProcessingResult.
    Each stage appends its status to record.metadata[stage_name].
    """

    concurrency: int = Field(default=1, description="Max concurrent record processing")
    max_records: Optional[int] = Field(default=None, description="Maximum records to process")
    stage_name: str = Field(..., description="Name for this processing stage")
    force_reprocess: bool = Field(default=False, description="Ignore cache and reprocess")

    # Inputs configured after instantiation
    source: Optional[Any] = Field(default=None, exclude=True, description="Source config or AsyncIterator")
    processors: list[Any] = Field(default_factory=list, exclude=True)  # List of processors to chain

    # Internal state
    _semaphore: asyncio.Semaphore = PrivateAttr()
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
        """Initialize semaphore."""
        self._semaphore = asyncio.Semaphore(self.concurrency)
        return self

    async def _process_single_record(self, record: BaseRecord) -> BaseRecord:
        """Process a single record through the entire processor chain.

        Each record flows through all processors individually.
        Raises exception on any error - no error forwarding.

        Args:
            record: Input record to process

        Returns:
            Final processed record with metadata

        Raises:
            Exception: If any processor fails or record is filtered out
        """
        if not self.processors:
            raise ValueError(f"[{self.stage_name}] No processors configured")

        start_time = time.time()
        current_record = record

        try:
            # Flow record through each processor in sequence
            for processor in self.processors:
                outputs = []
                async for output_record in processor.process(current_record):
                    outputs.append(output_record)

                if not outputs:
                    # Record was filtered out by this processor
                    raise ValueError(f"Record {record.record_id} filtered out by processor {processor}")
                elif len(outputs) == 1:
                    # Normal 1:1 flow
                    current_record = outputs[0]
                else:
                    # 1:N expansion - for now, take first output
                    # TODO: Handle 1:N properly in future iteration
                    current_record = outputs[0]
                    logger.warning(f"Processor yielded {len(outputs)} records, taking first one")

            # Add success metadata to final record (immutable copy)
            processing_time_ms = int((time.time() - start_time) * 1000)
            final_metadata = {
                **current_record.metadata,
                self.stage_name: {
                    "status": "processed",
                    "timestamp": time.time(),
                    "processing_time_ms": processing_time_ms,
                }
            }

            return current_record.model_copy(update={"metadata": final_metadata})

        except Exception as e:
            # Let exception bubble up - TaskGroup will handle error collection
            logger.error(f"Error processing record {record.record_id} in stage {self.stage_name}: {e}")
            raise

    async def __call__(self) -> AsyncIterator[BaseRecord]:
        """Process records from source with TaskGroup-based concurrency.

        Each record flows through the entire processor chain individually.
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
                    f"failed={self._failed} pending={pending_count}"
                )
                last_log = now

        try:
            # Ensure we have an async iterator
            source_iter = self.source if hasattr(self.source, "__anext__") else self.source.__aiter__()

            pending_tasks: set[asyncio.Task] = set()
            completed_records = asyncio.Queue()

            async def process_and_queue(record: BaseRecord):
                """Process a record and put result in queue."""
                try:
                    processed_record = await self._process_single_record(record)
                    await completed_records.put(processed_record)
                    self._processed += 1
                except Exception as e:
                    self._failed += 1
                    # Let exception bubble up for TaskGroup
                    raise

            async with asyncio.TaskGroup() as tg:
                # Producer: Create tasks for incoming records
                async def producer():
                    nonlocal pending_tasks
                    async for record in source_iter:
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
                        task = tg.create_task(process_and_queue(record))
                        pending_tasks.add(task)

                        maybe_log_status(len(pending_tasks))

                    # Signal completion by putting None
                    await completed_records.put(None)

                # Consumer: Yield completed records
                async def consumer():
                    while True:
                        record = await completed_records.get()
                        if record is None:  # End signal
                            break
                        yield record

                # Start producer
                producer_task = tg.create_task(producer())

                # Yield from consumer
                async for record in consumer():
                    yield record

            logger.info(
                f"✅ Stage '{self.stage_name}' complete: attempted={self._attempted} "
                f"processed={self._processed} failed={self._failed}"
            )

        except* Exception as eg:
            # TaskGroup collects all exceptions
            logger.error(f"Stage '{self.stage_name}' had {len(eg.exceptions)} errors")
            for exc in eg.exceptions:
                logger.error(f"Task error: {exc}")
            raise


def chain_stages(*stages: PipelineOrchestrator) -> AsyncIterator[BaseRecord]:
    """Chain multiple pipeline stages together.

    Each stage's output becomes the next stage's input.

    Args:
        *stages: Variable number of PipelineOrchestrator instances

    Returns:
        Async iterator of final processed records

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
