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

    async def _process_record(self, record: BaseRecord) -> list[BaseRecord]:
        """Process a single record through the chain of processors.

        Returns a list of output records (can be empty, one, or many).
        """
        async with self._semaphore:
            start_time = time.time()

            try:
                if not self.processors:
                    logger.error(f"[{self.stage_name}] No processors configured")
                    self._failed += 1
                    return []

                # Start with the input record
                current_records = [record]

                # Chain through each processor
                for i, processor in enumerate(self.processors):
                    next_records = []

                    # Process each record from the previous stage
                    for rec in current_records:
                        async for output_record in processor.process(rec):
                            if output_record is not None:
                                next_records.append(output_record)

                    # Update current records for next processor
                    current_records = next_records

                    # If no records produced, stop the chain
                    if not current_records:
                        break

                # Track results
                if not current_records:
                    # No output from the chain
                    self._skipped += 1
                    record.metadata[self.stage_name] = {
                        "status": "skipped",
                        "timestamp": time.time(),
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                    }
                else:
                    self._processed += len(current_records)
                    # Track success in metadata for each output
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    for output_record in current_records:
                        output_record.metadata[self.stage_name] = {
                            "status": "processed",
                            "timestamp": time.time(),
                            "processing_time_ms": processing_time_ms,
                        }

                return current_records

            except Exception as e:
                logger.error(f"Error processing record {record.record_id} in stage {self.stage_name}: {e}")
                self._failed += 1

                # Track error in metadata
                record.metadata[self.stage_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time(),
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

                # Append to error list if record has one
                if hasattr(record, "error"):
                    if not isinstance(record.error, list):
                        record.error = []
                    from buttermilk._core.contract import ErrorEvent

                    record.error.append(ErrorEvent(content=f"Stage {self.stage_name}: {e}", source=self.stage_name))

                return []

    async def __call__(self) -> AsyncIterator[BaseRecord]:
        """Process records from source through the configured processor.

        Yields successfully processed records only (skipped/failed are filtered out).
        """
        if self.source is None:
            logger.error(f"[{self.stage_name}] No source iterator configured")
            return

        pending: set[asyncio.Task] = set()
        log_interval = 15.0
        last_log = time.monotonic()

        async def maybe_log_status():
            nonlocal last_log
            now = time.monotonic()
            if now - last_log >= log_interval:
                logger.debug(
                    f"📊 Stage '{self.stage_name}': attempted={self._attempted} processed={self._processed} "
                    f"skipped={self._skipped} failed={self._failed} pending={len(pending)}"
                )
                last_log = now

        try:
            # Ensure we have an async iterator
            source_iter = self.source if hasattr(self.source, "__anext__") else self.source.__aiter__()

            async for record in source_iter:
                self._attempted += 1

                # Check if we've hit max_records
                if self.max_records is not None and self._processed >= self.max_records:
                    logger.info(f"🔚 Stage '{self.stage_name}' reached max_records ({self._processed}) – stopping")
                    break

                # Maintain concurrency limit
                while len(pending) >= self.concurrency:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        results = task.result()
                        await maybe_log_status()
                        # Yield each output record
                        for result in results:
                            yield result

                # Schedule processing
                pending.add(asyncio.create_task(self._process_record(record)))

            # Process remaining tasks
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    results = task.result()
                    await maybe_log_status()
                    # Yield each output record
                    for result in results:
                        yield result

            logger.info(
                f"✅ Stage '{self.stage_name}' complete: attempted={self._attempted} "
                f"processed={self._processed} skipped={self._skipped} failed={self._failed}"
            )

        except Exception as e:
            logger.error(f"Stage '{self.stage_name}' aborted: {e}")
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
