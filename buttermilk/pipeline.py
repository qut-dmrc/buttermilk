"""Async pipeline orchestrator extracted and simplified from vector.py.

This module provides a concurrent async pipeline orchestrator that processes
records through stages, tracking metadata and errors without complex result objects.
"""

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, Optional, Any
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import pydantic
from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord


class PipelineOrchestrator(BaseModel):
    """Concurrent async pipeline orchestrator for processing records through stages.

    Simplified from DocProcessor to use BaseRecord metadata tracking instead of ProcessingResult.
    Each stage appends its status to record.metadata[stage_name].
    """

    concurrency: int = Field(default=20, description="Max concurrent record processing")
    max_records: Optional[int] = Field(default=None, description="Maximum records to process")
    stage_name: str = Field(..., description="Name for this processing stage")
    force_reprocess: bool = Field(default=False, description="Ignore cache and reprocess")

    # Inputs configured after instantiation
    source: Optional[AsyncIterator[BaseRecord]] = Field(default=None, exclude=True)
    processor: Optional[Callable[[BaseRecord], Awaitable[Optional[BaseRecord]]]] = Field(
        default=None, exclude=True
    )

    # Internal state
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _attempted: int = PrivateAttr(default=0)
    _processed: int = PrivateAttr(default=0)
    _skipped: int = PrivateAttr(default=0)
    _failed: int = PrivateAttr(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="after")
    def _init(self):
        """Initialize semaphore."""
        self._semaphore = asyncio.Semaphore(self.concurrency)
        return self

    async def _process_record(self, record: BaseRecord) -> Optional[BaseRecord]:
        """Process a single record with metadata tracking."""
        async with self._semaphore:
            start_time = time.time()

            try:
                if self.processor is None:
                    logger.error(f"[{self.stage_name}] No processor configured")
                    self._failed += 1
                    return None

                logger.info(
                    f"🔷 [{self.stage_name}-{record.record_id}] Processing record"
                )

                # Process the record
                result = await self.processor(record)

                if result is None:
                    # Record was filtered out
                    self._skipped += 1
                    # Still track in metadata that we attempted processing
                    record.metadata[self.stage_name] = {
                        "status": "skipped",
                        "timestamp": time.time(),
                        "processing_time_ms": int((time.time() - start_time) * 1000)
                    }
                    return None

                # Success - track in metadata
                processing_time_ms = int((time.time() - start_time) * 1000)
                result.metadata[self.stage_name] = {
                    "status": "processed",
                    "timestamp": time.time(),
                    "processing_time_ms": processing_time_ms
                }

                self._processed += 1
                return result

            except Exception as e:
                logger.error(f"Error processing record {record.record_id} in stage {self.stage_name}: {e}")
                self._failed += 1

                # Track error in metadata
                record.metadata[self.stage_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time(),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }

                # Append to error list if record has one
                if hasattr(record, 'error'):
                    if not isinstance(record.error, list):
                        record.error = []
                    from buttermilk._core.contract import ErrorEvent
                    record.error.append(ErrorEvent(
                        content=f"Stage {self.stage_name}: {e}",
                        source=self.stage_name
                    ))

                return None

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
                    logger.info(
                        f"🔚 Stage '{self.stage_name}' reached max_records ({self._processed}) – stopping"
                    )
                    break

                # Maintain concurrency limit
                while len(pending) >= self.concurrency:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        result = task.result()
                        await maybe_log_status()
                        if result is not None:
                            yield result

                # Schedule processing
                pending.add(asyncio.create_task(self._process_record(record)))

            # Process remaining tasks
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result = task.result()
                    await maybe_log_status()
                    if result is not None:
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
        stages[i].source = stages[i-1]()

    # Return the final stage's iterator
    return stages[-1]()