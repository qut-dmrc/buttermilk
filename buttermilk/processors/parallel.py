"""Parallel processor for running multiple processors on the same input in parallel.

Unlike VariantProcessor which runs variants of a single processor class,
ParallelProcessor runs different processor instances in parallel on the same input.
This is useful for A/B testing across different processor types or combining
LLM-based and API-based classifiers.
"""

import asyncio
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord


class ParallelProcessor(BaseModel):
    """Run multiple different processors in parallel on the same input record.

    This processor enables combining different processor types (e.g., LLMCore + ToxicityModel)
    within a single pipeline stage. Each processor runs independently on the original
    input record, and all results are yielded as they complete.

    Unlike chained processors where output flows from one to the next, ParallelProcessor
    runs all processors on the SAME input record, yielding all outputs together.

    Attributes:
        processors: List of processor configurations (instantiated via hydra)
        fail_on_error: If True, raise on first processor failure. If False, log and continue.

    Example:
        ```yaml
        processors:
          - _target_: buttermilk.processors.ParallelProcessor
            processors:
              - _target_: buttermilk.processors.VariantProcessor
                processor_obj: buttermilk._core.llm_core.LLMCore
                variants:
                  model: [gpt-4, claude-3]
              - _target_: buttermilk.toxicity.Cope
        ```
    """

    processors: list[Any] = Field(
        default_factory=list,
        description="List of processor instances to run in parallel",
    )
    fail_on_error: bool = Field(
        default=True,
        description="If True, raise on first processor failure. If False, log and continue but raise if ALL fail.",
    )

    model_config = {"arbitrary_types_allowed": True}

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Run all processors in parallel on the same input, yield results as they complete.

        Args:
            record: Input record to process (same record goes to all processors)
            processor_stage: Unique stage identifier for caching
            parent_trace_id: Optional trace ID for distributed tracing
            **kwargs: Additional arguments passed to processors

        Yields:
            BaseRecord outputs from all processors, with parallel metadata added

        Raises:
            Exception: If fail_on_error=True and any processor fails.
            Exception: If fail_on_error=False but ALL processors fail.
        """

        async def stream_processor_outputs(processor: Any, proc_idx: int, output_queue: asyncio.Queue) -> None:
            """Stream outputs from one processor to the queue as they're produced."""
            proc_stage = f"{processor_stage}_p{proc_idx}"
            try:
                async for output in processor.process(
                    record,
                    processor_stage=proc_stage,
                    parent_trace_id=parent_trace_id,
                    **kwargs,
                ):
                    # Add parallel metadata immediately and put in queue
                    metadata = output.metadata.copy() if output.metadata else {}
                    metadata["parallel"] = {
                        "processor_index": proc_idx,
                        "total_processors": len(self.processors),
                        "processor_class": type(processor).__name__,
                        "stage": processor_stage,
                    }
                    await output_queue.put((proc_idx, output.model_copy(update={"metadata": metadata}), None))
            except Exception as e:
                # Put error in queue
                await output_queue.put((proc_idx, None, e))

        # Create queue for streaming results
        output_queue: asyncio.Queue = asyncio.Queue()

        # Create tasks for all processors
        tasks = [asyncio.create_task(stream_processor_outputs(proc, idx, output_queue)) for idx, proc in enumerate(self.processors)]

        logger.debug(
            f"ParallelProcessor running {len(tasks)} processors in parallel",
            processor_stage=processor_stage,
            processor_count=len(tasks),
        )

        # Track original task count to know when all tasks are done
        original_task_count = len(tasks)
        completed_count = 0
        success_count = 0
        first_error: Exception | None = None

        # Yield results as they arrive in the queue
        while completed_count < original_task_count or not output_queue.empty():
            # Check if any tasks completed (successfully or with error)
            for task in list(tasks):  # Iterate over copy to allow removal
                if task.done() and not task.cancelled():
                    completed_count += 1
                    tasks.remove(task)

            # Try to get items from queue (with timeout to check task completion)
            try:
                proc_idx, output, error = await asyncio.wait_for(output_queue.get(), timeout=0.1)

                if error is not None:
                    # Handle error
                    logger.warning(
                        f"Processor {proc_idx} failed in parallel execution",
                        processor_stage=processor_stage,
                        processor_idx=proc_idx,
                        processor_class=type(self.processors[proc_idx]).__name__ if proc_idx >= 0 else "unknown",
                        error=str(error),
                    )

                    if self.fail_on_error:
                        # Cancel remaining tasks and raise
                        for task in tasks:
                            task.cancel()
                        raise error

                    # Track first error for potential re-raise
                    if first_error is None:
                        first_error = error
                else:
                    # Yield successful output immediately
                    success_count += 1
                    yield output

            except asyncio.TimeoutError:
                # No items in queue yet, continue checking tasks
                continue

        # If ALL processors failed, raise the first error
        if success_count == 0 and first_error is not None:
            raise first_error
