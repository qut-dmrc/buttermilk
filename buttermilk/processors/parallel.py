"""Parallel processor for running multiple processors on the same input in parallel.

Unlike VariantProcessor which runs variants of a single processor class,
ParallelProcessor runs different processor instances in parallel on the same input.
This is useful for A/B testing across different processor types or combining
LLM-based and API-based classifiers.
"""

import asyncio
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field, PrivateAttr

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
        default=False,
        description="If True, raise on first processor failure. If False, log and continue.",
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
            Exception: If fail_on_error=True and any processor fails
        """

        async def collect_processor_outputs(
            processor: Any, proc_idx: int
        ) -> tuple[int, list[BaseRecord]]:
            """Collect all outputs from one processor."""
            proc_stage = f"{processor_stage}_p{proc_idx}"
            outputs: list[BaseRecord] = []
            async for output in processor.process(
                record,
                processor_stage=proc_stage,
                parent_trace_id=parent_trace_id,
                **kwargs,
            ):
                outputs.append(output)
            return proc_idx, outputs

        # Create tasks for all processors
        tasks = [
            asyncio.create_task(collect_processor_outputs(proc, idx))
            for idx, proc in enumerate(self.processors)
        ]

        logger.debug(
            f"ParallelProcessor running {len(tasks)} processors in parallel",
            processor_stage=processor_stage,
            processor_count=len(tasks),
        )

        # Yield results as each processor completes
        for coro in asyncio.as_completed(tasks):
            proc_idx = -1
            try:
                proc_idx, outputs = await coro
                processor_class = type(self.processors[proc_idx]).__name__

                for output in outputs:
                    # Add parallel metadata
                    metadata = output.metadata.copy() if output.metadata else {}
                    metadata["parallel"] = {
                        "processor_index": proc_idx,
                        "total_processors": len(self.processors),
                        "processor_class": processor_class,
                        "stage": processor_stage,
                    }
                    yield output.model_copy(update={"metadata": metadata})

            except Exception as e:
                if self.fail_on_error:
                    raise
                # Yield error record so pipeline can track the failure
                failed_idx = proc_idx if proc_idx >= 0 else -1
                error_record = record.model_copy(
                    update={
                        "error": record.error + [str(e)] if record.error else [str(e)],
                        "metadata": {
                            **record.metadata,
                            "parallel": {
                                "processor_index": failed_idx,
                                "total_processors": len(self.processors),
                                "processor_class": type(self.processors[failed_idx]).__name__ if failed_idx >= 0 else "unknown",
                                "stage": processor_stage,
                                "failed": True,
                            },
                        },
                    }
                )
                logger.warning(
                    "Processor failed in parallel execution, continuing with others",
                    processor_stage=processor_stage,
                    processor_idx=failed_idx,
                    error=str(e),
                )
                yield error_record
