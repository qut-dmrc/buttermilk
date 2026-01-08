"""Variant processor for running multiple processor configurations in parallel.

This module provides VariantProcessor, which takes a processor class and
variant configurations, instantiates multiple processor variants, and runs
them in parallel on each input record, yielding results as they complete.
"""

import asyncio
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord


class VariantProcessor(BaseModel):
    """Run multiple processor variants in parallel, yielding results as they complete.

    This processor enables A/B testing of different processor configurations
    within a single pipeline run. Each variant is instantiated with different
    parameters, and all variants process each input record in parallel.

    Results stream as variants complete - fast variants don't wait for slow ones.
    By default, individual variant failures are logged but don't stop other variants.

    Attributes:
        processor_obj: Processor class path to instantiate (e.g., 'buttermilk.processors.LLMCore')
        variants: Parameter variations (e.g., {'model': ['gpt-4', 'claude-3']})
        parameters: Base parameters merged with variant params

    Example:
        ```yaml
        processors:
          - _target_: buttermilk.processors.VariantProcessor
            processor_obj: buttermilk.processors.LLMCore
            variants:
              model: ["gpt-4", "claude-3", "gemini-pro"]
              temperature: [0.7]
            parameters:
              template: "default"
        ```

    Output metadata includes variant tracking:
        ```python
        record.metadata["variant"] = {
            "index": 0,           # Which variant produced this
            "total": 3,           # Total number of variants
            "processor_class": "LLMCore",
            "stage": "pipeline/00.VariantProcessor/abc123",
        }
        ```
    """

    processor_obj: str = Field(description="Processor class path to instantiate (e.g., 'buttermilk.processors.LLMCore')")
    variants: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameter variations (e.g., {'model': ['gpt-4', 'claude-3']})",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Base parameters merged with variant params",
    )
    fail_on_error: bool = Field(
        default=True,
        description="If True, raise on first variant failure. If False, log and continue.",
    )
    model_config = {"arbitrary_types_allowed": True}

    _processors: list[Any] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Instantiate variant processors after model creation."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        variant_cfg = ProcessorVariants(
            processor_obj=self.processor_obj,
            variants=self.variants,
            parameters=self.parameters,
        )

        self._processors = [proc_cls(**cfg) for proc_cls, cfg in variant_cfg.get_configs()]

        logger.debug(
            f"VariantProcessor instantiated {len(self._processors)} variants",
            processor_obj=self.processor_obj,
            variant_count=len(self._processors),
        )

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Run all variants in parallel, yield results as they complete.

        Args:
            record: Input record to process
            processor_stage: Unique stage identifier for caching
            parent_trace_id: Optional trace ID for distributed tracing
            **kwargs: Additional arguments passed to variant processors

        Yields:
            BaseRecord outputs from each variant, with variant metadata added.

        Raises:
            Exception: If fail_on_error=True and any variant fails.
            Exception: If fail_on_error=False but ALL variants fail.
        """

        async def stream_variant_outputs(processor: Any, variant_idx: int, output_queue: asyncio.Queue) -> None:
            """Stream outputs from one variant processor to the queue as they're produced."""
            variant_stage = f"{processor_stage}_v{variant_idx}"
            try:
                async for output in processor.process(
                    record,
                    processor_stage=variant_stage,
                    parent_trace_id=parent_trace_id,
                    **kwargs,
                ):
                    # Add variant metadata immediately and put in queue
                    metadata = output.metadata.copy() if output.metadata else {}
                    metadata["variant"] = {
                        "index": variant_idx,
                        "total": len(self._processors),
                        "processor_class": type(processor).__name__,
                        "stage": processor_stage,
                    }
                    # Put (variant_idx, output, None) - None means no error
                    await output_queue.put((variant_idx, output.model_copy(update={"metadata": metadata}), None))
            except Exception as e:
                # Put (variant_idx, None, error) - signal failure
                await output_queue.put((variant_idx, None, e))

        # Create queue for streaming results
        output_queue: asyncio.Queue = asyncio.Queue()

        # Create tasks for all variants
        tasks = [asyncio.create_task(stream_variant_outputs(proc, idx, output_queue)) for idx, proc in enumerate(self._processors)]

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
                variant_idx, output, error = await asyncio.wait_for(output_queue.get(), timeout=0.1)

                if error is not None:
                    # Handle error
                    logger.warning(
                        f"Variant {variant_idx} failed: {error}",
                        record_id=record.record_id,
                        processor_stage=processor_stage,
                        variant_idx=variant_idx,
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
                    # Success - yield the output
                    success_count += 1
                    yield output
            except asyncio.TimeoutError:
                # No items in queue yet, continue checking tasks
                continue

        # If ALL variants failed, raise the first error
        if success_count == 0 and first_error is not None:
            raise first_error
