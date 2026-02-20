"""Variant processor for running multiple processor configurations in parallel.

This module provides VariantProcessor, which takes a processor class and
variant configurations, instantiates multiple processor variants, and runs
them in parallel on each input record, yielding results as they complete.
"""

import asyncio
from typing import Any, AsyncGenerator

from pydantic import Field, PrivateAttr

from buttermilk._core.log import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import RecordBufferedException


class VariantProcessor(ProcessorCore):
    """Run multiple processor variants in parallel, yielding results as they complete.

    This processor enables A/B testing of different processor configurations
    within a single pipeline run. Each variant is instantiated with different
    parameters, and all variants process each input record in parallel.

    Results stream as variants complete - fast variants don't wait for slow ones.
    By default, individual variant failures are logged but don't stop other variants.

    Attributes:
        processor_obj: Processor class path to instantiate (e.g., 'buttermilk.processors.LLMProcessor')
        variants: Parameter variations (e.g., {'model': ['gpt-4o', 'claude-3-5-sonnet']})
        parameters: Base parameters merged with variant params
        fail_on_error: If True, raise on first variant failure. If False, log and continue.

    Example:
        ```yaml
        processors:
          - _target_: buttermilk.processors.VariantProcessor
            processor_obj: buttermilk.processors.LLMProcessor
            variants:
              model: ["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"]
              temperature: [0.7]
            parameters:
              template: "default"
        ```

    Output metadata includes variant tracking:
        ```python
        record.metadata["variant"] = {
            "index": 0,           # Which variant produced this
            "total": 3,           # Total number of variants
            "processor_class": "LLMProcessor",
            "stage": "pipeline/00.VariantProcessor/abc123",
        }
        ```
    """

    processor_obj: str = Field(description="Processor class path to instantiate (e.g., 'buttermilk.processors.LLMProcessor')")
    variants: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameter variations (e.g., {'model': ['gpt-4o', 'claude-3-5-sonnet']})",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Base parameters merged with variant params",
    )
    fail_on_error: bool = Field(
        default=True,
        description="If True, raise on first variant failure. If False, log and continue.",
    )

    _processors: list[Any] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Instantiate variant processors after model creation."""
        from buttermilk._core.pipeline_config import ProcessorVariants

        variant_cfg = ProcessorVariants(
            processor_obj=self.processor_obj,
            variants=self.variants,
            parameters=self.parameters,
        )

        # Generate all configurations and instantiate processors
        # ProcessorVariants handles the cartesian product logic
        self._processors = []
        for proc_cls, cfg in variant_cfg.get_configs():
            try:
                proc = proc_cls(**cfg)
                self._processors.append(proc)
            except Exception as e:
                logger.error(
                    f"Failed to instantiate variant of {self.processor_obj}",
                    config=cfg,
                    error=str(e),
                )
                if self.fail_on_error:
                    raise

        logger.info(
            f"VariantProcessor instantiated {len(self._processors)} variants",
            processor_obj=self.processor_obj,
            variant_count=len(self._processors),
        )

    async def _process_record(  # noqa: PLR0912
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Run all variants in parallel, yield results as they complete.

        Args:
            context: Processing context containing the record to process

        Yields:
            BaseRecord outputs from each variant, with variant metadata added.

        Raises:
            Exception: If fail_on_error=True and any variant fails.
            Exception: If fail_on_error=False but ALL variants fail.
        """
        record = context.record
        processor_stage = context.session_id

        async def stream_variant_outputs(processor: Any, variant_idx: int, output_queue: asyncio.Queue) -> None:
            """Stream outputs from one variant processor to the queue as they're produced."""
            try:
                # Create a specialized context for this variant
                # We reuse the span but could create a sub-span if needed
                variant_context = ProcessingContext(
                    session_id=f"{processor_stage}_v{variant_idx}",
                    record=record,
                    batch_id=context.batch_id,
                    span=context.span,
                    ui_callback=context.ui_callback,
                )

                async for output in processor.process(variant_context):
                    processed_output = output
                    # Add variant metadata immediately and put in queue
                    # Support typed data flow: only add metadata to records that support it
                    if hasattr(output, "metadata") and hasattr(output, "model_copy"):
                        metadata = output.metadata.copy() if output.metadata else {}
                        metadata["variant"] = {
                            "index": variant_idx,
                            "total": len(self._processors),
                            "processor_class": type(processor).__name__,
                            "stage": processor_stage,
                        }
                        # Add variant parameters to metadata for traceability
                        # This allows downstream processors to see what parameters produced this record
                        if hasattr(processor, "model_dump"):
                            metadata["variant_params"] = processor.model_dump(exclude_none=True)

                        processed_output = output.model_copy(update={"metadata": metadata})

                    # Put (variant_idx, output, None) - None means no error
                    await output_queue.put((variant_idx, processed_output, None))
            except RecordBufferedException as e:
                # Issue 1: Propagate RecordBufferedException to main loop
                await output_queue.put((variant_idx, None, e))
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
        try:
            while completed_count < original_task_count or not output_queue.empty():
                # Check if any tasks completed (successfully or with error)
                # We use list(tasks) to allow removal during iteration
                for task in list(tasks):
                    if task.done():
                        completed_count += 1
                        tasks.remove(task)

                # Try to get items from queue (with timeout to check task completion)
                try:
                    # Small timeout to keep the loop responsive to task completion
                    variant_idx, output, error = await asyncio.wait_for(output_queue.get(), timeout=0.1)

                    if error is not None:
                        # Issue 1: Handle RecordBufferedException specially
                        if isinstance(error, RecordBufferedException):
                            # Propagate buffer signal immediately
                            # This cancels other variants and bubbles up to pipeline
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            raise error

                        # Handle variant failure
                        logger.warning(
                            f"Variant {variant_idx} failed: {error}",
                            record_id=getattr(record, "record_id", "unknown"),
                            variant_idx=variant_idx,
                            error=str(error),
                        )

                        if self.fail_on_error:
                            # Cancel remaining tasks and raise
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            raise error

                        # Track first error for potential re-raise if all fail
                        if first_error is None:
                            first_error = error
                    else:
                        # Success - yield the output
                        success_count += 1
                        yield output
                except asyncio.TimeoutError:
                    # No items in queue yet, continue checking tasks
                    continue
        finally:
            # Cleanup: ensure no pending tasks are leaked
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # If ALL variants failed and we haven't raised yet, raise the first error
        if success_count == 0 and first_error is not None:
            raise first_error

    async def flush(self) -> AsyncGenerator[BaseRecord, None]:
        """Delegate flush to all inner processors."""
        for processor in self._processors:
            if hasattr(processor, "flush"):
                async for output in processor.flush():
                    yield output
