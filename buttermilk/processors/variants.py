"""Variant processor for running multiple processor configurations in parallel.

This module provides VariantProcessor, which takes a processor class and
variant configurations, instantiates multiple processor variants, and runs
them in parallel on each input record, yielding results as they complete.
"""

import asyncio
from typing import Any, AsyncGenerator

from opentelemetry import trace
from pydantic import Field, PrivateAttr

from buttermilk._core.log import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore


class VariantProcessor(ProcessorCore):
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
            "params": {...},      # Variant-specific config
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

    _processors: list[Any] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Instantiate variant processors and validate params after model creation."""
        from buttermilk._core.pipeline_config import ProcessorVariants
        from buttermilk.utils.validators import import_class_from_path

        # Load processor class for fail-fast validation
        try:
            processor_class = import_class_from_path(self.processor_obj)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Failed to load processor class from '{self.processor_obj}': {e}") from e

        # Fail-fast: validate variant keys against processor's declared fields
        if hasattr(processor_class, "model_fields"):
            valid_fields = set(processor_class.model_fields.keys())

            invalid_variant_keys = set(self.variants.keys()) - valid_fields
            if invalid_variant_keys:
                raise ValueError(
                    f"Processor '{processor_class.__name__}' does not accept variant params: "
                    f"{sorted(invalid_variant_keys)}. Valid fields: {sorted(valid_fields)}"
                )

            invalid_param_keys = set(self.parameters.keys()) - valid_fields
            if invalid_param_keys:
                raise ValueError(
                    f"Processor '{processor_class.__name__}' does not accept parameters: "
                    f"{sorted(invalid_param_keys)}. Valid fields: {sorted(valid_fields)}"
                )

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

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """Run all variants in parallel, yield results as they complete.

        Args:
            context: ProcessingContext with record and session state

        Yields:
            Output objects from each variant, with variant metadata added.

        Raises:
            Exception: If fail_on_error=True and any variant fails.
            Exception: If fail_on_error=False but ALL variants fail.
        """

        async def stream_variant_outputs(processor: Any, variant_idx: int, output_queue: asyncio.Queue) -> None:
            """Stream outputs from one variant processor to the queue as they're produced."""
            try:
                # Create child context for this variant
                child_context = ProcessingContext(
                    session_id=context.session_id,
                    batch_id=context.batch_id,
                    record=context.record,
                    span=trace.get_current_span(),
                    resources=context.resources,
                    metadata=context.metadata.copy(),
                )
                async for output in processor.process(child_context):
                    # Add variant metadata if output is a BaseRecord-like object
                    if hasattr(output, "metadata") and hasattr(output, "model_copy"):
                        metadata = output.metadata.copy() if output.metadata else {}
                        metadata["variant"] = {
                            "index": variant_idx,
                            "total": len(self._processors),
                            "processor_class": type(processor).__name__,
                            "params": getattr(processor, "config_dict", {}),
                        }
                        output = output.model_copy(update={"metadata": metadata})
                    await output_queue.put((variant_idx, output, None))
            except Exception as e:
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
            for task in list(tasks):
                if task.done() and not task.cancelled():
                    completed_count += 1
                    tasks.remove(task)

            # Try to get items from queue (with timeout to check task completion)
            try:
                variant_idx, output, error = await asyncio.wait_for(output_queue.get(), timeout=0.1)

                if error is not None:
                    logger.warning(
                        f"Variant {variant_idx} failed: {error}",
                        record_id=getattr(context.record, "record_id", None),
                        variant_idx=variant_idx,
                        error=str(error),
                    )

                    if self.fail_on_error:
                        for task in tasks:
                            task.cancel()
                        raise error

                    if first_error is None:
                        first_error = error
                else:
                    success_count += 1
                    yield output
            except asyncio.TimeoutError:
                continue

        # If ALL variants failed, raise the first error
        if success_count == 0 and first_error is not None:
            raise first_error

    async def flush(self) -> AsyncGenerator[Any, None]:
        """Flush any buffered records from all variant processors."""
        for proc in self._processors:
            async for output in proc.flush():
                yield output
