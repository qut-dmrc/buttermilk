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
        num_runs: Number of times to replicate each variant configuration
        parameters: Base parameters merged with variant params
        fail_on_error: If True, raise on first variant failure. If False, log and continue.

    Example:
        ```yaml
        processors:
          - _target_: buttermilk.processors.VariantProcessor
            processor_obj: buttermilk.processors.LLMCore
            variants:
              model: ["gpt-4", "claude-3", "gemini-pro"]
              temperature: [0.7]
            parameters:
              prompt_template: "default"
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

    processor_obj: str = Field(
        description="Processor class path to instantiate (e.g., 'buttermilk.processors.LLMCore')"
    )
    variants: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameter variations (e.g., {'model': ['gpt-4', 'claude-3']})",
    )
    num_runs: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate each variant configuration",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Base parameters merged with variant params",
    )
    fail_on_error: bool = Field(
        default=False,
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
            num_runs=self.num_runs,
            parameters=self.parameters,
        )

        self._processors = [
            proc_cls(**cfg) for proc_cls, cfg in variant_cfg.get_configs()
        ]

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
            BaseRecord outputs from each variant, with variant metadata added

        Raises:
            Exception: If fail_on_error=True and any variant fails
        """

        async def collect_variant_outputs(
            processor: Any, variant_idx: int
        ) -> tuple[int, list[BaseRecord]]:
            """Collect all outputs from one variant processor."""
            variant_stage = f"{processor_stage}_v{variant_idx}"
            outputs: list[BaseRecord] = []
            async for output in processor.process(
                record,
                processor_stage=variant_stage,
                parent_trace_id=parent_trace_id,
                **kwargs,
            ):
                outputs.append(output)
            return variant_idx, outputs

        # Create tasks for all variants
        tasks = [
            asyncio.create_task(collect_variant_outputs(proc, idx))
            for idx, proc in enumerate(self._processors)
        ]

        # Yield results as each variant completes
        for coro in asyncio.as_completed(tasks):
            try:
                variant_idx, outputs = await coro
                processor_class = type(self._processors[variant_idx]).__name__

                for output in outputs:
                    # Add variant metadata
                    metadata = output.metadata.copy() if output.metadata else {}
                    metadata["variant"] = {
                        "index": variant_idx,
                        "total": len(self._processors),
                        "processor_class": processor_class,
                        "stage": processor_stage,
                    }
                    yield output.model_copy(update={"metadata": metadata})

            except Exception as e:
                if self.fail_on_error:
                    raise
                logger.warning(
                    "Variant failed in parallel execution, continuing with others",
                    processor_stage=processor_stage,
                    variant_idx=variant_idx if "variant_idx" in dir() else "unknown",
                    error=str(e),
                )
