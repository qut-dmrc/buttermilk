"""Variant Batch Processor - Config-driven cartesian expansion of models x templates.

Replaces manually-listed VertexBatchProcessor entries with a single config block
that auto-expands the cartesian product of models and templates.

Usage:
    ```yaml
    processors:
      - _target_: buttermilk.processors.BatchAccumulator
        batch_size: 100000
        batch_processors:
          - _target_: buttermilk.processors.VariantBatchProcessor
            models:
              - google/gemini-3-flash-preview
              - deepseek-ai/deepseek-v3.2-maas
            templates:
              - political_speech/satirize
              - political_speech/flyer
            wait_for_completion: false
            dry_run: false
    ```

This expands to 4 inner VertexBatchProcessor instances (2 models x 2 templates),
each reusing all existing logic for JSONL building, GCS upload, manifest creation,
and result mapping.
"""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import Field

from buttermilk import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord

from .vertex_batch import VertexBatchProcessor


class VariantBatchProcessor(BatchProcessorCore):
    """Batch processor that expands models x templates into individual batch jobs.

    Takes lists of models and templates, generates the cartesian product,
    and runs a VertexBatchProcessor for each (model, template) pair.

    Shared config (wait_for_completion, dry_run, template_vars, output_model,
    fail_on_unfilled_parameters, max_tokens) is passed through to each inner
    processor.

    Attributes:
        models: List of model identifiers to expand
        templates: List of template names to expand
    """

    models: list[str] = Field(..., description="List of model identifiers")
    templates: list[str] = Field(..., description="List of template names")

    # Shared config passed through to inner VertexBatchProcessors
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Static variables for template rendering",
    )
    output_model: str | None = Field(
        default=None,
        description="Pydantic model path for structured output",
    )
    fail_on_unfilled_parameters: bool = Field(
        default=True,
        description="Fail if template parameters are unfilled",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for response",
    )
    wait_for_completion: bool = Field(
        default=True,
        description="If True, block until batch job completes",
    )
    poll_interval: int = Field(
        default=30,
        description="Seconds between status checks when waiting",
    )
    max_wait_hours: int = Field(
        default=24,
        description="Maximum hours to wait for batch job completion",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, prepare batch requests without submitting to API",
    )

    def model_post_init(self, __context: Any) -> None:
        """Log the expansion plan."""
        total = len(self.models) * len(self.templates)
        logger.info(
            f"VariantBatchProcessor initialized: {len(self.models)} models x {len(self.templates)} templates = {total} variants",
            models=self.models,
            templates=self.templates,
        )

    async def _process_batch(
        self,
        records: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Expand models x templates and run each variant sequentially.

        For each (model, template) pair in the cartesian product:
        1. Instantiate a VertexBatchProcessor with shared config
        2. Run its _process_batch on the same contexts
        3. Collect all output records

        Args:
            records: List of ProcessingContext objects to process

        Returns:
            Combined list of output records from all variants
        """
        if not records:
            return []

        variants = list(itertools.product(self.models, self.templates))
        total = len(variants)

        logger.info(
            f"VariantBatchProcessor expanding {total} variants "
            f"({len(self.models)} models x {len(self.templates)} templates) "
            f"over {len(records)} records",
        )

        all_outputs: list[BaseRecord] = []

        for idx, (model, template) in enumerate(variants):
            logger.info(
                f"VariantBatchProcessor [{idx + 1}/{total}] model={model}, template={template}",
            )

            # Instantiate inner processor with shared config
            inner = VertexBatchProcessor(
                model=model,
                template=template,
                template_vars=self.template_vars,
                output_model=self.output_model,
                fail_on_unfilled_parameters=self.fail_on_unfilled_parameters,
                max_tokens=self.max_tokens,
                wait_for_completion=self.wait_for_completion,
                poll_interval=self.poll_interval,
                max_wait_hours=self.max_wait_hours,
                dry_run=self.dry_run,
                processor_index=idx,
                name=f"{self.name or 'variant'}_{model}_{template}" if self.name else None,
            )

            try:
                outputs = await inner._process_batch(records)
                all_outputs.extend(outputs)
                logger.info(
                    f"VariantBatchProcessor [{idx + 1}/{total}] complete: {len(outputs)} output records",
                )
            except Exception as e:
                logger.error(
                    f"VariantBatchProcessor [{idx + 1}/{total}] failed: model={model}, template={template}, error={e}",
                )
                # Continue with other variants rather than failing the whole batch
                continue

        logger.info(
            f"VariantBatchProcessor complete: {len(all_outputs)} total output records from {total} variants",
        )

        return all_outputs
