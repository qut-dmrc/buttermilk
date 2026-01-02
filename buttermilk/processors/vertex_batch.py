"""Vertex AI Batch Processor with criteria caching.

This processor combines Vertex AI's batch prediction API with explicit
context caching for efficient large-scale evaluation runs.

Key features:
- Caches criteria templates (system prompt + criteria) for reuse
- Submits batch jobs via Vertex AI Batch Prediction API (50% cost savings)
- Supports both Gemini and Claude models on Vertex AI
- Integrates with buttermilk's session save_dir for GCS operations

Usage:
    ```yaml
    processors:
      - _target_: buttermilk.processors.VertexBatchProcessor
        model: gemini-2.5-flash
        template: judge_criteria
        template_vars:
          criteria: "{{ criteria }}"
        cache_ttl: "3600s"
        variants:
          criteria: ["criteria_A", "criteria_B", "criteria_C"]
    ```
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from pydantic import Field, PrivateAttr

from buttermilk import bm, logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk._core.unified_batch_processor import UnifiedBatchProcessor
from buttermilk._core.vertex_batch import BatchJobManager, BatchRequest
from buttermilk._core.vertex_caching import CriteriaCacheManager
from buttermilk.utils.templating import load_template


class VertexBatchProcessor(UnifiedBatchProcessor):
    """Batch processor using Vertex AI with criteria caching.

    Combines batch prediction (50% cost savings) with context caching
    (~90% savings on repeated criteria) for efficient large-scale evaluation.

    For each batch of records:
    1. Creates context caches for each unique criteria template
    2. Builds JSONL batch input with cache references + record content
    3. Submits batch job to Vertex AI
    4. Polls for completion and parses results
    5. Yields enriched records with LLM outputs

    Attributes:
        model: Vertex AI model (e.g., "gemini-2.5-flash", "claude-sonnet-4")
        template: Jinja2 template name or path for criteria
        template_vars: Static variables for template rendering
        cache_ttl: Cache time-to-live (default: "3600s" = 1 hour)
        output_col: Column name for LLM output in enriched records
        max_tokens: Maximum tokens for response (Claude only)
    """

    model: str = Field(..., description="Vertex AI model identifier")
    template: str = Field(..., description="Jinja2 template for criteria")
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Static variables for template rendering",
    )
    system_instruction: str | None = Field(
        default=None,
        description="Optional system instruction for caching",
    )
    cache_ttl: str = Field(
        default="3600s",
        description="Cache TTL (e.g., '3600s' for 1 hour)",
    )
    output_col: str = Field(
        default="llm_output",
        description="Column name for LLM output",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for response (Claude)",
    )

    # Internal components
    _cache_manager: CriteriaCacheManager | None = PrivateAttr(default=None)
    _batch_manager: BatchJobManager | None = PrivateAttr(default=None)
    _template_content: str | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize managers after Pydantic initialization."""
        super().model_post_init(__context)

        # Load template content
        self._template_content = load_template(self.template)

        logger.info(
            "VertexBatchProcessor initialized",
            model=self.model,
            template=self.template,
            cache_ttl=self.cache_ttl,
        )

    def _ensure_managers(self) -> None:
        """Lazily initialize cache and batch managers.

        Deferred to allow bm.genai to be available after session init.
        """
        if self._cache_manager is None:
            self._cache_manager = CriteriaCacheManager(
                client=bm.genai,
                ttl=self.cache_ttl,
                model=self.model,
            )

        if self._batch_manager is None:
            self._batch_manager = BatchJobManager(
                client=bm.genai,
            )

    def _render_criteria(self, template_vars: dict[str, Any]) -> str:
        """Render the criteria template with provided variables.

        Args:
            template_vars: Variables for template rendering

        Returns:
            Rendered criteria string
        """
        from jinja2 import Template

        if self._template_content is None:
            raise RuntimeError("Template not loaded")

        template = Template(self._template_content)
        merged_vars = {**self.template_vars, **template_vars}
        return template.render(**merged_vars)

    def _get_criteria_key(self, template_vars: dict[str, Any]) -> str:
        """Generate a key for criteria variant identification.

        Args:
            template_vars: Variables used for this criteria variant

        Returns:
            String key identifying this criteria variant
        """
        # Use the variant-specific values to create a key
        variant_parts = []
        for key, value in sorted(template_vars.items()):
            if key not in self.template_vars:  # Only variant-specific vars
                variant_parts.append(f"{key}={value}")
        return "_".join(variant_parts) if variant_parts else "default"

    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> AsyncGenerator[list[BaseRecord], None]:
        """Process a batch of records through Vertex AI batch prediction.

        Args:
            contexts: List of processing contexts containing records

        Yields:
            Lists of enriched BaseRecord objects with LLM outputs
        """
        self._ensure_managers()

        if not contexts:
            return

        # Collect unique criteria variants and their rendered content
        criteria_variants: dict[str, str] = {}  # key -> rendered content
        criteria_caches: dict[str, str] = {}  # key -> cache resource name

        # Build batch requests
        batch_requests: list[BatchRequest] = []

        for ctx in contexts:
            record = ctx.record

            # Get variant-specific template vars from record metadata
            variant_vars = ctx.metadata.get("variant_vars", {})
            criteria_key = self._get_criteria_key(variant_vars)

            # Render and cache criteria if not already done
            if criteria_key not in criteria_variants:
                rendered_criteria = self._render_criteria(variant_vars)
                criteria_variants[criteria_key] = rendered_criteria

                # Create cache for Gemini (Claude uses inline caching in batch)
                if not self._is_claude_model():
                    cache_name = self._cache_manager.get_or_create_cache(
                        criteria_content=rendered_criteria,
                        display_name=f"{self.name or 'vertex_batch'}_{criteria_key}",
                        system_instruction=self.system_instruction,
                        model=self.model,
                    )
                    criteria_caches[criteria_key] = cache_name
                    logger.debug(
                        f"Created cache for criteria {criteria_key}: {cache_name}"
                    )

            # Build batch request
            custom_id = f"{record.record_id}_{criteria_key}"
            batch_requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    record_id=record.record_id,
                    criteria_key=criteria_key,
                    content=record.content or "",
                    cache_name=criteria_caches.get(criteria_key),
                )
            )

        logger.info(
            f"Processing batch of {len(batch_requests)} requests "
            f"with {len(criteria_variants)} unique criteria variants",
            model=self.model,
        )

        # Submit batch job and wait for results
        # For Claude, pass criteria contents for inline caching
        criteria_contents = criteria_variants if self._is_claude_model() else None

        results = await self._batch_manager.run_batch_and_wait(
            model=self.model,
            requests=batch_requests,
            criteria_contents=criteria_contents,
        )

        # Map results back to records
        result_map = {r.custom_id: r for r in results}
        enriched_records: list[BaseRecord] = []

        for ctx in contexts:
            record = ctx.record
            variant_vars = ctx.metadata.get("variant_vars", {})
            criteria_key = self._get_criteria_key(variant_vars)
            custom_id = f"{record.record_id}_{criteria_key}"

            result = result_map.get(custom_id)

            if result and result.response:
                # Enrich record with LLM output
                enriched_metadata = {
                    **(record.metadata or {}),
                    f"vertex_batch_{self.name or 'processor'}": {
                        "model": self.model,
                        "criteria_key": criteria_key,
                        "usage": result.usage,
                    },
                }

                enriched_record = record.model_copy(
                    update={
                        self.output_col: result.response,
                        "metadata": enriched_metadata,
                    }
                )
                enriched_records.append(enriched_record)

            elif result and result.error:
                logger.warning(
                    f"Batch request failed for {record.record_id}: {result.error}"
                )
                # Add error to record
                error_record = record.model_copy(
                    update={
                        "error": [*(record.error or []), result.error],
                    }
                )
                enriched_records.append(error_record)
            else:
                logger.warning(f"No result found for {custom_id}")
                enriched_records.append(record)

        yield enriched_records

    def _is_claude_model(self) -> bool:
        """Check if the configured model is a Claude model.

        Returns:
            True if model is Claude/Anthropic
        """
        return "claude" in self.model.lower() or "anthropic" in self.model.lower()

    async def finalize(self) -> None:
        """Clean up resources after processing.

        Cleans up any expired caches from the local registry.
        """
        if self._cache_manager:
            self._cache_manager.cleanup_expired()

        logger.info("VertexBatchProcessor finalized")
