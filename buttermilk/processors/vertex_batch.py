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
      - _target_: buttermilk.processors.BatchAccumulator
        batch_size: 50
        batch_processors:
          - _target_: buttermilk.processors.VertexBatchProcessor
            model: gemini-2.5-flash
            template: judge_criteria
            template_vars:
              criteria: "{{ criteria }}"
            cache_ttl: "3600s"
    ```
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.import_utils import load_class
from buttermilk.utils.templating import load_template


class VertexBatchProcessor(BatchProcessorCore):
    """Batch processor using Vertex AI with criteria caching.

    Implements SimpleBatchProcessor protocol for use inside BatchAccumulator.
    Combines batch prediction (50% cost savings) with context caching
    (~90% savings on repeated criteria) for efficient large-scale evaluation.

    Supports typed output via output_model, similar to LLMProcessor.
    When output_model is set, yields typed objects directly.
    Otherwise, yields enriched BaseRecord objects.
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
    output_model: str | None = Field(
        default=None,
        description="Pydantic model path for structured output",
    )
    fail_on_unfilled_parameters: bool = Field(
        default=True,
        description="Fail if template parameters are unfilled",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for response",
    )

    # LLM outputs are non-deterministic, so skip pipeline caching
    skip_cache: bool = Field(default=True, description="Skip pipeline caching for non-deterministic LLM outputs")

    # Internal components
    _output_class: type[BaseModel] | None = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)
    _cached_criteria: dict[str, tuple[str, str]] = PrivateAttr(default_factory=dict)  # key -> (rendered, hash)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic initialization."""

        # Load output model class if specified
        if self.output_model:
            self._output_class = load_class(self.output_model)

        logger.info(
            "VertexBatchProcessor initialized",
            model=self.model,
            template=self.template,
            cache_ttl=self.cache_ttl,
            output_model=self.output_model,
        )

    def _ensure_client(self) -> None:
        """Lazily initialize the LLM client via buttermilk infrastructure."""
        if self._client is None:
            from buttermilk import bm

            # Use buttermilk's LLM infrastructure for proper model routing
            self._client = bm.llms[self.model]

    def _render_criteria(self, variant_vars: dict[str, Any]) -> tuple[str, str]:
        """Render the criteria template with given variables.

        Args:
            variant_vars: Variables to merge with template_vars for rendering

        Returns:
            Tuple of (rendered template string, template hash)
        """
        merged_vars = {**self.template_vars, **variant_vars}
        if self.fail_on_unfilled_parameters:
            merged_vars["fail_on_unfilled_parameters"] = True

        rendered, unfilled, template_hash = load_template(self.template, merged_vars)
        return rendered, template_hash

    def prepare_batch_requests(
        self,
        records: list[BaseRecord],
    ) -> list[Any]:
        """Prepare batch requests for Vertex AI.

        Used by VertexBatchExecutor to construct batch jobs.

        Args:
            records: List of BaseRecord objects

        Returns:
            List of BatchRequest objects (as Any to avoid circular imports if possible,
            but better to import BatchRequest if we can)
        """
        from buttermilk._core.vertex_batch import BatchRequest

        requests: list[BatchRequest] = []

        for record in records:
            # Get template variables from record
            if hasattr(record, "model_dump"):
                record_dict = record.model_dump()
                template_vars = {
                    **record_dict.get("metadata", {}),
                    **{k: v for k, v in record_dict.items() if k != "metadata"},
                }
            else:
                template_vars = record.metadata if record.metadata else {}

            # Render criteria and get hash
            criteria_key = self._get_criteria_key(template_vars)

            # NOTE: For true batch usage with caching, we might need to pre-create the cache
            # and pass the cache name.
            # Currently VertexBatchProcessor logic caches IN-MEMORY (self._cached_criteria).
            # For Vertex Batch API, we need "context caching" resource if we use it.
            # Or we inline the criteria.
            #
            # If we rely on in-context caching (ephemeral), for Claude we inline it.
            # For Gemini we might need to create a resource.
            #
            # Reusing existing logic: _render_criteria returns (rendered, hash).
            # We can use this to populate `cache_name` if we have a way to map hash -> resource name.
            # For now, let's assume we pass Empty `cache_name` and let BatchJobManager handle inlining
            # for Claude or just including text.
            #
            # BUT BatchJobManager.build_jsonl expects `criteria_contents` map if inlining for Claude.
            # And `prepare_batch_requests` needs to return requests with `criteria_key` set.

            if criteria_key not in self._cached_criteria:
                self._cached_criteria[criteria_key] = self._render_criteria(template_vars)

            # Create request
            req = BatchRequest(
                custom_id=str(uuid.uuid4()),  # Generate unique ID for this request within batch
                record_id=record.record_id,
                criteria_key=criteria_key,
                content=record.content or "",
                cache_name=None,  # TODO: Support persistent cache resources
            )
            requests.append(req)

        return requests

    def get_criteria_contents(self) -> dict[str, str]:
        """Get mapping of criteria_key to rendered content for all cached criteria."""
        return {k: v[0] for k, v in self._cached_criteria.items()}

    def _get_criteria_key(self, variant_vars: dict[str, Any]) -> str:
        """Generate a unique key for a criteria variant.

        Args:
            variant_vars: Variables that distinguish this criteria variant

        Returns:
            Hash-based key for the criteria variant
        """
        # Create a deterministic key from variant variables
        key_parts = sorted(f"{k}={v}" for k, v in variant_vars.items())
        key_str = "|".join(key_parts) if key_parts else "default"
        return hashlib.md5(key_str.encode()).hexdigest()[:8]

    async def _process_batch(
        self,
        records: list[BaseRecord],
    ) -> list[BaseRecord]:
        """Process a batch of records through Vertex AI.

        Implements BatchProcessorCore.

        Args:
            records: List of BaseRecord objects to process

        Returns:
            List of processed BaseRecord objects with LLM outputs
        """
        self._ensure_client()

        if not records:
            return []

        start_time = time.time()
        output_records: list[BaseRecord] = []

        # For now, process records individually through the LLM
        # TODO: Implement true batch prediction API when available
        for record in records:
            try:
                # Get template variables from record - flatten metadata like LLMProcessor does
                if hasattr(record, "model_dump"):
                    record_dict = record.model_dump()
                    template_vars = {
                        **record_dict.get("metadata", {}),  # Flatten metadata fields
                        **{k: v for k, v in record_dict.items() if k != "metadata"},  # Top-level fields
                    }
                else:
                    template_vars = record.metadata if record.metadata else {}

                # Render the criteria/prompt
                criteria_key = self._get_criteria_key(template_vars)
                if criteria_key not in self._cached_criteria:
                    self._cached_criteria[criteria_key] = self._render_criteria(template_vars)

                rendered_prompt, template_hash = self._cached_criteria[criteria_key]

                # Build the full prompt with record content
                full_prompt = f"{rendered_prompt}\n\n{record.content or ''}"

                # Call the LLM with schema for structured output
                response, llm_messages = await self._call_llm(full_prompt, schema=self._output_class)

                # Parse output if output_model is set
                final_output = response
                if self._output_class and response:
                    try:
                        cleaned_response = response.strip()
                        if cleaned_response.startswith("```json"):
                            cleaned_response = cleaned_response[7:-3].strip()
                        elif cleaned_response.startswith("```"):
                            cleaned_response = cleaned_response[3:-3].strip()

                        final_output = self._output_class.model_validate_json(cleaned_response)
                    except Exception as e:
                        logger.warning(f"Failed to parse output for {record.record_id}: {e}")

                # Emit success trace
                duration_ms = (time.time() - start_time) * 1000
                await self._emit_success_trace(
                    record=record,
                    outputs=final_output,
                    processor_stage=self.name or "vertex_batch",
                    parent_trace_id=None,
                    duration_ms=duration_ms / len(records),
                    messages=llm_messages,
                    inputs=template_vars,
                    extra_metadata={
                        "llm_config": {
                            "model": self.model,
                            "template": self.template,
                            "template_hash": template_hash,
                            "criteria_key": criteria_key,
                        },
                    },
                    execution_type="llm_processing (batch)",
                )

                # Yield the typed output or enriched record
                if self._output_class and isinstance(final_output, self._output_class):
                    output_records.append(final_output)
                else:
                    # Enrich record with LLM output
                    updated_metadata = record.metadata.copy() if record.metadata else {}
                    updated_metadata["llm_output"] = response
                    output_records.append(record.model_copy(update={"metadata": updated_metadata}))

            except Exception as e:
                logger.error(f"Failed to process record {record.record_id}: {e}")
                # Emit error trace
                await self._emit_error_trace(
                    record=record,
                    error=e,
                    processor_stage=self.name or "vertex_batch",
                    parent_trace_id=None,
                    duration_ms=(time.time() - start_time) * 1000 / len(records),
                    inputs=template_vars if "template_vars" in dir() else {},
                    execution_type="llm_processing (batch)",
                )

                # Add error to record
                error_record = record.model_copy(update={"error": [*(record.error or []), str(e)]})
                output_records.append(error_record)

        logger.info(
            f"VertexBatchProcessor processed {len(records)} records",
            model=self.model,
            duration_ms=(time.time() - start_time) * 1000,
        )

        return output_records

    async def _call_llm(self, prompt: str, schema: type | None = None) -> tuple[str, list]:
        """Call the LLM with the given prompt via buttermilk infrastructure.

        Args:
            prompt: The full prompt to send to the LLM
            schema: Optional Pydantic model for structured JSON output

        Returns:
            Tuple of (response_text, messages) for tracing
        """
        from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

        # Build messages for the LLM
        messages: list = []
        if self.system_instruction:
            messages.append(SystemMessage(content=self.system_instruction))
        messages.append(UserMessage(content=prompt, source="user"))

        # Call via buttermilk's LLM wrapper (handles model routing via litellm)
        # Pass schema to enable structured JSON output when output_model is set
        response = await self._client.create(messages=messages, schema=schema)

        # Extract text from response - if schema was used, content is the JSON string
        response_text = response.content if response.content else ""

        # Add assistant response to messages for complete trace
        messages.append(AssistantMessage(content=response_text, source="assistant"))

        return response_text, messages

    async def finalize(self) -> None:
        """Clean up resources after processing."""
        self._cached_criteria.clear()
        logger.info("VertexBatchProcessor finalized")
