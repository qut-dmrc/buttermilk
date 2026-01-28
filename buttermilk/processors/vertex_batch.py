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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk._core.vertex_batch import BatchJobManager, BatchResult
from buttermilk.utils.import_utils import load_class
from buttermilk.utils.templating import load_template

if TYPE_CHECKING:
    from google.genai.types import BatchJob


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

    # Batch processing configuration
    wait_for_completion: bool = Field(
        default=True,
        description="If True, block until batch job completes. If False, return job_id immediately.",
    )
    poll_interval: int = Field(
        default=30,
        description="Seconds between status checks when waiting for batch completion",
    )
    max_wait_hours: int = Field(
        default=24,
        description="Maximum hours to wait for batch job completion before timeout",
    )

    # LLM outputs are non-deterministic, so skip pipeline caching
    skip_cache: bool = Field(default=True, description="Skip pipeline caching for non-deterministic LLM outputs")

    # Internal components
    _output_class: type[BaseModel] | None = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)
    _manager: BatchJobManager | None = PrivateAttr(default=None)
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

    def _ensure_manager(self) -> BatchJobManager:
        """Lazily initialize the BatchJobManager via buttermilk infrastructure."""
        if self._manager is None:
            from buttermilk import bm

            self._manager = BatchJobManager(
                client=bm.genai,
                poll_interval=self.poll_interval,
                max_wait_hours=self.max_wait_hours,
            )
        return self._manager

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
        """Process a batch of records through Vertex AI Batch Prediction API.

        Implements BatchProcessorCore.

        Args:
            records: List of BaseRecord objects to process

        Returns:
            List of processed BaseRecord objects with LLM outputs
        """
        if not records:
            return []

        start_time = time.time()
        manager = self._ensure_manager()

        # Prepare batch requests (this also populates self._cached_criteria)
        requests = self.prepare_batch_requests(records)
        criteria_contents = self.get_criteria_contents()

        logger.info(
            f"Submitting batch job with {len(requests)} requests",
            model=self.model,
            wait_for_completion=self.wait_for_completion,
        )

        # Submit the batch job
        job: BatchJob = await manager.submit_batch(
            model=self.model,
            requests=requests,
            criteria_contents=criteria_contents,
        )

        # Extract job_id from the job name for tracing
        batch_job_id = job.name.split("/")[-1] if job.name else "unknown"

        if not self.wait_for_completion:
            # Non-blocking mode: return records with job_id in metadata
            return self._create_pending_records(records, batch_job_id)

        # Blocking mode: wait for completion and parse results
        try:
            job = await manager.wait_for_completion(job)
        except (TimeoutError, RuntimeError) as e:
            logger.error(f"Batch job failed: {e}")
            return self._create_error_records(records, str(e), batch_job_id, start_time)

        # Parse results from GCS
        output_uri = manager._get_output_uri_for_job(job.name)
        results = manager.parse_results(output_uri, requests)

        # Map results back to records
        output_records = await self._map_results_to_records(
            records=records,
            results=results,
            batch_job_id=batch_job_id,
            start_time=start_time,
        )

        logger.info(
            f"VertexBatchProcessor processed {len(records)} records via batch API",
            model=self.model,
            batch_job_id=batch_job_id,
            duration_ms=(time.time() - start_time) * 1000,
        )

        return output_records

    def _create_pending_records(
        self,
        records: list[BaseRecord],
        batch_job_id: str,
    ) -> list[BaseRecord]:
        """Create records with pending batch job metadata for non-blocking mode.

        Args:
            records: Original input records
            batch_job_id: The batch job ID for later retrieval

        Returns:
            Records with batch_job_id in metadata (no LLM output yet)
        """
        output_records = []
        for record in records:
            updated_metadata = record.metadata.copy() if record.metadata else {}
            updated_metadata["batch_job_id"] = batch_job_id
            updated_metadata["batch_status"] = "pending"
            output_records.append(record.model_copy(update={"metadata": updated_metadata}))

        logger.info(
            f"Batch job {batch_job_id} submitted (non-blocking mode)",
            record_count=len(records),
        )
        return output_records

    def _create_error_records(
        self,
        records: list[BaseRecord],
        error_message: str,
        batch_job_id: str,
        start_time: float,
    ) -> list[BaseRecord]:
        """Create error records when batch job fails.

        Args:
            records: Original input records
            error_message: The error message
            batch_job_id: The batch job ID
            start_time: When processing started

        Returns:
            Records with error information
        """
        output_records = []
        for record in records:
            error_record = record.model_copy(
                update={"error": [*(record.error or []), error_message]}
            )
            output_records.append(error_record)

        return output_records

    async def _map_results_to_records(
        self,
        records: list[BaseRecord],
        results: list[BatchResult],
        batch_job_id: str,
        start_time: float,
    ) -> list[BaseRecord]:
        """Map batch results back to enriched records.

        Args:
            records: Original input records
            results: Batch results from the API
            batch_job_id: The batch job ID for tracing
            start_time: When processing started

        Returns:
            Enriched records with LLM outputs
        """
        # Build lookup from record_id to result
        result_map = {r.record_id: r for r in results}
        output_records: list[BaseRecord] = []
        duration_ms = (time.time() - start_time) * 1000

        for record in records:
            result = result_map.get(record.record_id)

            if result is None:
                # No result found for this record
                logger.warning(f"No batch result found for record {record.record_id}")
                error_record = record.model_copy(
                    update={"error": [*(record.error or []), "No batch result found"]}
                )
                output_records.append(error_record)
                continue

            if result.error:
                # Individual request failed
                logger.warning(f"Batch request failed for {record.record_id}: {result.error}")
                await self._emit_error_trace(
                    record=record,
                    error=Exception(result.error),
                    processor_stage=self.name or "vertex_batch",
                    parent_trace_id=None,
                    duration_ms=duration_ms / len(records),
                    inputs={},
                    execution_type="llm_processing (batch)",
                )
                error_record = record.model_copy(
                    update={"error": [*(record.error or []), result.error]}
                )
                output_records.append(error_record)
                continue

            # Parse output if output_model is set
            response = result.response or ""
            final_output: Any = response

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

            # Emit success trace with batch_job_id
            await self._emit_success_trace(
                record=record,
                outputs=final_output,
                processor_stage=self.name or "vertex_batch",
                parent_trace_id=None,
                duration_ms=duration_ms / len(records),
                messages=[],  # Batch API doesn't return full message history
                inputs={},
                extra_metadata={
                    "llm_config": {
                        "model": self.model,
                        "template": self.template,
                        "criteria_key": result.criteria_key,
                        "batch_job_id": batch_job_id,
                    },
                    "usage": result.usage,
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
                updated_metadata["batch_job_id"] = batch_job_id
                output_records.append(record.model_copy(update={"metadata": updated_metadata}))

        return output_records

    async def finalize(self) -> None:
        """Clean up resources after processing."""
        self._cached_criteria.clear()
        logger.info("VertexBatchProcessor finalized")
