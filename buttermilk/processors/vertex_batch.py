"""Vertex AI Batch Processor.

This processor uses Vertex AI's batch prediction API for efficient large-scale
evaluation runs.

Key features:
- Submits batch jobs via Vertex AI Batch Prediction API (50% cost savings)
- Supports both Gemini and Claude models on Vertex AI
- Integrates with buttermilk's session save_dir for GCS operations
- Template rendering matches LLMCore behavior

Usage:
    ```yaml
    processors:
      - _target_: buttermilk.processors.BatchAccumulator
        batch_size: 50
        batch_processors:
          - _target_: buttermilk.processors.VertexBatchProcessor
            model: gemini-2.5-flash
            template: judge_template
            template_vars:
              some_var: "value"
    ```
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.exceptions import FatalError
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk._core.vertex_batch import BatchJobManager, BatchResult
from buttermilk.utils.import_utils import load_class
from buttermilk.utils.templating import make_messages, render_template

if TYPE_CHECKING:
    from google.genai.types import BatchJob


class VertexBatchProcessor(BatchProcessorCore):
    """Batch processor using Vertex AI Batch Prediction API.

    Implements SimpleBatchProcessor protocol for use inside BatchAccumulator.
    Uses batch prediction for 50% cost savings on large-scale evaluation.

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
    dry_run: bool = Field(
        default=False,
        description="If True, prepare and log batch requests without submitting to API",
    )

    # Internal components
    _output_class: type[BaseModel] | None = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)
    _manager: BatchJobManager | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic initialization."""

        # Load output model class if specified
        if self.output_model:
            self._output_class = load_class(self.output_model)

        logger.info(
            "VertexBatchProcessor initialized",
            model=self.model,
            template=self.template,
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
            from google import genai

            from buttermilk import bm

            # Resolve short alias to full model name to check if it's Gemini 3
            resolved_model = self.model
            if self.model in bm.llms.connections:
                config = bm.llms.connections[self.model]
                resolved_model = config.configs.get("model", self.model)

            # Gemini 3 models require the global endpoint
            if "gemini-3" in resolved_model.lower():
                project_id = bm.cloud_manager.gcp_cloud_cfg.project_id
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location="global",
                )
                logger.info(f"Using global endpoint for Gemini 3 model: {self.model} -> {resolved_model}")
            else:
                client = bm.genai

            self._manager = BatchJobManager(
                client=client,
                poll_interval=self.poll_interval,
                max_wait_hours=self.max_wait_hours,
            )
        return self._manager

    def prepare_batch_requests(
        self,
        records: list[BaseRecord],
    ) -> list[Any]:
        """Prepare batch requests for Vertex AI.

        Used by VertexBatchExecutor to construct batch jobs.

        Args:
            records: List of BaseRecord objects

        Returns:
            List of BatchRequest objects
        """
        from buttermilk._core.llms import autogen_to_litellm_messages
        from buttermilk._core.vertex_batch import BatchRequest

        requests: list[BatchRequest] = []

        for record in records:
            # Prepare template variables
            # Mix in record fields so template can access {{ record.foo }} or {{ foo }}
            if hasattr(record, "model_dump"):
                record_dict = record.model_dump()
                # Prioritize record fields, but allow explicit template_vars to override?
                # Usually we want record data to be available.
                # LLMCore strategy: merge record fields into template_vars.
                variant_vars = {
                    **self.template_vars,
                    **record_dict.get("metadata", {}),
                    **{k: v for k, v in record_dict.items() if k != "metadata"},
                }
            else:
                variant_vars = {
                    **self.template_vars,
                    **(record.metadata if record.metadata else {}),
                }

            # Additional context for template if needed
            variant_vars["record"] = record

            # Render template to get full message history
            # This uses the standard LLMCore logic via utility functions
            try:
                result = render_template(
                    template=self.template,
                    template_vars=variant_vars,
                    base_template_vars=self.template_vars,
                    fail_on_unfilled=self.fail_on_unfilled_parameters,
                )

                messages, _ = make_messages(result.rendered)
            except Exception as e:
                logger.warning(f"Failed to render template for record {record.record_id}: {e}")
                # We could skip or add error record. For now, skip to avoid blocking batch.
                continue

            if not messages:
                logger.warning(f"Skipping record {record.record_id}: generated no messages")
                continue

            # Convert to LiteLLM format (standardized intermediate format)
            # BatchJobManager will convert this to provider-specific (Gemini/Claude) format
            litellm_messages = autogen_to_litellm_messages(messages)

            req = BatchRequest(
                custom_id=str(uuid.uuid4()),
                record_id=record.record_id,
                messages=litellm_messages,
            )
            requests.append(req)

        if not requests:
            raise FatalError("No valid records found for batch processing")

        return requests

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

        # Prepare batch requests
        requests = self.prepare_batch_requests(records)

        # Dry-run mode: log prepared requests and return placeholder records
        # Check before _ensure_manager() to avoid requiring bm initialization
        if self.dry_run:
            return self._handle_dry_run(records, requests)

        manager = self._ensure_manager()

        logger.info(
            f"Submitting batch job with {len(requests)} requests",
            model=self.model,
            wait_for_completion=self.wait_for_completion,
        )

        # Submit the batch job
        job: BatchJob = await manager.submit_batch(
            model=self.model,
            requests=requests,
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
            error_record = record.model_copy(update={"error": [*(record.error or []), error_message]})
            output_records.append(error_record)

        return output_records

    def _handle_dry_run(
        self,
        records: list[BaseRecord],
        requests: list[Any],
    ) -> list[BaseRecord]:
        """Handle dry-run mode: write batch JSONL to GCS without submitting to API.

        Writes the prepared batch file to GCS for inspection, validation, or
        later submission. This allows reviewing the exact payload that would
        be sent to the Vertex AI Batch API.

        Args:
            records: Original input records
            requests: Prepared batch requests

        Returns:
            Records with dry_run metadata including the GCS URI of the batch file
        """
        import uuid

        from buttermilk.utils.save import upload_text

        logger.info(
            f"[DRY RUN] Preparing batch job with {len(requests)} requests",
            model=self.model,
            template=self.template,
            record_count=len(records),
        )

        # Build JSONL content using the same logic as real submission
        manager = self._ensure_manager()
        jsonl_content = manager.build_jsonl(requests, self.model)

        # Generate a dry-run job ID and upload to GCS
        dry_run_job_id = f"dry_run_{uuid.uuid4().hex[:12]}"
        batch_dir = manager._resolve_batch_dir(dry_run_job_id)
        input_uri = f"{batch_dir}/input.jsonl"

        # Upload the JSONL file to GCS
        result_uri = upload_text(jsonl_content, uri=input_uri, content_type="application/jsonl")

        logger.info(
            f"[DRY RUN] Batch file written to GCS",
            uri=result_uri,
            request_count=len(requests),
            model=self.model,
        )

        # Log summary of requests for inspection
        for i, req in enumerate(requests[:5]):  # Log first 5 requests
            logger.debug(
                f"[DRY RUN] Request {i + 1}/{len(requests)}",
                record_id=req.record_id,
                custom_id=req.custom_id,
                message_count=len(req.messages) if req.messages else 0,
            )
        if len(requests) > 5:
            logger.debug(f"[DRY RUN] ... and {len(requests) - 5} more requests")

        # Return records with dry_run metadata including the GCS URI
        output_records = []
        for record in records:
            updated_metadata = record.metadata.copy() if record.metadata else {}
            updated_metadata["dry_run"] = True
            updated_metadata["batch_status"] = "dry_run"
            updated_metadata["model"] = self.model
            updated_metadata["template"] = self.template
            updated_metadata["dry_run_uri"] = result_uri
            updated_metadata["dry_run_job_id"] = dry_run_job_id
            output_records.append(record.model_copy(update={"metadata": updated_metadata}))

        logger.info(
            f"[DRY RUN] Complete - {len(records)} records prepared, batch file at {result_uri}",
            model=self.model,
            dry_run_uri=result_uri,
        )

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
                error_record = record.model_copy(update={"error": [*(record.error or []), "No batch result found"]})
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
                error_record = record.model_copy(update={"error": [*(record.error or []), result.error]})
                output_records.append(error_record)
                continue

            # Parse output if output_model is set
            # NS: TODO: this logic is duplicated in llm_processor or llmCore. Use the proper json parsing util we made.
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
