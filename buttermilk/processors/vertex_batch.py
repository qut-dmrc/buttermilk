"""Batch LLM Processors.

This module provides:
- BatchLLMProcessor: Provider-agnostic base class for batch LLM request preparation
  and result mapping. Works with any executor (OpenAI, Vertex, etc.).
- VertexBatchProcessor: Vertex AI-specific subclass that adds batch submission via
  the Vertex AI Batch Prediction API.

Usage with OpenAI/Azure executor:
    ```python
    runner = BatchPipelineRunner(
        source=records,
        batch_processor=BatchLLMProcessor(
            model="gpt-chat",
            template="my_template",
        ),
        executor=OpenAIBatchExecutor(),
    )
    ```

Usage with Vertex AI executor (unchanged):
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

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.exceptions import FatalError
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk._core.vertex_batch import BatchResult
from buttermilk.utils.import_utils import load_class
from buttermilk.utils.templating import make_messages, render_template

if TYPE_CHECKING:
    from google.genai.types import BatchJob

    from buttermilk._core.vertex_batch import BatchJobManager


# =============================================================================
# BatchLLMProcessor -- provider-agnostic base class
# =============================================================================


class BatchLLMProcessor(BatchProcessorCore):
    """Provider-agnostic batch LLM processor.

    Handles template rendering, batch request preparation, and result-to-record
    mapping. Does NOT submit jobs to any API -- that is the executor's
    responsibility, or can be done by a provider-specific subclass like
    VertexBatchProcessor.

    This class is designed to work with any BatchExecutor (OpenAI, Vertex, etc.)
    via the prepare_batch_requests() method.
    """

    model: str = Field(..., description="Model identifier (as registered in buttermilk)")
    template: str | None = Field(default=None, description="Jinja2 template for criteria. When used with ParameterExpansionProcessor, template comes from context.variant_params.")
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
        description="Maximum tokens for response. If None, read from model config (max_output_tokens).",
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
    processor_index: int = Field(
        default=0,
        description="Index of this processor in a multi-processor pipeline (for manifest traceability)",
    )

    _output_class: type[BaseModel] | None = PrivateAttr(default=None)
    _output_schema: dict[str, Any] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic initialization."""

        # Load output model class if specified
        if self.output_model:
            self._output_class = load_class(self.output_model)
            # Pre-resolve the JSON schema for batch structured output
            self._output_schema = self._resolve_output_schema()

        logger.info(
            f"{self.__class__.__name__} initialized",
            model=self.model,
            template=self.template,
            output_model=self.output_model,
        )

    def _resolve_field(self, field_name: str, context: ProcessingContext) -> Any:
        """Resolve a configuration field, checking context.variant_params first.

        Resolution order:
        1. context.variant_params[field_name] (from ParameterExpansionProcessor)
        2. self[field_name] (configured default)

        Args:
            field_name: Name of the field to resolve (e.g., 'model', 'template')
            context: ProcessingContext with variant_params

        Returns:
            Resolved value for the field.
        """
        # 1. Check context variant_params (set by BatchAccumulator from _variant_params)
        if context.variant_params and field_name in context.variant_params:
            return context.variant_params[field_name]

        # 2. Fallback to configured default
        return getattr(self, field_name, None)

    def _resolve_output_schema(self) -> dict[str, Any] | None:
        """Resolve output_class to a JSON schema dict.

        Base implementation returns raw Pydantic JSON schema, suitable for
        OpenAI/Azure structured output. Subclasses (e.g. VertexBatchProcessor)
        can override to apply provider-specific transforms.

        Returns:
            JSON schema dict, or None if no output_class.
        """
        if not self._output_class:
            return None

        return self._output_class.model_json_schema()

    def _get_resolved_max_tokens(self) -> int:
        """Resolve max_tokens from explicit config or model registry.

        Returns:
            max_tokens value - explicit if set, otherwise from model config,
            falling back to 4096 as default.
        """
        # If explicitly set, use that value
        if self.max_tokens is not None:
            return self.max_tokens

        # Try to read from model config
        from buttermilk import bm

        if self.model in bm.llms.connections:
            config = bm.llms.connections[self.model]
            config_max_tokens = config.configs.get("max_output_tokens")
            if config_max_tokens is not None:
                return config_max_tokens

        # Default fallback
        return 4096

    def prepare_batch_requests(
        self,
        contexts: list[ProcessingContext],
    ) -> list[Any]:
        """Prepare batch requests from contexts.

        Renders templates, converts to LiteLLM message format, and creates
        BatchRequest objects. Provider-agnostic -- works with any executor.

        Args:
            contexts: List of ProcessingContext objects

        Returns:
            List of BatchRequest objects
        """
        from buttermilk._core.llms import autogen_to_litellm_messages
        from buttermilk._core.vertex_batch import BatchRequest

        requests: list[BatchRequest] = []

        for context in contexts:
            record = context.record

            # Dynamically resolve model and template from context variant_params
            resolved_model = self._resolve_field("model", context)
            resolved_template = self._resolve_field("template", context)

            if not resolved_template:
                logger.warning(f"Skipping record {record.record_id}: no template resolved (not in variant_params or config)")
                continue

            # Prepare template variables
            # Mix in record fields so template can access {{ record.foo }} or {{ foo }}
            if hasattr(record, "model_dump"):
                record_dict = record.model_dump()
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

            # Also merge variant_params into template vars for template rendering
            if context.variant_params:
                variant_vars.update(context.variant_params)

            # Additional context for template if needed
            variant_vars["record"] = record

            # Render template to get full message history
            try:
                result = render_template(
                    template=resolved_template,
                    template_vars=variant_vars,
                    base_template_vars=self.template_vars,
                    fail_on_unfilled=self.fail_on_unfilled_parameters,
                )

                messages, _ = make_messages(result.rendered)
            except Exception as e:
                logger.warning(f"Failed to render template for record {record.record_id}: {e}")
                continue

            if not messages:
                logger.warning(f"Skipping record {record.record_id}: generated no messages")
                continue

            # Convert to LiteLLM format (standardized intermediate format)
            litellm_messages = autogen_to_litellm_messages(messages)

            # Extract variant info for traceability
            variant_from_metadata = None
            if record.metadata:
                variant_suffix = record.metadata.get("variant_suffix")
                if variant_suffix:
                    variant_from_metadata = variant_suffix
                elif record.metadata.get("variant"):
                    variant_from_metadata = record.metadata.get("variant")
                elif record.metadata.get("variant_name") or record.metadata.get("instruction_type"):
                    variant_from_metadata = record.metadata.get("variant_name") or record.metadata.get("instruction_type")

            # Extract processor_index from variant metadata if present
            processor_index = self.processor_index
            if record.metadata:
                variant_info = record.metadata.get("variant", {})
                if isinstance(variant_info, dict) and "index" in variant_info:
                    processor_index = variant_info.get("index")

            # Build structured variant with full model info for traceability
            structured_variant: dict[str, Any] = {
                "template": resolved_template,
                "model": resolved_model,
            }
            from buttermilk import bm

            if resolved_model in bm.llms.connections:
                config = bm.llms.connections[resolved_model]
                structured_variant["model_config"] = {
                    "resolved_model": config.configs.get("model", resolved_model),
                    "region": config.configs.get("region"),
                    "max_output_tokens": self._get_resolved_max_tokens(),
                }
            if variant_from_metadata:
                if isinstance(variant_from_metadata, dict):
                    structured_variant["metadata_variant"] = variant_from_metadata
                else:
                    structured_variant["metadata_variant"] = str(variant_from_metadata)

            req = BatchRequest(
                custom_id=str(uuid.uuid4()),
                record_id=record.record_id,
                messages=litellm_messages,
                model=resolved_model,
                variant=structured_variant,
                processor_index=processor_index,
                response_schema=self._output_schema,
            )
            requests.append(req)

        if not requests:
            raise FatalError("No valid records found for batch processing")

        return requests

    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Process a batch of contexts.

        Base implementation raises NotImplementedError. This class is designed
        to be used with an external executor via prepare_batch_requests().
        Provider-specific subclasses (e.g. VertexBatchProcessor) can override
        this to implement direct batch submission.

        Args:
            contexts: List of ProcessingContext objects to process

        Returns:
            List of processed BaseRecord objects with LLM outputs
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._process_batch() is not implemented. "
            f"Use an executor (e.g. OpenAIBatchExecutor, VertexBatchExecutor) "
            f"with prepare_batch_requests() instead, or use VertexBatchProcessor "
            f"for direct Vertex AI batch submission."
        )

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
                    processor_stage=self.name or "batch_llm",
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
                processor_stage=self.name or "batch_llm",
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
                    "cost_usd": result.cost_usd,
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
                if result.cost_usd is not None:
                    updated_metadata["cost_usd"] = result.cost_usd
                output_records.append(record.model_copy(update={"metadata": updated_metadata}))

        return output_records


# =============================================================================
# VertexBatchProcessor -- Vertex AI-specific subclass
# =============================================================================


class VertexBatchProcessor(BatchLLMProcessor):
    """Batch processor using Vertex AI Batch Prediction API.

    Extends BatchLLMProcessor with Vertex AI-specific batch submission,
    schema resolution, and dry-run support.

    Uses batch prediction for 50% cost savings on large-scale evaluation.

    Supports typed output via output_model, similar to LLMProcessor.
    When output_model is set, yields typed objects directly.
    Otherwise, yields enriched BaseRecord objects.
    """

    dry_run: bool = Field(
        default=False,
        description="If True, prepare and log batch requests without submitting to API",
    )

    # Vertex-specific internal components
    _client: Any = PrivateAttr(default=None)
    _manager: BatchJobManager | None = PrivateAttr(default=None)

    def _resolve_output_schema(self) -> dict[str, Any] | None:
        """Resolve output_class to a Vertex AI-appropriate JSON schema dict.

        Applies Vertex AI transforms (resolve $refs, make all required,
        convert enum values for Gemini).

        Returns:
            Transformed JSON schema dict, or None if no output_class.
        """
        if not self._output_class:
            return None

        from buttermilk._core.json_schema import prepare_schema_for_vertex
        from buttermilk._core.vertex_batch import _is_claude_model, _is_llama_model

        # Gemini requires enum values converted to strings; Claude and Llama do not
        is_gemini = not _is_claude_model(self.model) and not _is_llama_model(self.model)
        return prepare_schema_for_vertex(self._output_class, is_gemini=is_gemini)

    def _get_resolved_region(self) -> str | None:
        """Resolve region from model registry config.

        Returns:
            Region string from model config, or None if not configured.
        """
        from buttermilk import bm

        if self.model in bm.llms.connections:
            config = bm.llms.connections[self.model]
            return config.configs.get("region")

        return None

    def _ensure_client(self) -> None:
        """Lazily initialize the LLM client via buttermilk infrastructure."""
        if self._client is None:
            from buttermilk import bm

            # Use buttermilk's LLM infrastructure for proper model routing
            self._client = bm.llms[self.model]

    def _ensure_manager(self) -> BatchJobManager:
        """Lazily initialize the BatchJobManager via buttermilk infrastructure."""
        from buttermilk._core.vertex_batch import BatchJobManager as _BatchJobManager

        if self._manager is None:
            from google import genai

            from buttermilk import bm

            # Resolve short alias to full model name and get config
            resolved_model = self.model
            resolved_region = self._get_resolved_region()

            if self.model in bm.llms.connections:
                config = bm.llms.connections[self.model]
                if "model" in config.configs:
                    resolved_model = config.configs["model"]

            # Gemini 3 models require the global endpoint
            if "gemini-3" in resolved_model.lower():
                project_id = bm.cloud_manager.gcp_cloud_cfg.project_id
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location="global",
                )
                logger.info(f"Using global endpoint for Gemini 3 model: {self.model} -> {resolved_model}")
            elif resolved_region:
                # Use region from model config if available
                project_id = bm.cloud_manager.gcp_cloud_cfg.project_id
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=resolved_region,
                )
                logger.info(f"Using region {resolved_region} for model: {self.model}")
            else:
                client = bm.genai

            self._manager = _BatchJobManager(
                client=client,
                poll_interval=self.poll_interval,
                max_wait_hours=self.max_wait_hours,
            )
        return self._manager

    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Process a batch of contexts through Vertex AI Batch Prediction API.

        Supports mixed models in a single batch by splitting into multiple
        Vertex AI batch jobs automatically.

        Args:
            contexts: List of ProcessingContext objects to process

        Returns:
            List of processed BaseRecord objects with LLM outputs
        """
        if not contexts:
            return []

        start_time = time.time()
        records = [ctx.record for ctx in contexts]

        # Prepare batch requests (handles dynamic model resolution per context)
        requests = self.prepare_batch_requests(contexts)

        # Dry-run mode: log prepared requests and return placeholder records
        if self.dry_run:
            return self._handle_dry_run(records, requests)

        # Group requests by model for Vertex AI (which requires one model per job)
        from collections import defaultdict

        requests_by_model = defaultdict(list)
        for req in requests:
            requests_by_model[req.model].append(req)

        manager = self._ensure_manager()

        # Internal helper to handle a single model's batch job
        async def handle_model_batch(model_name: str, model_requests: list[Any]) -> list[BatchResult]:
            logger.info(
                f"Submitting batch job for model {model_name} with {len(model_requests)} requests",
                model=model_name,
                wait_for_completion=self.wait_for_completion,
            )

            # Submit the batch job
            job: BatchJob = await manager.submit_batch(
                model=model_name,
                requests=model_requests,
            )

            # Extract job_id from the job name for tracing
            batch_job_id = job.name.split("/")[-1] if job.name else "unknown"

            if not self.wait_for_completion:
                # In non-blocking mode, we can't easily return results now
                # We return a special result indicating pending status
                return [
                    BatchResult(
                        custom_id=req.custom_id,
                        record_id=req.record_id,
                        error="PENDING",
                        model=model_name,
                    )
                    for req in model_requests
                ]

            # Blocking mode: wait for completion and parse results
            try:
                completed_job = await manager.wait_for_completion(job)
            except (TimeoutError, RuntimeError) as e:
                logger.error(f"Batch job for {model_name} failed: {e}")
                return [
                    BatchResult(
                        custom_id=req.custom_id,
                        record_id=req.record_id,
                        error=str(e),
                        model=model_name,
                    )
                    for req in model_requests
                ]

            # Combined processing: join manifest, calculate costs, and save combined results/summary to GCS
            processed = manager.process_job_results(batch_job_id)
            return processed["batch_results"]

        # Run all model batches in parallel
        logger.info(
            f"VertexBatchProcessor splitting {len(requests)} requests across {len(requests_by_model)} models",
            models=list(requests_by_model.keys()),
        )

        all_model_tasks = [handle_model_batch(m, reqs) for m, reqs in requests_by_model.items()]
        batch_results_lists = await asyncio.gather(*all_model_tasks)

        # Flatten results
        all_results = [res for sublist in batch_results_lists for res in sublist]

        # In non-blocking mode, some results might be placeholders with "PENDING" error
        # We need to handle this by creating pending records
        if not self.wait_for_completion:
            # For simplicity, if ANY job was non-blocking, we mark all records as pending
            # (In practice, wait_for_completion is a processor-level flag)
            # Use a dummy job ID or the first one found
            dummy_job_id = "multi_batch_pending"
            return self._create_pending_records(records, dummy_job_id)

        # Map results back to records
        output_records = await self._map_results_to_records(
            records=records,
            results=all_results,
            batch_job_id="multi_batch",
            start_time=start_time,
        )

        logger.info(
            f"VertexBatchProcessor processed {len(records)} records via {len(requests_by_model)} batch API jobs",
            batch_job_count=len(requests_by_model),
            duration_ms=(time.time() - start_time) * 1000,
        )

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
        resolved_max_tokens = self._get_resolved_max_tokens()
        jsonl_content = manager.build_jsonl(requests, self.model, max_tokens=resolved_max_tokens)

        # Generate a dry-run job ID and upload to GCS
        dry_run_job_id = f"dry_run_{uuid.uuid4().hex[:12]}"
        batch_dir = manager._resolve_batch_dir(dry_run_job_id)
        input_uri = f"{batch_dir}/input.jsonl"

        # Upload the JSONL file to GCS
        result_uri = upload_text(jsonl_content, uri=input_uri, content_type="application/jsonl")

        # Save manifest for job recovery inspection during dry run
        # Pass explicit region since dry_run vertex_job_name isn't a real Vertex path
        resolved_region = self._get_resolved_region()
        if resolved_region is None:
            # Check if Gemini 3 model (uses global endpoint)
            from buttermilk import bm

            resolved_model = self.model
            if self.model in bm.llms.connections:
                config = bm.llms.connections[self.model]
                if "model" in config.configs:
                    resolved_model = config.configs["model"]
            if "gemini-3" in resolved_model.lower():
                resolved_region = "global"

        output_uri = manager._get_output_uri(dry_run_job_id)
        manager._save_manifest(
            job_id=dry_run_job_id,
            vertex_job_name=f"dry_run_{dry_run_job_id}",
            model=self.model,
            input_uri=result_uri,
            output_uri=output_uri,
            requests=requests,
            region=resolved_region,
        )

        logger.info(
            "[DRY RUN] Batch file written to GCS",
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


# =============================================================================
# OpenAIBatchProcessor -- Azure OpenAI / OpenAI-specific subclass
# =============================================================================


class OpenAIBatchProcessor(BatchLLMProcessor):
    """Batch processor using OpenAI / Azure OpenAI Batch API.

    Extends BatchLLMProcessor with OpenAI-specific batch submission via
    the OpenAI Batch API. Works with both direct OpenAI and Azure OpenAI.

    Uses batch prediction for cost savings on large-scale evaluation.
    Azure OpenAI requires a GlobalBatch deployment (not GlobalStandard).

    Supports typed output via output_model, similar to VertexBatchProcessor.
    When output_model is set, yields typed objects directly.
    Otherwise, yields enriched BaseRecord objects.
    """

    dry_run: bool = Field(
        default=False,
        description="If True, prepare and log batch requests without submitting to API",
    )

    # OpenAI-specific internal components
    _openai_client: Any = PrivateAttr(default=None)
    _manager: Any = PrivateAttr(default=None)

    def _ensure_openai_client(self) -> Any:
        """Lazily initialize the OpenAI/AzureOpenAI client from buttermilk model registry."""
        if self._openai_client is not None:
            return self._openai_client

        from buttermilk import bm

        if self.model not in bm.llms.connections:
            raise ValueError(
                f"Model '{self.model}' not found in buttermilk LLM registry. "
                f"Available models: {list(bm.llms.connections.keys())}"
            )

        config = bm.llms.connections[self.model]

        if config.client_type.value not in ("azure", "openai"):
            raise ValueError(
                f"OpenAIBatchProcessor requires an OpenAI or Azure model, "
                f"but '{self.model}' has client_type='{config.client_type.value}'"
            )

        if config.client_type.value == "azure":
            from openai import AzureOpenAI

            api_version = config.configs.get("api_version", "2024-12-01-preview")
            self._openai_client = AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.base_url,
                api_version=api_version,
            )
        else:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self._openai_client = OpenAI(**kwargs)

        return self._openai_client

    def _ensure_manager(self) -> Any:
        """Lazily initialize the OpenAIBatchJobManager."""
        if self._manager is not None:
            return self._manager

        from buttermilk._core.vertex_batch import OpenAIBatchJobManager

        from buttermilk import bm

        client = self._ensure_openai_client()
        config = bm.llms.connections[self.model]

        # Azure uses /chat/completions, direct OpenAI uses /v1/chat/completions
        endpoint = "/chat/completions" if config.client_type.value == "azure" else "/v1/chat/completions"

        self._manager = OpenAIBatchJobManager(
            client=client,
            endpoint=endpoint,
            poll_interval=self.poll_interval,
            max_wait_hours=self.max_wait_hours,
        )
        return self._manager

    def _get_batch_model_name(self) -> str:
        """Resolve the actual deployment/model name for batch API requests.

        For Azure, this is the deployment name (from configs.model or configs.deployment).
        For direct OpenAI, this is the model name (e.g., "gpt-4o-mini").
        """
        from buttermilk import bm

        if self.model in bm.llms.connections:
            config = bm.llms.connections[self.model]
            # Azure deployments use a specific deployment name
            deployment = config.configs.get("deployment") or config.configs.get("model")
            if deployment:
                return deployment

        return self.model

    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Process a batch of contexts through OpenAI/Azure OpenAI Batch API.

        Args:
            contexts: List of ProcessingContext objects to process

        Returns:
            List of processed BaseRecord objects with LLM outputs
        """
        if not contexts:
            return []

        start_time = time.time()
        records = [ctx.record for ctx in contexts]

        # Prepare batch requests (handles dynamic model resolution per context)
        requests = self.prepare_batch_requests(contexts)

        # Dry-run mode: log prepared requests and return placeholder records
        if self.dry_run:
            return self._handle_dry_run(records, requests)

        manager = self._ensure_manager()
        batch_model = self._get_batch_model_name()

        logger.info(
            f"OpenAIBatchProcessor submitting {len(requests)} requests",
            model=self.model,
            batch_model=batch_model,
            wait_for_completion=self.wait_for_completion,
        )

        # Submit the batch
        resolved_max_tokens = self._get_resolved_max_tokens()

        if self.wait_for_completion:
            # Blocking: submit, wait, download results
            try:
                results = await manager.run_batch_and_wait(
                    model=batch_model,
                    requests=requests,
                    metadata={"processor": "OpenAIBatchProcessor", "buttermilk_model": self.model},
                    max_tokens=resolved_max_tokens,
                )
            except (TimeoutError, RuntimeError) as e:
                logger.error(f"OpenAI batch job failed: {e}")
                return self._create_error_records(records, str(e), "openai_batch_error", start_time)

            # Map results back to records
            output_records = await self._map_results_to_records(
                records=records,
                results=results,
                batch_job_id="openai_batch",
                start_time=start_time,
            )
        else:
            # Non-blocking: submit and return pending records
            try:
                submit_result = await manager.submit_batch(
                    model=batch_model,
                    requests=requests,
                    metadata={"processor": "OpenAIBatchProcessor", "buttermilk_model": self.model},
                    max_tokens=resolved_max_tokens,
                )
                batch_job_id = submit_result.get("openai_batch_id", "unknown")
            except Exception as e:
                logger.error(f"OpenAI batch submission failed: {e}")
                return self._create_error_records(records, str(e), "openai_batch_error", start_time)

            output_records = self._create_pending_records(records, batch_job_id)

        logger.info(
            f"OpenAIBatchProcessor processed {len(records)} records",
            model=self.model,
            duration_ms=(time.time() - start_time) * 1000,
        )

        return output_records

    def _handle_dry_run(
        self,
        records: list[BaseRecord],
        requests: list[Any],
    ) -> list[BaseRecord]:
        """Handle dry-run mode: build JSONL without submitting.

        Args:
            records: Original input records
            requests: Prepared batch requests

        Returns:
            Records with dry_run metadata
        """
        manager = self._ensure_manager()
        batch_model = self._get_batch_model_name()
        resolved_max_tokens = self._get_resolved_max_tokens()

        jsonl_content = manager.build_jsonl(requests, batch_model, max_tokens=resolved_max_tokens)

        logger.info(
            f"[DRY RUN] OpenAI batch prepared with {len(requests)} requests",
            model=self.model,
            batch_model=batch_model,
            record_count=len(records),
        )

        # Try to save to storage for inspection
        dry_run_uri = None
        try:
            dry_run_job_id = f"openai_dry_run_{uuid.uuid4().hex[:12]}"
            batch_dir = manager._resolve_batch_dir(dry_run_job_id)
            input_uri = f"{batch_dir}/input.jsonl"

            from buttermilk.utils.save import upload_text

            dry_run_uri = upload_text(jsonl_content, uri=input_uri, content_type="application/jsonl")
            logger.info(f"[DRY RUN] Batch file written to {dry_run_uri}")

            # Save manifest for traceability (matches Vertex dry run behaviour)
            manager._save_manifest(
                job_id=dry_run_job_id,
                openai_batch_id=f"dry_run_{dry_run_job_id}",
                model=self.model,
                input_file_id=dry_run_uri,
                requests=requests,
                metadata={"processor": "OpenAIBatchProcessor", "dry_run": "true"},
            )
        except Exception as e:
            logger.warning(f"[DRY RUN] Could not save batch file to storage: {e}")

        # Log sample requests
        for i, req in enumerate(requests[:5]):
            logger.debug(
                f"[DRY RUN] Request {i + 1}/{len(requests)}",
                record_id=req.record_id,
                custom_id=req.custom_id,
                message_count=len(req.messages) if req.messages else 0,
            )

        # Return records with dry_run metadata
        output_records = []
        for record in records:
            updated_metadata = record.metadata.copy() if record.metadata else {}
            updated_metadata["dry_run"] = True
            updated_metadata["batch_status"] = "dry_run"
            updated_metadata["model"] = self.model
            updated_metadata["template"] = self.template
            if dry_run_uri:
                updated_metadata["dry_run_uri"] = dry_run_uri
            output_records.append(record.model_copy(update={"metadata": updated_metadata}))

        return output_records
