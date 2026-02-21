<<<<<<< HEAD
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
=======
"""Vertex AI Batch Processor with criteria caching.

This processor combines Vertex AI's batch prediction API with explicit
context caching for efficient large-scale evaluation runs.

Key features:
- Caches criteria templates (system prompt + criteria) for reuse
- Submits batch jobs via Vertex AI Batch Prediction API (50% cost savings)
- Supports both Gemini and Claude models on Vertex AI
- Integrates with buttermilk's session save_dir for GCS operations

Usage:
>>>>>>> origin/stable
    ```yaml
    processors:
      - _target_: buttermilk.processors.BatchAccumulator
        batch_size: 50
        batch_processors:
          - _target_: buttermilk.processors.VertexBatchProcessor
            model: gemini-2.5-flash
<<<<<<< HEAD
            template: judge_template
            template_vars:
              some_var: "value"
=======
            template: judge_criteria
            template_vars:
              criteria: "{{ criteria }}"
            cache_ttl: "3600s"
>>>>>>> origin/stable
    ```
"""

from __future__ import annotations

<<<<<<< HEAD
import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any
=======
import hashlib
import time
import uuid
from typing import Any
>>>>>>> origin/stable

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
<<<<<<< HEAD
from buttermilk._core.exceptions import FatalError
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.types import BatchResult, BatchRequest
from buttermilk.utils.import_utils import load_class
from buttermilk.utils.templating import make_messages, render_template

if TYPE_CHECKING:
    from google.genai.types import BatchJob

    from buttermilk.batch.managers.vertex import BatchJobManager


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
=======
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
>>>>>>> origin/stable
    template: str = Field(..., description="Jinja2 template for criteria")
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Static variables for template rendering",
    )
<<<<<<< HEAD
=======
    system_instruction: str | None = Field(
        default=None,
        description="Optional system instruction for caching",
    )
    cache_ttl: str = Field(
        default="3600s",
        description="Cache TTL (e.g., '3600s' for 1 hour)",
    )
>>>>>>> origin/stable
    output_model: str | None = Field(
        default=None,
        description="Pydantic model path for structured output",
    )
    fail_on_unfilled_parameters: bool = Field(
        default=True,
        description="Fail if template parameters are unfilled",
    )
<<<<<<< HEAD
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
=======
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
>>>>>>> origin/stable

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic initialization."""

        # Load output model class if specified
        if self.output_model:
            self._output_class = load_class(self.output_model)
<<<<<<< HEAD
            # Pre-resolve the JSON schema for batch structured output
            self._output_schema = self._resolve_output_schema()

        logger.info(
            f"{self.__class__.__name__} initialized",
            model=self.model,
            template=self.template,
            output_model=self.output_model,
        )

    def _resolve_field(self, field_name: str, record: BaseRecord) -> Any:
        """Resolve a configuration field, checking for overrides in record metadata.

        Resolution order:
        1. record.metadata[field_name]
        2. self[field_name] (configured default)

        Args:
            field_name: Name of the field to resolve (e.g., 'model', 'template')
            record: Record to check metadata

        Returns:
            Resolved value for the field.
        """
        # 1. Check record metadata
        if record.metadata:
            if field_name in record.metadata:
                return record.metadata[field_name]
            # Also check nested 'variant_params' from VariantProcessor
            variant_params = record.metadata.get("variant_params", {})
            if isinstance(variant_params, dict) and field_name in variant_params:
                return variant_params[field_name]

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
        records: list[BaseRecord],
    ) -> list[Any]:
        """Prepare batch requests from records.

        Renders templates, converts to LiteLLM message format, and creates
        BatchRequest objects. Provider-agnostic -- works with any executor.

        Args:
            records: List of BaseRecord objects

        Returns:
            List of BatchRequest objects
        """
        from buttermilk._core.llms import autogen_to_litellm_messages
        from buttermilk.batch.types import BatchRequest

        requests: list[BatchRequest] = []

        for record in records:
            # Dynamically resolve model and template for this record
            resolved_model = self._resolve_field("model", record)
            resolved_template = self._resolve_field("template", record)

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
                    template=resolved_template,
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

            # Extract variant from record metadata if present (set by VariantProcessor or ParameterExpansionProcessor)
            # Build structured variant dict for full traceability
            variant_from_metadata = None
            if record.metadata:
                # ParameterExpansionProcessor uses variant_suffix for tracking (e.g., "criteria=tja")
                variant_suffix = record.metadata.get("variant_suffix")
                if variant_suffix:
                    variant_from_metadata = variant_suffix
                # Also check for explicit variant key (from VariantProcessor)
                elif record.metadata.get("variant"):
                    variant_from_metadata = record.metadata.get("variant")
                # Fallback for old style metadata
                elif record.metadata.get("variant_name") or record.metadata.get("instruction_type"):
                    variant_from_metadata = record.metadata.get("variant_name") or record.metadata.get("instruction_type")

            # Extract processor_index from variant metadata if present, else use configured default
            processor_index = self.processor_index  # Default from processor config
            if record.metadata:
                variant_info = record.metadata.get("variant", {})
                if isinstance(variant_info, dict) and "index" in variant_info:
                    processor_index = variant_info.get("index")

            # Build structured variant with full model info for traceability
            # This ensures manifests contain complete provenance information
            structured_variant: dict[str, Any] = {
                "template": resolved_template,
                "model": resolved_model,
            }
            # Include model parameters from config if available
            from buttermilk import bm

            if resolved_model in bm.llms.connections:
                config = bm.llms.connections[resolved_model]
                structured_variant["model_config"] = {
                    "resolved_model": config.configs.get("model", resolved_model),
                    "region": config.configs.get("region"),
                    "max_output_tokens": self._get_resolved_max_tokens(),
                }
            # Include variant from metadata (e.g., criteria name from ParameterExpansionProcessor)
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
        records: list[BaseRecord],
    ) -> list[BaseRecord]:
        """Process a batch of records.

        Base implementation raises NotImplementedError. This class is designed
        to be used with an external executor via prepare_batch_requests().
        Provider-specific subclasses (e.g. VertexBatchProcessor) can override
        this to implement direct batch submission.

        Args:
            records: List of BaseRecord objects to process

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

        Applies Vertex AI transforms (resolve , make all required,
        convert enum values for Gemini).

        Returns:
            Transformed JSON schema dict, or None if no output_class.
        """
        if not self._output_class:
            return None

        from buttermilk._core.json_schema import prepare_schema_for_vertex
        from buttermilk.batch.converters import _is_claude_model, _is_llama_model

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

=======

        logger.info(
            "VertexBatchProcessor initialized",
            model=self.model,
            template=self.template,
            cache_ttl=self.cache_ttl,
            output_model=self.output_model,
        )

>>>>>>> origin/stable
    def _ensure_client(self) -> None:
        """Lazily initialize the LLM client via buttermilk infrastructure."""
        if self._client is None:
            from buttermilk import bm

            # Use buttermilk's LLM infrastructure for proper model routing
            self._client = bm.llms[self.model]

<<<<<<< HEAD
    def _ensure_manager(self) -> BatchJobManager:
        """Lazily initialize the BatchJobManager via buttermilk infrastructure."""
        from buttermilk.batch.managers.vertex import BatchJobManager as _BatchJobManager

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
=======
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
>>>>>>> origin/stable

    async def _process_batch(
        self,
        records: list[BaseRecord],
    ) -> list[BaseRecord]:
<<<<<<< HEAD
        """Process a batch of records through Vertex AI Batch Prediction API.

        Supports mixed models in a single batch by splitting into multiple
        Vertex AI batch jobs automatically.
=======
        """Process a batch of records through Vertex AI.

        Implements BatchProcessorCore.
>>>>>>> origin/stable

        Args:
            records: List of BaseRecord objects to process

        Returns:
            List of processed BaseRecord objects with LLM outputs
        """
<<<<<<< HEAD
=======
        self._ensure_client()

>>>>>>> origin/stable
        if not records:
            return []

        start_time = time.time()
<<<<<<< HEAD

        # Prepare batch requests (handles dynamic model resolution per record)
        requests = self.prepare_batch_requests(records)

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
=======
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
>>>>>>> origin/stable
            duration_ms=(time.time() - start_time) * 1000,
        )

        return output_records

<<<<<<< HEAD
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
=======
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
>>>>>>> origin/stable
