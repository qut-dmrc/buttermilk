"""OpenAI Batch Processor.

This processor uses OpenAI's Batch API for efficient large-scale evaluation runs.
Supports both OpenAI (including xAI/Grok) and Azure OpenAI.

Key features:
- Submits batch jobs via OpenAI Batch API (50% cost savings)
- Supports OpenAI, Azure OpenAI, and compatible providers (e.g., xAI)
- Integrates with buttermilk's session save_dir for GCS operations
- Template rendering matches LLMCore behavior
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
from buttermilk._core.vertex_batch import BatchRequest, BatchResult, OpenAIBatchJobManager
from buttermilk.utils.import_utils import load_class
from buttermilk.utils.templating import make_messages, render_template

if TYPE_CHECKING:
    pass


class OpenAIBatchProcessor(BatchProcessorCore):
    """Batch processor using OpenAI Batch API.

    Implements SimpleBatchProcessor protocol for use inside BatchAccumulator.
    Uses batch prediction for 50% cost savings on large-scale evaluation.

    Supports typed output via output_model, similar to LLMProcessor.
    When output_model is set, yields typed objects directly.
    Otherwise, yields enriched BaseRecord objects.
    """

    model: str = Field(..., description="OpenAI/Azure model identifier")
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
    dry_run: bool = Field(
        default=False,
        description="If True, prepare and log batch requests without submitting to API",
    )
    processor_index: int = Field(
        default=0,
        description="Index of this processor in a multi-processor pipeline (for manifest traceability)",
    )

    # Internal components
    _output_class: type[BaseModel] | None = PrivateAttr(default=None)
    _output_schema: dict[str, Any] | None = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)
    _manager: OpenAIBatchJobManager | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic initialization."""
        # Load output model class if specified
        if self.output_model:
            self._output_class = load_class(self.output_model)
            # Pre-resolve the JSON schema for batch structured output
            self._output_schema = self._resolve_output_schema()

        logger.info(
            "OpenAIBatchProcessor initialized",
            model=self.model,
            template=self.template,
            output_model=self.output_model,
        )

    def _resolve_output_schema(self) -> dict[str, Any] | None:
        """Resolve output_class to a JSON schema dict.

        Applies necessary transformations (make all required, resolve refs).
        """
        if not self._output_class:
            return None

        from buttermilk._core.json_schema import make_all_properties_required, resolve_json_schema_refs

        # OpenAI requires strict schema adherence
        schema = self._output_class.model_json_schema()
        schema = resolve_json_schema_refs(schema)
        schema = make_all_properties_required(schema)

        # Ensure additionalProperties: false is set for strict mode
        # This is handled by make_all_properties_required if configured correctly,
        # but OpenAI requires strict: true in response_format which implies it.

        return schema

    def _get_resolved_max_tokens(self) -> int | None:
        """Resolve max_tokens from explicit config or model registry."""
        if self.max_tokens is not None:
            return self.max_tokens

        from buttermilk import bm

        if self.model in bm.llms.connections:
            config = bm.llms.connections[self.model]
            return config.configs.get("max_output_tokens")

        return None

    def _ensure_manager(self) -> OpenAIBatchJobManager:
        """Lazily initialize the OpenAIBatchJobManager."""
        if self._manager is None:
            import openai

            from buttermilk import bm
            from buttermilk._core.llms import ClientType

            if self.model not in bm.llms.connections:
                # Fallback to direct OpenAI client if not in registry (e.g. ad-hoc model name)
                # Assume standard OpenAI environment variables
                logger.warning(f"Model {self.model} not found in registry, assuming standard OpenAI")
                client = openai.OpenAI()
                endpoint = "/v1/chat/completions"
            else:
                config = bm.llms.connections[self.model]

                # Determine client type and endpoint
                if config.client_type == ClientType.AZURE:
                    api_key = config.api_key
                    # For Azure, base_url is the endpoint (e.g. https://resource.openai.azure.com/)
                    azure_endpoint = config.base_url
                    api_version = config.configs.get("api_version", "2024-06-01")  # Default to a recent version

                    logger.info(f"Using Azure OpenAI client for {self.model} (endpoint: {azure_endpoint})")
                    client = openai.AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=azure_endpoint,
                        api_version=api_version,
                    )
                    # Azure OpenAI Batch API uses /chat/completions as the inner URL
                    endpoint = "/chat/completions"

                else:
                    # Standard OpenAI or compatible (xAI, etc.)
                    api_key = config.api_key
                    base_url = config.base_url  # Could be xAI URL or None (default OpenAI)

                    logger.info(f"Using OpenAI client for {self.model} (base_url: {base_url})")
                    client = openai.OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                    )
                    endpoint = "/v1/chat/completions"

            self._manager = OpenAIBatchJobManager(
                client=client,
                endpoint=endpoint,
                poll_interval=self.poll_interval,
                max_wait_hours=self.max_wait_hours,
            )

        return self._manager

    def prepare_batch_requests(
        self,
        records: list[BaseRecord],
    ) -> list[Any]:
        """Prepare batch requests."""
        from buttermilk._core.llms import autogen_to_litellm_messages

        requests: list[BatchRequest] = []

        for record in records:
            # Prepare template variables
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

            variant_vars["record"] = record

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
                continue

            if not messages:
                logger.warning(f"Skipping record {record.record_id}: generated no messages")
                continue

            litellm_messages = autogen_to_litellm_messages(messages)

            # Extract variant info
            variant_from_metadata = None
            if record.metadata:
                variant_suffix = record.metadata.get("variant_suffix")
                if variant_suffix:
                    variant_from_metadata = variant_suffix
                elif record.metadata.get("variant"):
                    variant_from_metadata = record.metadata.get("variant")
                elif record.metadata.get("variant_name"):
                    variant_from_metadata = record.metadata.get("variant_name")

            processor_index = self.processor_index
            if record.metadata:
                variant_info = record.metadata.get("variant", {})
                if isinstance(variant_info, dict) and "index" in variant_info:
                    processor_index = variant_info.get("index")

            structured_variant: dict[str, Any] = {
                "template": self.template,
                "model": self.model,
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
                model=self.model,
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
        """Process a batch of records through OpenAI Batch API."""
        if not records:
            return []

        start_time = time.time()

        requests = self.prepare_batch_requests(records)

        if self.dry_run:
            return self._handle_dry_run(records, requests)

        manager = self._ensure_manager()

        logger.info(
            f"Submitting OpenAI batch job with {len(requests)} requests",
            model=self.model,
            wait_for_completion=self.wait_for_completion,
        )

        try:
            submit_result = await manager.submit_batch(
                model=self.model,
                requests=requests,
                max_tokens=self._get_resolved_max_tokens(),
            )

            openai_batch_id = submit_result["openai_batch_id"]
            job_id = submit_result["job_id"]

            if not self.wait_for_completion:
                return self._create_pending_records(records, job_id)

            await manager.wait_for_completion(openai_batch_id)

            results = manager.download_results(openai_batch_id, requests)

            # Calculate costs for each result (download_results returns raw results without cost)
            from buttermilk.utils.pricing import calculate_token_cost

            for result in results:
                if result.usage:
                    _, _, cost = calculate_token_cost(
                        model=self.model,
                        usage_dict=result.usage,
                    )
                    result.cost_usd = cost

            output_records = await self._map_results_to_records(
                records=records,
                results=results,
                batch_job_id=job_id,
                start_time=start_time,
            )

            logger.info(
                f"OpenAIBatchProcessor processed {len(records)} records",
                model=self.model,
                batch_job_id=job_id,
                duration_ms=(time.time() - start_time) * 1000,
            )

            return output_records

        except Exception as e:
            logger.error(f"Batch job failed: {e}")
            # Use a generic job_id for error records if we didn't get one
            job_id = locals().get("job_id", "unknown_failed_job")
            return self._create_error_records(records, str(e), job_id, start_time)

    def _create_pending_records(self, records: list[BaseRecord], batch_job_id: str) -> list[BaseRecord]:
        output_records = []
        for record in records:
            updated_metadata = record.metadata.copy() if record.metadata else {}
            updated_metadata["batch_job_id"] = batch_job_id
            updated_metadata["batch_status"] = "pending"
            output_records.append(record.model_copy(update={"metadata": updated_metadata}))
        return output_records

    def _create_error_records(self, records: list[BaseRecord], error_message: str, batch_job_id: str, start_time: float) -> list[BaseRecord]:
        output_records = []
        for record in records:
            error_record = record.model_copy(update={"error": [*(record.error or []), error_message]})
            output_records.append(error_record)
        return output_records

    def _handle_dry_run(self, records: list[BaseRecord], requests: list[Any]) -> list[BaseRecord]:
        """Handle dry-run mode."""
        from buttermilk.utils.save import upload_text

        manager = self._ensure_manager()
        jsonl_content = manager.build_jsonl(requests, self.model, max_tokens=self._get_resolved_max_tokens())

        dry_run_job_id = f"dry_run_{uuid.uuid4().hex[:12]}"
        batch_dir = manager._resolve_batch_dir(dry_run_job_id)
        input_uri = f"{batch_dir}/input.jsonl"

        result_uri = upload_text(jsonl_content, uri=input_uri, content_type="application/jsonl")

        logger.info(
            "[DRY RUN] OpenAI Batch file written to GCS",
            uri=result_uri,
            request_count=len(requests),
            model=self.model,
        )

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

        return output_records

    async def _map_results_to_records(
        self,
        records: list[BaseRecord],
        results: list[BatchResult],
        batch_job_id: str,
        start_time: float,
    ) -> list[BaseRecord]:
        """Map batch results back to enriched records."""
        result_map = {r.record_id: r for r in results}
        output_records: list[BaseRecord] = []
        duration_ms = (time.time() - start_time) * 1000

        for record in records:
            result = result_map.get(record.record_id)

            if result is None:
                error_record = record.model_copy(update={"error": [*(record.error or []), "No batch result found"]})
                output_records.append(error_record)
                continue

            if result.error:
                await self._emit_error_trace(
                    record=record,
                    error=Exception(result.error),
                    processor_stage=self.name or "openai_batch",
                    parent_trace_id=None,
                    duration_ms=duration_ms / len(records),
                    inputs={},
                    execution_type="llm_processing (batch)",
                )
                error_record = record.model_copy(update={"error": [*(record.error or []), result.error]})
                output_records.append(error_record)
                continue

            response = result.response or ""
            final_output: Any = response

            if self._output_class and response:
                try:
                    # OpenAI structured output is already JSON string, validate it
                    final_output = self._output_class.model_validate_json(response)
                except Exception as e:
                    logger.warning(f"Failed to parse output for {record.record_id}: {e}")

            await self._emit_success_trace(
                record=record,
                outputs=final_output,
                processor_stage=self.name or "openai_batch",
                parent_trace_id=None,
                duration_ms=duration_ms / len(records),
                messages=[],
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

            if self._output_class and isinstance(final_output, self._output_class):
                output_records.append(final_output)
            else:
                updated_metadata = record.metadata.copy() if record.metadata else {}
                updated_metadata["llm_output"] = response
                updated_metadata["batch_job_id"] = batch_job_id
                if result.cost_usd is not None:
                    updated_metadata["cost_usd"] = result.cost_usd
                output_records.append(record.model_copy(update={"metadata": updated_metadata}))

        return output_records
