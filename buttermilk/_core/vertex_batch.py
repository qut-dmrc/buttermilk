"""Vertex AI batch prediction utilities.
<!-- NS TODO: let's fix the name collision between this and processors.vertex_batch -->
This module provides utilities for submitting and managing batch prediction
jobs on Vertex AI, with support for both Gemini and Claude models.

Uses buttermilk's existing save utilities for GCS operations.

Usage:
    from buttermilk._core.vertex_batch import BatchJobManager
    from buttermilk import bm

    manager = BatchJobManager(client=bm.genai)
    job = await manager.submit_batch(
        model="gemini-2.5-flash",
        requests=requests,
    )
    results = await manager.wait_for_results(job)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk.utils.pricing import calculate_token_cost
from buttermilk.utils.save import scrub_serializable, upload_json, upload_text

if TYPE_CHECKING:
    from google.genai.types import BatchJob

from buttermilk._core.log import logger


class BatchRequest(BaseModel):
    """A single request in a batch job.

    Attributes:
        custom_id: Unique identifier for mapping results back to input
        record_id: Original record ID from the source data
        messages: The messages to send to the model (LiteLLM format)
        model: Model identifier used for this request (for result analysis)
        variant: Variant identifier (e.g., label string or structured dict)
        processor_index: Index of this processor in a multi-processor pipeline
    """

    custom_id: str
    record_id: str
    messages: list[dict[str, Any]]
    model: str | None = None
    variant: str | dict[str, Any] | None = None
    processor_index: int | None = None


class BatchResult(BaseModel):
    """Result from a batch job request.

    Attributes:
        custom_id: The custom_id from the request
        record_id: Original record ID
        response: The model's response content
        error: Error message if request failed
        usage: Token usage information
        model: Model identifier used for this request (from BatchRequest)
        variant: Variant identifier (from BatchRequest)
        processor_index: Processor index (from BatchRequest)
    """

    custom_id: str
    record_id: str
    response: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    variant: str | dict[str, Any] | None = None
    processor_index: int | None = None
    cost_usd: float | None = None

    @property
    def composite_key(self) -> str:
        """Generate a composite key for unique identification across variants.

        Format: {record_id}[_{variant}][_{model}][_{processor_index}]
        Only includes non-None components.

        Returns:
            str: Composite key for unique identification
        """
        parts = [self.record_id]
        if self.variant:
            if isinstance(self.variant, dict):
                # For dict variants, generate a stable string representation
                # Focus on identifying keys/values
                variant_str = json.dumps(self.variant, sort_keys=True)
                parts.append(variant_str)
            else:
                parts.append(str(self.variant))
        if self.model:
            parts.append(self.model)
        if self.processor_index is not None:
            parts.append(str(self.processor_index))
        return "_".join(parts)


# =============================================================================
# Message Format Converters
# =============================================================================
# Unified abstraction for provider-specific batch request/response formatting.
# Each provider (Gemini, Claude) implements these interfaces to handle their
# specific message format requirements while sharing common logic.

# Model name patterns that indicate Claude/Anthropic models
_CLAUDE_MODEL_PATTERNS = ("claude", "anthropic")


class BatchMessageConverter(ABC):
    """Abstract base for converting LiteLLM messages to provider-specific batch format."""

    @abstractmethod
    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Convert a BatchRequest to provider-specific JSONL entry.

        Args:
            request: BatchRequest with messages in LiteLLM format

        Returns:
            Dictionary ready for JSONL serialization
        """

    @abstractmethod
    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response text from provider-specific batch result.

        Args:
            entry: Result entry from batch output

        Returns:
            Extracted text response or None
        """


class GeminiMessageConverter(BatchMessageConverter):
    """Convert messages to/from Gemini batch format.

    Gemini format:
    - System messages → "system_instruction": {"parts": [{"text": ...}]}
    - User/Assistant → "contents": [{"role": "user"|"model", "parts": [{"text": ...}]}]
    - Role mapping: "assistant" → "model"
    """

    # Gemini uses "model" instead of "assistant"
    ROLE_MAP = {"assistant": "model"}

    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Build a Gemini batch request entry."""
        contents = []
        system_parts = []

        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_parts.append({"text": content})
            else:
                # Map role and wrap content in Gemini's parts structure
                gemini_role = self.ROLE_MAP.get(role, role)
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        entry: dict[str, Any] = {
            "custom_id": request.custom_id,
            "request": {"contents": contents},
        }

        if system_parts:
            entry["request"]["system_instruction"] = {"parts": system_parts}

        return entry

    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response from Gemini batch result.

        Path: response.candidates[0].content.parts[0].text
        """
        response = entry.get("response", {})
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text")
        return None


class ClaudeMessageConverter(BatchMessageConverter):
    """Convert messages to/from Claude batch format.

    Claude format:
    - System messages → "system": "concatenated text"
    - User/Assistant → "messages": [{"role": "user"|"assistant", "content": ...}]
    - Requires: anthropic_version, max_tokens
    """

    DEFAULT_MAX_TOKENS = 4096
    ANTHROPIC_VERSION = "vertex-2023-10-16"

    def __init__(self, max_tokens: int | None = None):
        self.max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Build a Claude batch request entry."""
        messages = []
        system_parts = []

        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_parts.append(content)
            else:
                # Claude keeps role names as-is, content is direct string
                messages.append({"role": role, "content": content})

        request_body: dict[str, Any] = {
            "anthropic_version": self.ANTHROPIC_VERSION,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if system_parts:
            # Claude concatenates multiple system messages
            request_body["system"] = "\n\n".join(system_parts)

        return {
            "custom_id": request.custom_id,
            "request": request_body,
        }

    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response from Claude batch result.

        Path: response.content[0].text (where type=="text")
        """
        response = entry.get("response", {})
        content = response.get("content", [])
        if content and isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    return block.get("text")
        return None


def _is_claude_model(model: str) -> bool:
    """Check if model identifier indicates a Claude/Anthropic model."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in _CLAUDE_MODEL_PATTERNS)


def get_message_converter(model: str, **kwargs: Any) -> BatchMessageConverter:
    """Factory function to get the appropriate converter for a model.

    Args:
        model: Model identifier (e.g., "gemini-2.5-flash", "claude-sonnet-4")
        **kwargs: Provider-specific options (e.g., max_tokens for Claude)

    Returns:
        Appropriate BatchMessageConverter instance
    """
    if _is_claude_model(model):
        return ClaudeMessageConverter(max_tokens=kwargs.get("max_tokens"))
    return GeminiMessageConverter()


class BatchJobManifest(BaseModel):
    """Persistent manifest for batch job recovery after process restart.

    Stored at `{save_dir}/batch/{job_id}/manifest.json` alongside the input.jsonl
    and output/ directory. Contains all context needed to reconstruct job state
    and map results back to source records.

    This enables:
    - Job recovery after process restart (fetch by job_id)
    - Result mapping without re-accessing source data (self-contained)
    - Audit trail for batch processing runs

    Attributes:
        job_id: Internal batch job identifier (e.g., "batch_abc123")
        vertex_job_name: Full Vertex AI resource name
            (e.g., "projects/.../locations/.../batchJobs/...")
        model: Model identifier used for the batch (e.g., "gemini-2.5-flash")
        region: GCP region where the job was submitted (e.g., "us-central1")
        submitted_at: ISO timestamp when job was submitted
        input_uri: GCS URI of the input JSONL file
        output_uri: GCS URI of the output directory
        request_count: Number of requests in the batch
        requests: Full BatchRequest objects for self-contained result mapping
    """

    job_id: str = Field(..., description="Internal batch job ID (e.g., 'batch_abc123')")
    vertex_job_name: str = Field(
        ...,
        description="Full Vertex AI resource name (projects/.../batchJobs/...)",
    )
    model: str = Field(..., description="Model identifier (e.g., 'gemini-2.5-flash')")
    region: str | None = Field(
        default=None,
        description="GCP region where the job was submitted (e.g., 'us-central1')",
    )
    submitted_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="ISO timestamp when job was submitted",
    )
    input_uri: str = Field(..., description="GCS URI of input JSONL file")
    output_uri: str = Field(..., description="GCS URI of output directory")
    request_count: int = Field(..., description="Number of requests in the batch")
    requests: list[BatchRequest] = Field(
        ...,
        description="Full BatchRequest objects for self-contained result mapping",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "batch_abc123def456",
                "vertex_job_name": "projects/my-project/locations/us-central1/batchJobs/12345",
                "model": "gemini-2.5-flash",
                "region": "us-central1",
                "submitted_at": "2026-01-23T07:00:00+00:00",
                "input_uri": "gs://bucket/session/batch/batch_abc123def456/input.jsonl",
                "output_uri": "gs://bucket/session/batch/batch_abc123def456/output/",
                "request_count": 500,
                "requests": [
                    {
                        "custom_id": "uuid-1",
                        "record_id": "rec-001",
                        "messages": [{"role": "user", "content": "Record content here..."}],
                    }
                ],
            }
        }
    )

    def get_region(self) -> str:
        """Get the region for this job.

        Returns the explicit region if set, otherwise parses it from vertex_job_name.
        Falls back to 'us-central1' if parsing fails.

        Returns:
            GCP region string (e.g., 'us-central1')
        """
        if self.region:
            return self.region

        # Parse from vertex_job_name: projects/{project}/locations/{location}/...
        try:
            parts = self.vertex_job_name.split("/")
            if "locations" in parts:
                loc_idx = parts.index("locations") + 1
                if loc_idx < len(parts):
                    return parts[loc_idx]
        except Exception:
            pass

        return "us-central1"  # Default fallback

    @classmethod
    def get_manifest_uri(cls, save_dir: str, job_id: str) -> str:
        """Get the standard manifest URI for a job.

        Args:
            save_dir: Session save directory (e.g., "gs://bucket/session")
            job_id: Batch job ID

        Returns:
            GCS URI for the manifest file
        """
        return f"{save_dir}/batch/{job_id}/manifest.json"


class BatchJobManager(BaseModel):
    """Manage Vertex AI batch prediction jobs.

    Handles JSONL generation, GCS upload (via bm.save), job submission,
    polling, and result retrieval.

    Uses buttermilk's session save_dir for GCS operations.

    Attributes:
        client: Google GenAI client (from bm.genai)
        poll_interval: Seconds between job status checks (default: 30)
        max_wait_hours: Maximum hours to wait for job completion (default: 24)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(..., description="Google GenAI client instance")
    poll_interval: int = Field(default=30, description="Seconds between status checks")
    max_wait_hours: int = Field(default=24, description="Max wait time in hours")

    # Track active jobs
    _active_jobs: dict[str, Any] = PrivateAttr(default_factory=dict)

    def _save_text(self, content: str, uri: str, content_type: str = "text/plain") -> str:
        """Save text content to GCS or local path.

        Args:
            content: Text content to save
            uri: Target URI (gs:// or local)
            content_type: MIME type

        Returns:
            The saved URI
        """
        if uri.startswith("gs://"):
            return upload_text(content, uri=uri, content_type=content_type)

        from cloudpathlib import AnyPath

        path = AnyPath(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)

    def _save_json(self, data: Any, uri: str) -> str:
        """Save JSON content to GCS or local path.

        Args:
            data: Data to serialize to JSON
            uri: Target URI (gs:// or local)

        Returns:
            The saved URI
        """
        if uri.startswith("gs://"):
            return upload_json(data, uri=uri)

        # Local save
        content = json.dumps(scrub_serializable(data), indent=2)
        return self._save_text(content, uri, content_type="application/json")

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return f"batch_{uuid.uuid4().hex[:12]}"

    def _build_gemini_request(
        self,
        request: BatchRequest,
    ) -> dict[str, Any]:
        """Build a Gemini batch request entry.

        Args:
            request: The batch request to format (with messages in LiteLLM format)

        Returns:
            JSONL-ready dictionary for Gemini batch API
        """
        return GeminiMessageConverter().build_request(request)

    def _build_claude_request(
        self,
        request: BatchRequest,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Build a Claude batch request entry.

        Args:
            request: The batch request to format (with messages in LiteLLM format)
            max_tokens: Maximum tokens for response

        Returns:
            JSONL-ready dictionary for Claude batch API
        """
        return ClaudeMessageConverter(max_tokens=max_tokens).build_request(request)

    def build_jsonl(
        self,
        requests: list[BatchRequest],
        model: str,
        max_tokens: int | None = None,
    ) -> str:
        """Build JSONL content for batch job.

        Args:
            requests: List of batch requests (each with messages in LiteLLM format)
            model: Model identifier (determines format)
            max_tokens: Maximum tokens for response (passed to Claude converter)

        Returns:
            JSONL string ready for upload
        """
        converter = get_message_converter(model, max_tokens=max_tokens)
        lines = [json.dumps(converter.build_request(request)) for request in requests]
        return "\n".join(lines)

    def _resolve_batch_dir(self, job_id: str) -> str:
        """Resolve the stable storage directory for a batch job.

        Pattern: <save_dir_base>/<project>/_batches/<job_id>/
        Fallback: <save_dir>/batch/<job_id>/ (if save_dir_base not configured)

        Args:
            job_id: The batch job identifier

        Returns:
            GCS URI or path for the batch job directory (without trailing slash)
        """
        from cloudpathlib import AnyPath

        from buttermilk import bm

        session = bm.session_info

        # Prefer stable path based on save_dir_base
        if session.save_dir_base:
            base = AnyPath(session.save_dir_base)
            # Use project name to organize batches within the bucket
            return str(base / session.project_name / "_batches" / job_id)

        # Fallback to session-scoped path
        if session.save_dir:
            return f"{session.save_dir}/batch/{job_id}"

        raise RuntimeError("No save_dir_base or save_dir configured in session")

    def _upload_to_gcs(self, content: str, job_id: str) -> str:
        """Upload JSONL content to GCS using bm.save.

        Args:
            content: JSONL string content to upload
            job_id: Job identifier for organizing files

        Returns:
            GCS URI of uploaded file
        """
        from buttermilk.utils.save import upload_text

        # Get stable batch directory
        batch_dir = self._resolve_batch_dir(job_id)

        # Construct URI
        uri = f"{batch_dir}/input.jsonl"

        # Use existing upload utility
        result_uri = self._save_text(content, uri=uri, content_type="application/jsonl")

        logger.info(f"Uploaded batch input to {result_uri}")
        return result_uri

    def _get_output_uri(self, job_id: str) -> str:
        """Get the output URI for a batch job.

        Args:
            job_id: Job identifier

        Returns:
            GCS URI for output directory
        """
        batch_dir = self._resolve_batch_dir(job_id)
        return f"{batch_dir}/output/"

    def _download_from_gcs(self, uri: str) -> str:
        """Download content from GCS.

        Args:
            uri: GCS URI to download

        Returns:
            File content as string
        """
        from cloudpathlib import AnyPath

        path = AnyPath(uri)
        return path.read_text()

    def _resolve_model_alias(self, model: str) -> str:
        """Resolve a short model alias to its full model name.

        Looks up the model in buttermilk's LLM registry. If found, returns
        the full model name from configs["model"]. Otherwise returns the
        input unchanged.

        Args:
            model: Model name - can be a short alias (e.g., "gemini-flash")
                   or a full name (e.g., "gemini-3-flash-preview")

        Returns:
            Full model name from registry, or original if not found
        """
        try:
            from buttermilk import bm

            if model in bm.llms.connections:
                config = bm.llms.connections[model]
                full_model = config.configs.get("model")
                if full_model:
                    logger.debug(f"Resolved model alias '{model}' to '{full_model}'")
                    return full_model
        except Exception as e:
            logger.debug(f"Could not resolve model alias '{model}': {e}")

        return model

    def _get_vertex_model_path(self, model: str) -> str:
        """Convert model name to Vertex AI model path.

        Resolves short model aliases (e.g., "gemini-flash") to full model names
        (e.g., "gemini-3-flash-preview") using the buttermilk model registry.

        Args:
            model: Model name - can be a short alias or full name

        Returns:
            Full Vertex AI model path suitable for the Batch API
        """
        # First, resolve short alias to full model name if it exists in the registry
        resolved_model = self._resolve_model_alias(model)

        if "claude" in resolved_model.lower() or "anthropic" in resolved_model.lower():
            # Claude models use publisher path
            claude_map = {
                "claude-sonnet-4": "publishers/anthropic/models/claude-sonnet-4",
                "claude-opus-4": "publishers/anthropic/models/claude-opus-4",
                "claude-haiku": "publishers/anthropic/models/claude-3-5-haiku",
            }
            return claude_map.get(resolved_model, f"publishers/anthropic/models/{resolved_model}")
        else:
            # Gemini models - strip google/ prefix if present (Batch API expects bare names)
            if resolved_model.startswith("google/"):
                return resolved_model[len("google/") :]
            return resolved_model

    async def submit_batch(
        self,
        model: str,
        requests: list[BatchRequest],
    ) -> "BatchJob":
        """Submit a batch prediction job.

        Args:
            model: Model to use (e.g., "gemini-2.5-flash")
            requests: List of batch requests (each with messages in LiteLLM format)

        Returns:
            BatchJob object for tracking

        Raises:
            RuntimeError: If job submission fails
        """
        from google.genai.types import CreateBatchJobConfig

        job_id = self._generate_job_id()

        # Build JSONL
        jsonl_content = self.build_jsonl(requests, model)

        # Upload to stable batch directory
        input_uri = self._upload_to_gcs(jsonl_content, job_id)
        output_uri = self._get_output_uri(job_id)

        # Get model path
        model_path = self._get_vertex_model_path(model)

        logger.info(
            f"Submitting batch job {job_id} with {len(requests)} requests",
            extra={"model": model_path, "input_uri": input_uri},
        )

        try:
            job = self.client.batches.create(
                model=model_path,
                src=input_uri,
                config=CreateBatchJobConfig(dest=output_uri),
            )

            self._active_jobs[job_id] = {
                "job": job,
                "requests": requests,
                "model": model,
                "output_uri": output_uri,
            }

            # Save manifest for job recovery
            self._save_manifest(
                job_id=job_id,
                vertex_job_name=job.name,
                model=model,
                input_uri=input_uri,
                output_uri=output_uri,
                requests=requests,
            )

            logger.info(f"Batch job submitted: {job.name} (ID: {job_id})")
            return job

        except Exception as e:
            raise RuntimeError(f"Failed to submit batch job: {e}") from e

    async def wait_for_completion(self, job: "BatchJob") -> "BatchJob":
        """Wait for a batch job to complete.

        Args:
            job: The batch job to wait for

        Returns:
            Updated BatchJob with final status

        Raises:
            TimeoutError: If job doesn't complete within max_wait_hours
            RuntimeError: If job fails
        """
        from google.genai.types import JobState

        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        max_iterations = (self.max_wait_hours * 3600) // self.poll_interval
        iteration = 0

        while iteration < max_iterations:
            await asyncio.sleep(self.poll_interval)
            iteration += 1

            try:
                job = self.client.batches.get(name=job.name)
            except Exception as e:
                logger.warning(f"Error polling job status: {e}")
                continue

            logger.debug(f"Job {job.name} status: {job.state}")

            if job.state in completed_states:
                if job.state == JobState.JOB_STATE_SUCCEEDED:
                    logger.info(f"Batch job {job.name} completed successfully")
                    return job
                elif job.state == JobState.JOB_STATE_FAILED:
                    raise RuntimeError(f"Batch job {job.name} failed")
                elif job.state == JobState.JOB_STATE_CANCELLED:
                    raise RuntimeError(f"Batch job {job.name} was cancelled")
                else:
                    raise RuntimeError(f"Batch job {job.name} in unexpected state: {job.state}")

        raise TimeoutError(f"Batch job {job.name} did not complete within {self.max_wait_hours} hours")

    def parse_results(
        self,
        output_uri: str,
        requests: list[BatchRequest],
    ) -> list[BatchResult]:
        """Parse batch job results from GCS.

        Uses cloudpathlib for GCS access (consistent with buttermilk patterns).

        Args:
            output_uri: GCS URI of output directory or file
            requests: Original requests for mapping custom_id -> request info

        Returns:
            List of BatchResult objects
        """
        from cloudpathlib import AnyPath

        # Build lookup from custom_id to request info
        request_map = {r.custom_id: r for r in requests}

        results = []

        try:
            output_path = AnyPath(output_uri)

            # Find all JSONL files in output directory (recursive - Vertex AI nests in subdirectories)
            if output_path.is_dir():
                jsonl_files = list(output_path.glob("**/*.jsonl"))
            else:
                jsonl_files = [output_path]

            for jsonl_file in jsonl_files:
                content = jsonl_file.read_text()
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        custom_id = entry.get("custom_id", "")
                        request = request_map.get(custom_id)

                        # Extract error from status field (Vertex AI format) or error field
                        error_msg = None
                        status_str = entry.get("status")
                        if status_str:
                            try:
                                status = json.loads(status_str)
                                if status.get("code") != 0:  # Non-zero = error
                                    error_msg = status.get("message")
                            except json.JSONDecodeError:
                                error_msg = status_str  # Use raw string if not JSON
                        if not error_msg:
                            error_msg = entry.get("error", {}).get("message")

                        result = BatchResult(
                            custom_id=custom_id,
                            record_id=request.record_id if request else "",
                            response=self._extract_response(entry),
                            error=error_msg,
                            usage=entry.get("response", {}).get("usage"),
                            # Propagate metadata from request for unique identification
                            model=request.model if request else None,
                            variant=request.variant if request else None,
                            processor_index=request.processor_index if request else None,
                        )
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse result line: {e}")

        except Exception as e:
            logger.error(f"Failed to parse batch results from {output_uri}: {e}")
            raise

        logger.info(f"Parsed {len(results)} results from batch output")
        return results

    def _extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response text from batch result entry.

        Handles both Gemini and Claude response formats by trying each converter.

        Args:
            entry: The result entry dictionary

        Returns:
            Response text or None
        """
        # Try Gemini format first (most common)
        result = GeminiMessageConverter().extract_response(entry)
        if result is not None:
            return result

        # Try Claude format
        result = ClaudeMessageConverter().extract_response(entry)
        if result is not None:
            return result

        # Fallback: try direct text field
        return entry.get("response", {}).get("text")

    async def run_batch_and_wait(
        self,
        model: str,
        requests: list[BatchRequest],
    ) -> list[BatchResult]:
        """Submit batch job, wait for completion, and return results.

        Convenience method that combines submit, wait, and parse.

        Args:
            model: Model to use
            requests: List of batch requests (each with messages in LiteLLM format)

        Returns:
            List of BatchResult objects
        """
        job = await self.submit_batch(model, requests)
        await self.wait_for_completion(job)

        # Get output URI from active jobs registry
        output_uri = self._get_output_uri_for_job(job.name)

        return self.parse_results(output_uri, requests)

    def _get_output_uri_for_job(self, job_name: str) -> str:
        """Get output URI for a job from the active jobs registry.

        Args:
            job_name: Full resource name like "projects/.../batchJobs/..."

        Returns:
            Output URI for the job

        Raises:
            RuntimeError: If job not found in registry
        """
        for job_id, info in self._active_jobs.items():
            if info["job"].name == job_name:
                return info["output_uri"]
        raise RuntimeError(f"Job {job_name} not found in active jobs registry")

    def _get_client_region(self) -> str | None:
        """Extract the region from the current client configuration.

        Returns:
            Region string if available, None otherwise
        """
        try:
            # The genai Client stores location in _api_client or similar
            if hasattr(self.client, "_location"):
                return self.client._location
            if hasattr(self.client, "location"):
                return self.client.location
            # Try to get from vertexai config
            if hasattr(self.client, "_api_client") and hasattr(self.client._api_client, "_location"):
                return self.client._api_client._location
        except Exception:
            pass
        return None

    def _parse_region_from_job_name(self, vertex_job_name: str) -> str | None:
        """Parse region from a Vertex AI resource name.

        Args:
            vertex_job_name: Full resource name like projects/.../locations/{region}/...

        Returns:
            Region string if found, None otherwise
        """
        try:
            parts = vertex_job_name.split("/")
            if "locations" in parts:
                loc_idx = parts.index("locations") + 1
                if loc_idx < len(parts):
                    return parts[loc_idx]
        except Exception:
            pass
        return None

    def _get_client_for_region(self, region: str) -> Any:
        """Get a GenAI client configured for a specific region.

        If the current client is already configured for this region, returns it.
        Otherwise creates a new client for the specified region.

        Args:
            region: GCP region (e.g., 'us-central1', 'us-east1')

        Returns:
            GenAI client configured for the specified region
        """
        current_region = self._get_client_region()
        if current_region == region:
            return self.client

        # Need to create a new client for the target region
        try:
            import os

            from google.genai import Client

            # Get project from environment or buttermilk config
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                try:
                    from buttermilk import bm
                    project = bm.session_info.project_name
                except Exception:
                    pass

            if not project:
                logger.warning(f"Could not determine project for region-specific client, using default client")
                return self.client

            logger.info(f"Creating client for region: {region} (current: {current_region})")
            return Client(vertexai=True, project=project, location=region)
        except Exception as e:
            logger.warning(f"Failed to create region-specific client for {region}: {e}. Using default.")
            return self.client

    def _save_manifest(
        self,
        job_id: str,
        vertex_job_name: str,
        model: str,
        input_uri: str,
        output_uri: str,
        requests: list[BatchRequest],
        region: str | None = None,
    ) -> str:
        """Save job manifest to GCS for recovery.

        Args:
            job_id: Local job identifier
            vertex_job_name: Full Vertex AI resource name
            model: Model identifier
            input_uri: GCS URI of input JSONL
            output_uri: GCS URI of output directory
            requests: List of batch requests
            region: Explicit region override (for dry runs where vertex_job_name isn't parseable)

        Returns:
            GCS URI of saved manifest
        """
        from buttermilk.utils.save import upload_text

        # Manifests always live at the root of the batch directory
        batch_dir = self._resolve_batch_dir(job_id)
        manifest_uri = f"{batch_dir}/manifest.json"

        # Use explicit region if provided, otherwise extract from job name or client config
        if region is None:
            region = self._parse_region_from_job_name(vertex_job_name) or self._get_client_region()

        manifest = BatchJobManifest(
            job_id=job_id,
            vertex_job_name=vertex_job_name,
            model=model,
            region=region,
            input_uri=input_uri,
            output_uri=output_uri,
            request_count=len(requests),
            requests=requests,
        )

        self._save_text(
            manifest.model_dump_json(indent=2),
            uri=manifest_uri,
            content_type="application/json",
        )

        logger.info(f"Saved manifest to {manifest_uri}")
        return manifest_uri

    def _load_manifest(self, job_id: str, search: bool = False) -> BatchJobManifest:
        """Load job manifest from GCS.

        Uses a multi-strategy approach to locate the manifest:
        1. Check stable persistent path at {save_dir_base}/{project}/_batches/{job_id}/
        2. Check current session path at {save_dir}/batch/{job_id}/
        3. (If search=True) Deep search across all runs in the bucket

        Args:
            job_id: Local job identifier
            search: If True, perform deep search across bucket if manifest not found
                   in standard locations. This can be slow for large buckets.

        Returns:
            BatchJobManifest object

        Raises:
            FileNotFoundError: If manifest not found
        """
        from cloudpathlib import AnyPath

        from buttermilk import bm

        session = bm.session_info
        manifest_path = None

        # Strategy 1: Check stable persistent path (O(1) lookup)
        if session.save_dir_base:
            try:
                base = AnyPath(session.save_dir_base)
                stable_path = base / session.project_name / "_batches" / job_id / "manifest.json"
                if stable_path.exists():
                    manifest_path = stable_path
                    logger.debug(f"Found manifest at stable location: {manifest_path}")
            except Exception as e:
                logger.debug(f"Failed to check stable path: {e}")

        # Strategy 2: Check current session path (Legacy/Fallback)
        if not manifest_path and session.save_dir:
            try:
                session_path = AnyPath(f"{session.save_dir}/batch/{job_id}/manifest.json")
                if session_path.exists():
                    manifest_path = session_path
                    logger.debug(f"Found manifest in current session: {manifest_path}")
            except Exception:
                pass

        # Strategy 3: Deep search across runs (if enabled)
        if not manifest_path and search and session.save_dir:
            try:
                save_dir = session.save_dir
                parts = save_dir.split("/")
                # Extract bucket from gs://bucket/... path
                if len(parts) > 2 and parts[0] == "gs:":
                    bucket = parts[2]
                    runs_root = f"gs://{bucket}/runs"
                    logger.info(f"Searching for manifest in {runs_root}...")
                    runs_path = AnyPath(runs_root)
                    found_manifests = list(runs_path.glob(f"**/batch/{job_id}/manifest.json"))
                    if found_manifests:
                        manifest_path = found_manifests[0]
                        logger.info(f"Found manifest via search: {manifest_path}")
            except Exception as e:
                logger.warning(f"Deep search failed: {e}")

        if not manifest_path:
            if search:
                raise FileNotFoundError(
                    f"Manifest not found for job_id: {job_id} (searched all locations)"
                )
            else:
                raise FileNotFoundError(
                    f"Manifest not found for job_id: {job_id}. "
                    f"Try using --search to search across all sessions, or "
                    f"--save-dir to specify the original session directory."
                )

        content = manifest_path.read_text()
        return BatchJobManifest.model_validate_json(content)

    def get_job_status(self, job_id: str, search: bool = False) -> dict[str, Any]:
        """Get the current status of a batch job.

        Args:
            job_id: Local job identifier (e.g., "batch_abc123")
            search: If True, search across all sessions in the bucket

        Returns:
            Dictionary with job status information:
            - job_id: Local job identifier
            - vertex_job_name: Full Vertex AI resource name
            - state: Job state string
            - is_complete: Whether job has finished
            - is_success: Whether job completed successfully
            - error: Error message if job failed

        Raises:
            FileNotFoundError: If manifest not found for job_id
        """
        from google.genai.types import JobState

        manifest = self._load_manifest(job_id, search=search)

        # Get the correct region for this job
        job_region = manifest.get_region()
        client = self._get_client_for_region(job_region)

        try:
            job = client.batches.get(name=manifest.vertex_job_name)
        except Exception as e:
            return {
                "job_id": job_id,
                "vertex_job_name": manifest.vertex_job_name,
                "region": job_region,
                "state": "UNKNOWN",
                "is_complete": False,
                "is_success": False,
                "error": f"Failed to fetch job status: {e}",
            }

        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        is_complete = job.state in completed_states
        is_success = job.state == JobState.JOB_STATE_SUCCEEDED

        error = None
        if job.state == JobState.JOB_STATE_FAILED:
            error = "Job failed"
        elif job.state == JobState.JOB_STATE_CANCELLED:
            error = "Job was cancelled"

        return {
            "job_id": job_id,
            "vertex_job_name": manifest.vertex_job_name,
            "state": str(job.state),
            "is_complete": is_complete,
            "is_success": is_success,
            "error": error,
        }

    def process_job_results(self, job_id: str, search: bool = False) -> dict[str, Any]:
        """Combine all results, join with manifest, calculate cost, and save to GCS.

        This method:
        1. Loads results from GCS (multiple files)
        2. Joins with original requests from manifest
        3. Calculates USD cost for each prediction
        4. Saves combined outputs to job directory on GCS
        5. Saves a full summary file with stats and total cost

        Args:
            job_id: Local job identifier
            search: If True, search across all sessions in the bucket

        Returns:
            Dictionary containing 'summary' and 'results' (list of dicts)
        """
        # 1. Load Manifest
        manifest = self._load_manifest(job_id, search=search)

        # 2. Parse Results
        results = self.parse_results(manifest.output_uri, manifest.requests)

        # 3. Join and Calculate Cost
        combined_data = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0

        for result in results:
            # Combine result with key manifest/request info
            entry = {
                "record_id": result.record_id,
                "custom_id": result.custom_id,
                "model": result.model or manifest.model,
                "variant": result.variant,
                "processor_index": result.processor_index,
                "response": result.response,
                "error": result.error,
                "usage": result.usage,
            }

            # Calculate cost
            if result.usage:
                p_tokens, c_tokens, cost = calculate_token_cost(
                    model=result.model or manifest.model,
                    usage_dict=result.usage,
                )
                result.cost_usd = cost
                entry["cost_usd"] = cost
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                total_cost_usd += cost

            combined_data.append(entry)

        # 4. Save Combined Results to GCS
        batch_dir = self._resolve_batch_dir(job_id)
        combined_uri = f"{batch_dir}/combined_results.jsonl"
        self._save_json(combined_data, uri=combined_uri)

        # 5. Create and Save Summary
        summary = {
            "job_id": job_id,
            "vertex_job_name": manifest.vertex_job_name,
            "model": manifest.model,
            "submitted_at": manifest.submitted_at,
            "processed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "request_count": manifest.request_count,
            "results_count": len(results),
            "success_count": sum(1 for r in results if not r.error),
            "error_count": sum(1 for r in results if r.error),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_cost_usd": total_cost_usd,
            "combined_results_uri": combined_uri,
        }

        summary_uri = f"{batch_dir}/summary.json"
        self._save_text(
            json.dumps(scrub_serializable(summary), indent=2),
            uri=summary_uri,
            content_type="application/json",
        )

        logger.info(
            f"Processed results for job {job_id}. Total cost: ${total_cost_usd:.4f}. Summary at {summary_uri}",
            job_id=job_id,
            total_cost=total_cost_usd,
            summary_uri=summary_uri,
        )

        return {
            "summary": summary,
            "results": combined_data,
            "batch_results": results,
        }

    def fetch_results(self, job_id: str, search: bool = False) -> dict[str, Any]:
        """Fetch and process batch results using only job_id.

        Loads the manifest from GCS, checks job status, and processes results
        if the job has completed successfully.

        Args:
            job_id: Local job identifier (e.g., "batch_abc123")
            search: If True, search across all sessions in the bucket

        Returns:
            If job succeeded: Summary and results dict (from process_job_results)
            If job still running: Dictionary with status info
            If job failed: Dictionary with error info

        Raises:
            FileNotFoundError: If manifest not found for job_id
        """
        from google.genai.types import JobState

        # Load manifest from GCS
        manifest = self._load_manifest(job_id, search=search)

        # Get the correct region for this job
        job_region = manifest.get_region()
        client = self._get_client_for_region(job_region)

        # Check job status
        try:
            job = client.batches.get(name=manifest.vertex_job_name)
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "error",
                "region": job_region,
                "error": f"Failed to fetch job status: {e}",
            }

        # Handle different job states
        if job.state == JobState.JOB_STATE_SUCCEEDED:
            logger.info(f"Job {job_id} completed, processing results")
            return self.process_job_results(job_id, search=search)

        elif job.state == JobState.JOB_STATE_FAILED:
            return {
                "job_id": job_id,
                "status": "failed",
                "state": str(job.state),
                "error": "Batch job failed",
            }

        elif job.state == JobState.JOB_STATE_CANCELLED:
            return {
                "job_id": job_id,
                "status": "cancelled",
                "state": str(job.state),
                "error": "Batch job was cancelled",
            }

        else:
            # Job still running
            return {
                "job_id": job_id,
                "status": "running",
                "state": str(job.state),
                "vertex_job_name": manifest.vertex_job_name,
                "request_count": manifest.request_count,
            }
