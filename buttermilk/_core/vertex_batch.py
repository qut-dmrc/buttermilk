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
        cache_refs=cache_refs,
    )
    results = await manager.wait_for_results(job)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from google.genai.types import BatchJob

from buttermilk._core.log import logger


class BatchRequest(BaseModel):
    """A single request in a batch job.

    Attributes:
        custom_id: Unique identifier for mapping results back to input
        record_id: Original record ID from the source data
        criteria_key: Key identifying which criteria template was used
        content: The record content to send as user message
        cache_name: Optional cache resource name for criteria
    """

    custom_id: str
    record_id: str
    criteria_key: str
    content: str
    cache_name: str | None = None


class BatchResult(BaseModel):
    """Result from a batch job request.

    Attributes:
        custom_id: The custom_id from the request
        record_id: Original record ID
        criteria_key: Criteria key used
        response: The model's response content
        error: Error message if request failed
        usage: Token usage information
    """

    custom_id: str
    record_id: str
    criteria_key: str
    response: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None


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
        submitted_at: ISO timestamp when job was submitted
        input_uri: GCS URI of the input JSONL file
        output_uri: GCS URI of the output directory
        request_count: Number of requests in the batch
        requests: Full BatchRequest objects for self-contained result mapping
        criteria_contents: Rendered criteria templates keyed by criteria_key
            (for Claude inline caching or audit purposes)
    """

    job_id: str = Field(..., description="Internal batch job ID (e.g., 'batch_abc123')")
    vertex_job_name: str = Field(
        ...,
        description="Full Vertex AI resource name (projects/.../batchJobs/...)",
    )
    model: str = Field(..., description="Model identifier (e.g., 'gemini-2.5-flash')")
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
    criteria_contents: dict[str, tuple[str, str]] = Field(
        default_factory=dict,
        description="Rendered criteria templates keyed by criteria_key (system, user)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "batch_abc123def456",
                "vertex_job_name": "projects/my-project/locations/us-central1/batchJobs/12345",
                "model": "gemini-2.5-flash",
                "submitted_at": "2026-01-23T07:00:00+00:00",
                "input_uri": "gs://bucket/session/batch/batch_abc123def456/input.jsonl",
                "output_uri": "gs://bucket/session/batch/batch_abc123def456/output/",
                "request_count": 500,
                "requests": [
                    {
                        "custom_id": "uuid-1",
                        "record_id": "rec-001",
                        "criteria_key": "a1b2c3d4",
                        "content": "Record content here...",
                        "cache_name": None,
                    }
                ],
                "criteria_contents": {"a1b2c3d4": ("You are a helpful assistant", "Evaluate content for...")},
            }
        }
    )

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

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return f"batch_{uuid.uuid4().hex[:12]}"

    def _build_gemini_request(
        self,
        request: BatchRequest,
        criteria: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build a Gemini batch request entry.

        Args:
            request: The batch request to format
            criteria: Optional (system_instruction, user_content) tuple

        Returns:
            JSONL-ready dictionary for Gemini batch API
        """
        # Combine user-part of criteria with record content
        user_parts = []
        if criteria and criteria[1]:
            user_parts.append({"text": criteria[1]})
        
        user_parts.append({"text": request.content})

        entry: dict[str, Any] = {
            "custom_id": request.custom_id,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": user_parts,
                    }
                ],
            },
        }

        # Add system instruction if available
        if criteria and criteria[0]:
            entry["request"]["system_instruction"] = {
                "parts": [{"text": criteria[0]}]
            }

        # Add cache reference if available
        if request.cache_name:
            entry["request"]["cached_content"] = request.cache_name

        return entry

    def _build_claude_request(
        self,
        request: BatchRequest,
        criteria: tuple[str, str] | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Build a Claude batch request entry.

        For Claude, we use inline cache_control since batch API may not
        support cached_content references directly.

        Args:
            request: The batch request to format
            criteria: Optional (system_instruction, user_content) tuple
            max_tokens: Maximum tokens for response

        Returns:
            JSONL-ready dictionary for Claude batch API
        """
        messages_content = []

        system_instruction = None
        user_criteria = None
        if criteria:
            system_instruction = criteria[0]
            user_criteria = criteria[1]

        # Add user-part of criteria with cache_control if provided
        if user_criteria:
            messages_content.append(
                {
                    "type": "text",
                    "text": user_criteria,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        # Add record content
        messages_content.append(
            {
                "type": "text",
                "text": request.content,
            }
        )

        request_body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [
                {
                    "role": "user",
                    "content": messages_content,
                }
            ],
            "max_tokens": max_tokens,
        }

        # Add system instruction if present
        if system_instruction:
            request_body["system"] = system_instruction

        return {
            "custom_id": request.custom_id,
            "request": request_body,
        }

    def build_jsonl(
        self,
        requests: list[BatchRequest],
        model: str,
        criteria_contents: dict[str, tuple[str, str]] | None = None,
    ) -> str:
        """Build JSONL content for batch job.

        Args:
            requests: List of batch requests
            model: Model identifier (determines format)
            criteria_contents: Map of criteria_key -> (system, user) for inline/system usage

        Returns:
            JSONL string ready for upload
        """
        lines = []
        is_claude = "claude" in model.lower() or "anthropic" in model.lower()

        for request in requests:
            # Get criteria content tuple if available
            criteria = criteria_contents.get(request.criteria_key) if criteria_contents else None
            
            if is_claude:
                entry = self._build_claude_request(request, criteria)
            else:
                entry = self._build_gemini_request(request, criteria)

            lines.append(json.dumps(entry))

        return "\n".join(lines)

    def _upload_to_gcs(self, content: str, job_id: str) -> str:
        """Upload JSONL content to GCS using bm.save.

        Args:
            content: JSONL string content to upload
            job_id: Job identifier for organizing files

        Returns:
            GCS URI of uploaded file
        """
        from buttermilk import bm
        from buttermilk.utils.save import upload_text

        # Get save directory from session
        save_dir = bm.session_info.save_dir
        if not save_dir:
            raise RuntimeError("No save_dir configured in session")

        # Construct URI within session's save directory
        uri = f"{save_dir}/batch/{job_id}/input.jsonl"

        # Use existing upload utility
        result_uri = upload_text(content, uri=uri, content_type="application/jsonl")

        logger.info(f"Uploaded batch input to {result_uri}")
        return result_uri

    def _get_output_uri(self, job_id: str) -> str:
        """Get the output URI for a batch job.

        Args:
            job_id: Job identifier

        Returns:
            GCS URI for output directory
        """
        from buttermilk import bm

        save_dir = bm.session_info.save_dir
        if not save_dir:
            raise RuntimeError("No save_dir configured in session")

        return f"{save_dir}/batch/{job_id}/output/"

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
        criteria_contents: dict[str, str] | None = None,
    ) -> "BatchJob":
        """Submit a batch prediction job.

        Args:
            model: Model to use (e.g., "gemini-2.5-flash")
            requests: List of batch requests
            criteria_contents: For Claude, map of criteria_key -> content

        Returns:
            BatchJob object for tracking

        Raises:
            RuntimeError: If job submission fails
        """
        from google.genai.types import CreateBatchJobConfig

        job_id = self._generate_job_id()

        # Build JSONL
        jsonl_content = self.build_jsonl(requests, model, criteria_contents)

        # Upload to GCS using session's save_dir
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
                criteria_contents=criteria_contents or {},
            )

            logger.info(f"Batch job submitted: {job.name}")
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
            requests: Original requests for mapping custom_id -> record info

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
                            criteria_key=request.criteria_key if request else "",
                            response=self._extract_response(entry),
                            error=error_msg,
                            usage=entry.get("response", {}).get("usage"),
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

        Handles both Gemini and Claude response formats.

        Args:
            entry: The result entry dictionary

        Returns:
            Response text or None
        """
        response = entry.get("response", {})

        # Gemini format: response.candidates[0].content.parts[0].text
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text")

        # Claude format: response.content[0].text
        content = response.get("content", [])
        if content and isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    return block.get("text")

        # Fallback: try direct text field
        return response.get("text")

    async def run_batch_and_wait(
        self,
        model: str,
        requests: list[BatchRequest],
        criteria_contents: dict[str, str] | None = None,
    ) -> list[BatchResult]:
        """Submit batch job, wait for completion, and return results.

        Convenience method that combines submit, wait, and parse.

        Args:
            model: Model to use
            requests: List of batch requests
            criteria_contents: For Claude inline caching

        Returns:
            List of BatchResult objects
        """
        job = await self.submit_batch(model, requests, criteria_contents)
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

    def _save_manifest(
        self,
        job_id: str,
        vertex_job_name: str,
        model: str,
        input_uri: str,
        output_uri: str,
        requests: list[BatchRequest],
        criteria_contents: dict[str, tuple[str, str]],
    ) -> str:
        """Save job manifest to GCS for recovery.

        Args:
            job_id: Local job identifier
            vertex_job_name: Full Vertex AI resource name
            model: Model identifier
            input_uri: GCS URI of input JSONL
            output_uri: GCS URI of output directory
            requests: List of batch requests
            criteria_contents: Rendered criteria templates

        Returns:
            GCS URI of saved manifest
        """
        from buttermilk import bm
        from buttermilk.utils.save import upload_text

        save_dir = bm.session_info.save_dir
        if not save_dir:
            raise RuntimeError("No save_dir configured in session")

        manifest = BatchJobManifest(
            job_id=job_id,
            vertex_job_name=vertex_job_name,
            model=model,
            input_uri=input_uri,
            output_uri=output_uri,
            request_count=len(requests),
            requests=requests,
            criteria_contents=criteria_contents,
        )

        manifest_uri = BatchJobManifest.get_manifest_uri(save_dir, job_id)
        upload_text(
            manifest.model_dump_json(indent=2),
            uri=manifest_uri,
            content_type="application/json",
        )

        logger.info(f"Saved manifest to {manifest_uri}")
        return manifest_uri

    def _load_manifest(self, job_id: str) -> BatchJobManifest:
        """Load job manifest from GCS.

        Args:
            job_id: Local job identifier

        Returns:
            BatchJobManifest object

        Raises:
            FileNotFoundError: If manifest not found
        """
        from cloudpathlib import AnyPath

        from buttermilk import bm

        save_dir = bm.session_info.save_dir
        if not save_dir:
            raise RuntimeError("No save_dir configured in session")

        manifest_uri = BatchJobManifest.get_manifest_uri(save_dir, job_id)
        manifest_path = AnyPath(manifest_uri)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for job_id: {job_id}")

        content = manifest_path.read_text()
        return BatchJobManifest.model_validate_json(content)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get the current status of a batch job.

        Args:
            job_id: Local job identifier (e.g., "batch_abc123")

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

        manifest = self._load_manifest(job_id)

        try:
            job = self.client.batches.get(name=manifest.vertex_job_name)
        except Exception as e:
            return {
                "job_id": job_id,
                "vertex_job_name": manifest.vertex_job_name,
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

    def fetch_results(self, job_id: str) -> list[BatchResult] | dict[str, Any]:
        """Fetch batch results using only job_id.

        Loads the manifest from GCS, checks job status, and parses results
        if the job has completed successfully.

        Args:
            job_id: Local job identifier (e.g., "batch_abc123")

        Returns:
            If job succeeded: List of BatchResult objects
            If job still running: Dictionary with status info
            If job failed: Dictionary with error info

        Raises:
            FileNotFoundError: If manifest not found for job_id (invalid or
                from different session)
        """
        from google.genai.types import JobState

        # Load manifest from GCS
        manifest = self._load_manifest(job_id)

        # Check job status
        try:
            job = self.client.batches.get(name=manifest.vertex_job_name)
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "error",
                "error": f"Failed to fetch job status: {e}",
            }

        # Handle different job states
        if job.state == JobState.JOB_STATE_SUCCEEDED:
            logger.info(f"Job {job_id} completed, parsing results")
            return self.parse_results(manifest.output_uri, manifest.requests)

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
