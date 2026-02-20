"""OpenAI Batch API manager."""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk._core.log import logger
from buttermilk.batch.converters import OpenAIMessageConverter
from buttermilk.batch.manifests import OpenAIBatchManifest
from buttermilk.batch.types import BatchRequest, BatchResult
from buttermilk.utils.pricing import calculate_token_cost
from buttermilk.utils.save import scrub_serializable, upload_text


class OpenAIBatchJobManager(BaseModel):
    """Manage OpenAI Batch API jobs.

    Handles JSONL generation, file upload, batch creation, polling,
    and result retrieval via the OpenAI API. Works with both direct
    OpenAI and Azure OpenAI clients.

    Uses buttermilk's session save_dir for manifest persistence.

    Attributes:
        client: OpenAI or AzureOpenAI client instance
        endpoint: API endpoint for batch requests. Use "/v1/chat/completions"
            for direct OpenAI, "/chat/completions" for Azure OpenAI.
        poll_interval: Seconds between job status checks (default: 30)
        max_wait_hours: Maximum hours to wait for job completion (default: 24)
        jsonl_url: URL path to use in the JSONL 'url' field.
            Default: "/v1/chat/completions".
            For Azure, set this to the relative path (e.g. "/chat/completions").
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(..., description="OpenAI or AzureOpenAI client instance")
    endpoint: str = Field(
        default="/v1/chat/completions",
        description="Batch endpoint path. '/v1/chat/completions' for OpenAI, '/chat/completions' for Azure.",
    )
    poll_interval: int = Field(default=30, description="Seconds between status checks")
    max_wait_hours: int = Field(default=24, description="Max wait time in hours")
    jsonl_url: str = Field(
        default="/v1/chat/completions",
        description="URL path to use in the JSONL 'url' field.",
    )

    # Track active jobs
    _active_jobs: dict[str, Any] = PrivateAttr(default_factory=dict)

    def _generate_job_id(self) -> str:
        """Generate a unique internal job ID."""
        return f"oai_batch_{uuid.uuid4().hex[:12]}"

    def _resolve_batch_dir(self, job_id: str) -> str:
        """Resolve the stable storage directory for a batch job.

        Pattern: <save_dir_base>/<project>/_batches/<job_id>/
        Fallback: <save_dir>/batch/<job_id>/

        Args:
            job_id: The batch job identifier

        Returns:
            GCS URI or path for the batch job directory (without trailing slash)
        """
        from cloudpathlib import AnyPath

        from buttermilk import bm

        session = bm.session_info

        if session.save_dir_base:
            base = AnyPath(session.save_dir_base)
            return str(base / session.project_name / "_batches" / job_id)

        if session.save_dir:
            return f"{session.save_dir}/batch/{job_id}"

        raise RuntimeError("No save_dir_base or save_dir configured in session")

    def _save_text(self, content: str, uri: str, content_type: str = "text/plain") -> str:
        """Save text content to GCS or local path."""
        if uri.startswith("gs://"):
            return upload_text(content, uri=uri, content_type=content_type)

        from cloudpathlib import AnyPath

        path = AnyPath(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)

    def build_jsonl(
        self,
        requests: list[BatchRequest],
        model: str,
        max_tokens: int | None = None,
    ) -> str:
        """Build JSONL content for OpenAI batch job.

        Args:
            requests: List of batch requests (each with messages in LiteLLM format)
            model: Model identifier (e.g., "gpt-4o")
            max_tokens: Maximum tokens for response

        Returns:
            JSONL string ready for upload
        """
        converter = OpenAIMessageConverter(max_tokens=max_tokens, model=model, url=self.jsonl_url)
        lines = [json.dumps(converter.build_request(request)) for request in requests]
        return "\n".join(lines)

    async def submit_batch(
        self,
        model: str,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Submit a batch prediction job to OpenAI.

        Steps:
        1. Build JSONL from requests
        2. Upload JSONL file to OpenAI
        3. Create batch job
        4. Save manifest for recovery

        Args:
            model: Model to use (e.g., "gpt-4o")
            requests: List of batch requests
            metadata: Optional metadata dict (up to 16 key-value pairs)
            max_tokens: Maximum tokens for response

        Returns:
            OpenAI batch object dict with id, status, etc.

        Raises:
            RuntimeError: If job submission fails
        """
        job_id = self._generate_job_id()

        # Build JSONL
        jsonl_content = self.build_jsonl(requests, model, max_tokens=max_tokens)
        jsonl_bytes = jsonl_content.encode("utf-8")

        logger.info(
            f"Submitting OpenAI batch job {job_id} with {len(requests)} requests",
            extra={"model": model},
        )

        try:
            # Step 1: Upload JSONL file to OpenAI
            import io

            uploaded_file = self.client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes)),
                purpose="batch",
            )
            input_file_id = uploaded_file.id
            logger.info(f"Uploaded input file: {input_file_id}")

            # Step 2: Create batch job
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint=self.endpoint,
                completion_window="24h",
                metadata=metadata,
            )

            self._active_jobs[job_id] = {
                "batch": batch,
                "requests": requests,
                "model": model,
                "input_file_id": input_file_id,
            }

            # Step 3: Save manifest for recovery (non-fatal if no save_dir configured)
            try:
                self._save_manifest(
                    job_id=job_id,
                    openai_batch_id=batch.id,
                    model=model,
                    input_file_id=input_file_id,
                    requests=requests,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to save manifest (batch still submitted): {e}")

            # Step 4: Also save the JSONL to our storage for audit
            try:
                batch_dir = self._resolve_batch_dir(job_id)
                self._save_text(
                    jsonl_content,
                    uri=f"{batch_dir}/input.jsonl",
                    content_type="application/jsonl",
                )
            except Exception as e:
                logger.warning(f"Failed to save input JSONL to storage: {e}")

            logger.info(f"OpenAI batch job submitted: {batch.id} (internal ID: {job_id})")

            return {
                "job_id": job_id,
                "openai_batch_id": batch.id,
                "input_file_id": input_file_id,
                "status": batch.status,
                "model": model,
                "request_count": len(requests),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to submit OpenAI batch job: {e}") from e

    async def wait_for_completion(
        self,
        openai_batch_id: str,
    ) -> Any:
        """Wait for an OpenAI batch job to complete.

        Args:
            openai_batch_id: The OpenAI batch ID (e.g., "batch_...")

        Returns:
            Completed batch object

        Raises:
            TimeoutError: If job doesn't complete within max_wait_hours
            RuntimeError: If job fails or expires
        """
        terminal_states = {"completed", "failed", "expired", "cancelled"}

        effective_interval = max(self.poll_interval, 1)
        max_iterations = (self.max_wait_hours * 3600) // effective_interval
        iteration = 0

        while iteration < max_iterations:
            await asyncio.sleep(effective_interval)
            iteration += 1

            try:
                batch = self.client.batches.retrieve(openai_batch_id)
            except Exception as e:
                logger.warning(f"Error polling OpenAI batch status: {e}")
                continue

            completed = getattr(batch.request_counts, "completed", 0) or 0
            total = getattr(batch.request_counts, "total", 0) or 0
            logger.debug(f"Batch {openai_batch_id} status: {batch.status} ({completed}/{total})")

            if batch.status in terminal_states:
                if batch.status == "completed":
                    logger.info(f"OpenAI batch {openai_batch_id} completed successfully")
                    return batch
                elif batch.status == "failed":
                    raise RuntimeError(f"OpenAI batch {openai_batch_id} failed")
                elif batch.status == "expired":
                    raise RuntimeError(f"OpenAI batch {openai_batch_id} expired (did not complete within completion window)")
                elif batch.status == "cancelled":
                    raise RuntimeError(f"OpenAI batch {openai_batch_id} was cancelled")

        raise TimeoutError(f"OpenAI batch {openai_batch_id} did not complete within {self.max_wait_hours} hours")

    def get_batch_status(self, openai_batch_id: str) -> dict[str, Any]:
        """Get the current status of an OpenAI batch job.

        Args:
            openai_batch_id: The OpenAI batch ID

        Returns:
            Dictionary with batch status information
        """
        try:
            batch = self.client.batches.retrieve(openai_batch_id)

            completed = getattr(batch.request_counts, "completed", 0) or 0
            failed = getattr(batch.request_counts, "failed", 0) or 0
            total = getattr(batch.request_counts, "total", 0) or 0

            terminal_states = {"completed", "failed", "expired", "cancelled"}

            return {
                "openai_batch_id": openai_batch_id,
                "status": batch.status,
                "is_complete": batch.status in terminal_states,
                "is_success": batch.status == "completed",
                "completed": completed,
                "failed": failed,
                "total": total,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }

        except Exception as e:
            return {
                "openai_batch_id": openai_batch_id,
                "status": "error",
                "is_complete": False,
                "is_success": False,
                "error": f"Failed to fetch batch status: {e}",
            }

    def download_results(
        self,
        openai_batch_id: str,
        requests: list[BatchRequest],
    ) -> list[BatchResult]:
        """Download and parse results from a completed OpenAI batch.

        Args:
            openai_batch_id: The OpenAI batch ID
            requests: Original requests for mapping custom_id -> request info

        Returns:
            List of BatchResult objects

        Raises:
            RuntimeError: If batch is not completed or download fails
        """
        batch = self.client.batches.retrieve(openai_batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Cannot download results: batch status is '{batch.status}', expected 'completed'")

        if not batch.output_file_id:
            raise RuntimeError(f"Batch {openai_batch_id} completed but has no output_file_id")

        request_map = {r.custom_id: r for r in requests}
        converter = OpenAIMessageConverter()
        results = []

        # Download output file
        output_content = self.client.files.content(batch.output_file_id).content
        output_text = output_content.decode("utf-8") if isinstance(output_content, bytes) else output_content

        for line in output_text.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                custom_id = entry.get("custom_id", "")
                request = request_map.get(custom_id)

                # Extract error
                error_msg = None
                error_obj = entry.get("error")
                if error_obj:
                    if isinstance(error_obj, dict):
                        error_msg = error_obj.get("message", str(error_obj))
                    else:
                        error_msg = str(error_obj)

                # Check response status code
                response = entry.get("response", {})
                if isinstance(response, dict):
                    status_code = response.get("status_code")
                    if status_code and status_code != 200 and not error_msg:
                        body = response.get("body", {})
                        err = body.get("error", {})
                        error_msg = err.get("message", f"HTTP {status_code}")

                # Extract usage from response body
                usage = None
                if isinstance(response, dict):
                    body = response.get("body", {})
                    if isinstance(body, dict):
                        usage = body.get("usage")

                result = BatchResult(
                    custom_id=custom_id,
                    record_id=request.record_id if request else "",
                    response=converter.extract_response(entry),
                    error=error_msg,
                    usage=usage,
                    model=request.model if request else None,
                    variant=request.variant if request else None,
                    processor_index=request.processor_index if request else None,
                )
                results.append(result)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse OpenAI result line: {e}")

        # Also download and log errors if present
        if batch.error_file_id:
            try:
                error_content = self.client.files.content(batch.error_file_id).content
                error_text = error_content.decode("utf-8") if isinstance(error_content, bytes) else error_content
                for line in error_text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        err_entry = json.loads(line)
                        custom_id = err_entry.get("custom_id", "")
                        request = request_map.get(custom_id)

                        # Check if we already have a result for this custom_id
                        existing_ids = {r.custom_id for r in results}
                        if custom_id not in existing_ids:
                            error_obj = err_entry.get("error", {})
                            error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)

                            results.append(
                                BatchResult(
                                    custom_id=custom_id,
                                    record_id=request.record_id if request else "",
                                    response=None,
                                    error=error_msg,
                                    model=request.model if request else None,
                                    variant=request.variant if request else None,
                                    processor_index=request.processor_index if request else None,
                                )
                            )
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.warning(f"Failed to download error file: {e}")

        logger.info(f"Parsed {len(results)} results from OpenAI batch output")
        return results

    async def run_batch_and_wait(
        self,
        model: str,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> list[BatchResult]:
        """Submit batch job, wait for completion, and return results.

        Convenience method that combines submit, wait, and download.

        Args:
            model: Model to use (e.g., "gpt-4o")
            requests: List of batch requests
            metadata: Optional metadata for the batch
            max_tokens: Maximum tokens for response

        Returns:
            List of BatchResult objects
        """
        submit_result = await self.submit_batch(model, requests, metadata=metadata, max_tokens=max_tokens)
        openai_batch_id = submit_result["openai_batch_id"]

        await self.wait_for_completion(openai_batch_id)

        return self.download_results(openai_batch_id, requests)

    def _save_manifest(
        self,
        job_id: str,
        openai_batch_id: str,
        model: str,
        input_file_id: str,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Save job manifest for recovery.

        Args:
            job_id: Internal job identifier
            openai_batch_id: OpenAI batch ID
            model: Model identifier
            input_file_id: OpenAI file ID for input
            requests: List of batch requests
            metadata: Optional metadata

        Returns:
            URI of saved manifest
        """
        batch_dir = self._resolve_batch_dir(job_id)
        manifest_uri = f"{batch_dir}/manifest.json"

        manifest = OpenAIBatchManifest(
            job_id=job_id,
            openai_batch_id=openai_batch_id,
            model=model,
            input_file_id=input_file_id,
            request_count=len(requests),
            requests=requests,
            metadata=metadata,
        )

        self._save_text(
            manifest.model_dump_json(indent=2),
            uri=manifest_uri,
            content_type="application/json",
        )

        logger.info(f"Saved OpenAI batch manifest to {manifest_uri}")
        return manifest_uri

    def _load_manifest(self, job_id: str) -> OpenAIBatchManifest:
        """Load job manifest from storage.

        Args:
            job_id: Internal job identifier

        Returns:
            OpenAIBatchManifest object

        Raises:
            FileNotFoundError: If manifest not found
        """
        from cloudpathlib import AnyPath

        from buttermilk import bm

        session = bm.session_info
        manifest_path = None

        # Check stable persistent path
        if session.save_dir_base:
            try:
                base = AnyPath(session.save_dir_base)
                stable_path = base / session.project_name / "_batches" / job_id / "manifest.json"
                if stable_path.exists():
                    manifest_path = stable_path
            except Exception as e:
                logger.debug(f"Failed to check stable path: {e}")

        # Check current session path
        if not manifest_path and session.save_dir:
            try:
                session_path = AnyPath(f"{session.save_dir}/batch/{job_id}/manifest.json")
                if session_path.exists():
                    manifest_path = session_path
            except Exception:
                pass

        if not manifest_path:
            raise FileNotFoundError(f"OpenAI batch manifest not found for job_id: {job_id}")

        content = manifest_path.read_text()
        return OpenAIBatchManifest.model_validate_json(content)

    def fetch_results(self, job_id: str) -> dict[str, Any]:
        """Fetch and process batch results using internal job_id.

        Loads manifest, checks status, downloads results if complete.

        Args:
            job_id: Internal job identifier

        Returns:
            If completed: dict with summary, results, batch_results
            If running: dict with status info
            If failed: dict with error info
        """
        manifest = self._load_manifest(job_id)

        status_info = self.get_batch_status(manifest.openai_batch_id)

        if not status_info["is_complete"]:
            return {
                "job_id": job_id,
                "openai_batch_id": manifest.openai_batch_id,
                "status": status_info["status"],
                "completed": status_info.get("completed", 0),
                "total": status_info.get("total", 0),
                "request_count": manifest.request_count,
            }

        if not status_info["is_success"]:
            return {
                "job_id": job_id,
                "openai_batch_id": manifest.openai_batch_id,
                "status": status_info["status"],
                "error": status_info.get("error", f"Batch {status_info['status']}"),
            }

        # Download and process results
        results = self.download_results(manifest.openai_batch_id, manifest.requests)

        # Calculate costs and build combined data
        combined_data = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0

        for result in results:
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

        # Save combined results
        try:
            batch_dir = self._resolve_batch_dir(job_id)
            combined_uri = f"{batch_dir}/combined_results.jsonl"
            self._save_text(
                "\n".join(json.dumps(scrub_serializable(e)) for e in combined_data),
                uri=combined_uri,
                content_type="application/jsonl",
            )
        except Exception as e:
            logger.warning(f"Failed to save combined results: {e}")

        summary = {
            "job_id": job_id,
            "openai_batch_id": manifest.openai_batch_id,
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
        }

        logger.info(f"Processed OpenAI batch results for job {job_id}. Total cost: ")

        return {
            "summary": summary,
            "results": combined_data,
            "batch_results": results,
        }
