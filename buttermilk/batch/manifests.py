"""Batch job manifests."""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field

from buttermilk.batch.types import BatchRequest


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


class OpenAIBatchManifest(BaseModel):
    """Persistent manifest for OpenAI batch job recovery after process restart.

    Stored alongside the input JSONL in the batch directory.
    Contains all context needed to reconstruct job state and map results
    back to source records.

    Attributes:
        job_id: Internal batch job identifier (e.g., "batch_abc123")
        openai_batch_id: OpenAI batch object ID (e.g., "batch_...")
        model: Model identifier used for the batch (e.g., "gpt-4o")
        submitted_at: ISO timestamp when job was submitted
        input_file_id: OpenAI file ID for the uploaded input JSONL
        request_count: Number of requests in the batch
        requests: Full BatchRequest objects for self-contained result mapping
        metadata: Optional metadata passed to OpenAI batch create
    """

    job_id: str = Field(..., description="Internal batch job ID")
    openai_batch_id: str = Field(..., description="OpenAI batch object ID")
    model: str = Field(..., description="Model identifier (e.g., 'gpt-4o')")
    submitted_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="ISO timestamp when job was submitted",
    )
    input_file_id: str = Field(..., description="OpenAI file ID for input JSONL")
    request_count: int = Field(..., description="Number of requests in the batch")
    requests: list[BatchRequest] = Field(
        ...,
        description="Full BatchRequest objects for self-contained result mapping",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional metadata passed to OpenAI batch create",
    )
