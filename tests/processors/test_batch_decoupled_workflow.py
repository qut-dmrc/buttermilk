"""Test batch job decoupled workflow: submit → restart → fetch.

This test validates that batch jobs survive process restarts by using
only the job_id to recover job state and fetch results.

Testing Philosophy (per TESTING.md):
- Mock ONLY at system boundaries (Vertex AI API, GCS upload)
- Let all BatchJobManager internal logic run for real
- Use local filesystem (tmp_path) for manifest/result storage
- Use real BatchJobManifest serialization/deserialization
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.vertex_batch import (
    BatchJobManager,
    BatchJobManifest,
    BatchRequest,
    BatchResult,
)


class FakeJobState:
    """Fake JobState enum for mocking Vertex AI responses."""

    JOB_STATE_SUCCEEDED = "JOB_STATE_SUCCEEDED"
    JOB_STATE_RUNNING = "JOB_STATE_RUNNING"
    JOB_STATE_FAILED = "JOB_STATE_FAILED"
    JOB_STATE_CANCELLED = "JOB_STATE_CANCELLED"
    JOB_STATE_PAUSED = "JOB_STATE_PAUSED"


def create_fake_batch_job(name: str, state: str) -> MagicMock:
    """Create a fake BatchJob object for testing."""
    job = MagicMock()
    job.name = name
    job.state = state
    return job


def create_fake_output_jsonl(requests: list[BatchRequest]) -> str:
    """Create fake Vertex AI batch output JSONL content."""
    lines = []
    for req in requests:
        entry = {
            "custom_id": req.custom_id,
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": f"Response for {req.record_id}"}]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


class TestBatchDecoupledWorkflow:
    """Test the submit → restart → fetch workflow for batch jobs."""

    @pytest.fixture
    def batch_requests(self) -> list[BatchRequest]:
        """Create sample batch requests for testing."""
        return [
            BatchRequest(
                custom_id="req-001",
                record_id="record-001",
                messages=[{"role": "user", "content": "Test content 1"}],
            ),
            BatchRequest(
                custom_id="req-002",
                record_id="record-002",
                messages=[{"role": "user", "content": "Test content 2"}],
            ),
            BatchRequest(
                custom_id="req-003",
                record_id="record-003",
                messages=[{"role": "user", "content": "Test content 3"}],
            ),
        ]

    @pytest.fixture
    def mock_vertex_client(self) -> MagicMock:
        """Create a mock Vertex AI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def local_save_dir(self, tmp_path: Path) -> str:
        """Create a local save directory that mimics GCS structure."""
        save_dir = tmp_path / "test-session"
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir)

    def _mock_upload_text(self, local_save_dir: str):
        """Create a mock upload_text that writes to local filesystem."""

        def upload_text_local(data: str, *, uri: str, **kwargs: Any) -> str:
            """Write to local filesystem instead of GCS."""
            # Convert gs:// URI to local path
            if uri.startswith("gs://"):
                # Strip gs://bucket/ prefix and use local dir
                local_path = Path(local_save_dir) / "/".join(uri.split("/")[3:])
            else:
                local_path = Path(uri)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(data)
            return str(local_path)

        return upload_text_local

    @pytest.mark.anyio
    async def test_submit_restart_fetch_workflow(
        self,
        real_bm,
        batch_requests: list[BatchRequest],
        mock_vertex_client: MagicMock,
        local_save_dir: str,
    ):
        """Test complete workflow: submit job, simulate restart, fetch results.

        This test verifies that:
        1. Job can be submitted and manifest is saved
        2. After "restart" (new manager instance), job status can be checked
        3. Results can be fetched using only job_id
        """
        # Configure real_bm to use local save_dir
        real_bm.session_info.save_dir = local_save_dir

        # Create vertex job name that will be returned
        vertex_job_name = "projects/test-project/locations/us-central1/batchJobs/12345"

        # Configure mock client for job submission
        mock_job = create_fake_batch_job(vertex_job_name, FakeJobState.JOB_STATE_RUNNING)
        mock_vertex_client.batches.create.return_value = mock_job

        # PHASE 1: Submit batch job
        # Mock only external boundaries: upload_text (GCS) and client.batches (Vertex AI)
        with patch(
            "buttermilk.utils.save.upload_text",
            side_effect=self._mock_upload_text(local_save_dir),
        ):
            manager1 = BatchJobManager(client=mock_vertex_client)

            # Submit the batch - this uses real BatchJobManager logic
            job = await manager1.submit_batch(
                model="gemini-2.5-flash",
                requests=batch_requests,
            )

            # Verify job was submitted
            assert job.name == vertex_job_name
            mock_vertex_client.batches.create.assert_called_once()

        # Extract job_id from the manager's internal state
        job_id = list(manager1._active_jobs.keys())[0]
        assert job_id.startswith("batch_")

        # Verify manifest was saved locally
        manifest_path = Path(local_save_dir) / "batch" / job_id / "manifest.json"
        assert manifest_path.exists(), f"Manifest should exist at {manifest_path}"

        # Read and validate manifest structure
        manifest_content = manifest_path.read_text()
        manifest_data = json.loads(manifest_content)
        assert manifest_data["job_id"] == job_id
        assert manifest_data["vertex_job_name"] == vertex_job_name
        assert manifest_data["request_count"] == 3
        assert len(manifest_data["requests"]) == 3

        # PHASE 2: Simulate process restart - create NEW manager
        # This manager has NO in-memory state from the previous submission
        manager2 = BatchJobManager(client=mock_vertex_client)

        # Configure mock for status check - job still running
        mock_vertex_client.batches.get.return_value = create_fake_batch_job(
            vertex_job_name, FakeJobState.JOB_STATE_RUNNING
        )

        # Mock JobState import for get_job_status
        with patch("google.genai.types.JobState", FakeJobState):
            # Check job status using only job_id (loads manifest from disk)
            status = manager2.get_job_status(job_id)

            assert status["job_id"] == job_id
            assert status["state"] == FakeJobState.JOB_STATE_RUNNING
            assert status["is_complete"] is False

        # PHASE 3: Job completes, fetch results
        # Configure mock for completed job
        mock_vertex_client.batches.get.return_value = create_fake_batch_job(
            vertex_job_name, FakeJobState.JOB_STATE_SUCCEEDED
        )

        # Create fake output file in the expected location
        output_dir = Path(local_save_dir) / "batch" / job_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "results.jsonl"
        output_file.write_text(create_fake_output_jsonl(batch_requests))

        # Mock JobState for fetch_results
        with patch("google.genai.types.JobState", FakeJobState):
            # Fetch results using only job_id
            results = manager2.fetch_results(job_id)

            # Should return list of BatchResult, not status dict
            assert isinstance(results, list), f"Expected list of results, got {type(results)}"
            assert len(results) == 3

            # Verify results are properly mapped back to original requests
            results_by_record = {r.record_id: r for r in results}
            assert "record-001" in results_by_record
            assert "record-002" in results_by_record
            assert "record-003" in results_by_record

            # Verify response content was extracted
            for result in results:
                assert result.response is not None
                assert "Response for" in result.response

    @pytest.mark.anyio
    async def test_fetch_results_job_still_running(
        self,
        real_bm,
        batch_requests: list[BatchRequest],
        mock_vertex_client: MagicMock,
        local_save_dir: str,
    ):
        """Test that fetch_results returns status info when job is still running."""
        real_bm.session_info.save_dir = local_save_dir

        vertex_job_name = "projects/test-project/locations/us-central1/batchJobs/running-job"
        mock_job = create_fake_batch_job(vertex_job_name, FakeJobState.JOB_STATE_RUNNING)
        mock_vertex_client.batches.create.return_value = mock_job

        # Submit job
        with patch(
            "buttermilk.utils.save.upload_text",
            side_effect=self._mock_upload_text(local_save_dir),
        ):
            manager = BatchJobManager(client=mock_vertex_client)
            await manager.submit_batch(
                model="gemini-2.5-flash",
                requests=batch_requests,
            )

        job_id = list(manager._active_jobs.keys())[0]

        # Simulate restart with new manager
        manager2 = BatchJobManager(client=mock_vertex_client)

        # Job still running
        mock_vertex_client.batches.get.return_value = create_fake_batch_job(
            vertex_job_name, FakeJobState.JOB_STATE_RUNNING
        )

        with patch("google.genai.types.JobState", FakeJobState):
            result = manager2.fetch_results(job_id)

            # Should return status dict, not results list
            assert isinstance(result, dict)
            assert result["status"] == "running"
            assert result["job_id"] == job_id

    @pytest.mark.anyio
    async def test_fetch_results_job_failed(
        self,
        real_bm,
        batch_requests: list[BatchRequest],
        mock_vertex_client: MagicMock,
        local_save_dir: str,
    ):
        """Test that fetch_results returns error info when job failed."""
        real_bm.session_info.save_dir = local_save_dir

        vertex_job_name = "projects/test-project/locations/us-central1/batchJobs/failed-job"
        mock_job = create_fake_batch_job(vertex_job_name, FakeJobState.JOB_STATE_RUNNING)
        mock_vertex_client.batches.create.return_value = mock_job

        # Submit job
        with patch(
            "buttermilk.utils.save.upload_text",
            side_effect=self._mock_upload_text(local_save_dir),
        ):
            manager = BatchJobManager(client=mock_vertex_client)
            await manager.submit_batch(
                model="gemini-2.5-flash",
                requests=batch_requests,
            )

        job_id = list(manager._active_jobs.keys())[0]

        # Simulate restart with new manager
        manager2 = BatchJobManager(client=mock_vertex_client)

        # Job failed
        mock_vertex_client.batches.get.return_value = create_fake_batch_job(
            vertex_job_name, FakeJobState.JOB_STATE_FAILED
        )

        with patch("google.genai.types.JobState", FakeJobState):
            result = manager2.fetch_results(job_id)

            # Should return error info
            assert isinstance(result, dict)
            assert result["status"] == "failed"
            assert "error" in result

    def test_manifest_not_found_raises_error(
        self,
        real_bm,
        mock_vertex_client: MagicMock,
        local_save_dir: str,
    ):
        """Test that fetching non-existent job raises FileNotFoundError."""
        real_bm.session_info.save_dir = local_save_dir

        manager = BatchJobManager(client=mock_vertex_client)

        with pytest.raises(FileNotFoundError):
            manager.fetch_results("nonexistent_job_id")

    def test_manifest_serialization_roundtrip(self, batch_requests: list[BatchRequest]):
        """Test that BatchJobManifest serializes and deserializes correctly."""
        original = BatchJobManifest(
            job_id="batch_test123",
            vertex_job_name="projects/p/locations/l/batchJobs/123",
            model="gemini-2.5-flash",
            input_uri="gs://bucket/input.jsonl",
            output_uri="gs://bucket/output/",
            request_count=3,
            requests=batch_requests,
        )

        # Serialize to JSON
        json_str = original.model_dump_json(indent=2)

        # Deserialize back
        restored = BatchJobManifest.model_validate_json(json_str)

        # Verify all fields match
        assert restored.job_id == original.job_id
        assert restored.vertex_job_name == original.vertex_job_name
        assert restored.model == original.model
        assert restored.request_count == original.request_count
        assert len(restored.requests) == len(original.requests)

        # Verify nested BatchRequest objects
        for orig_req, rest_req in zip(original.requests, restored.requests):
            assert rest_req.custom_id == orig_req.custom_id
            assert rest_req.record_id == orig_req.record_id
            assert rest_req.messages == orig_req.messages
