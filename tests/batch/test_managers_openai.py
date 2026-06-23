"""Tests for OpenAIBatchJobManager.

Tests the OpenAI Batch API job manager: JSONL building, file upload,
batch creation, polling, result downloading, and manifest persistence.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from buttermilk.batch.managers import (
    BatchRequest,
    OpenAIBatchJobManager,
    OpenAIBatchManifest,
)

# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_SCHEMA = {
    "title": "JudgeReasons",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["compliant", "non_compliant"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
}


def _make_requests(n: int = 3, model: str = "gpt-4o") -> list[BatchRequest]:
    """Create sample batch requests."""
    return [
        BatchRequest(
            custom_id=f"req_{i}",
            record_id=f"rec_{i}",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Evaluate item {i}"},
            ],
            model=model,
        )
        for i in range(n)
    ]


def _make_openai_result_line(custom_id: str, content: str, usage: dict | None = None) -> str:
    """Create a single OpenAI batch result JSONL line."""
    entry = {
        "id": f"resp_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "id": f"chatcmpl_{custom_id}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage
                or {
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "total_tokens": 70,
                },
            },
        },
        "error": None,
    }
    return json.dumps(entry)


def _make_openai_error_line(custom_id: str, error_msg: str) -> str:
    """Create a single OpenAI batch error JSONL line."""
    entry = {
        "id": f"resp_{custom_id}",
        "custom_id": custom_id,
        "response": None,
        "error": {"message": error_msg},
    }
    return json.dumps(entry)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock()


@pytest.fixture
def manager(mock_openai_client):
    """Create an OpenAIBatchJobManager with mock client."""
    return OpenAIBatchJobManager(client=mock_openai_client)


# =============================================================================
# JSONL Building Tests
# =============================================================================


class TestOpenAIBatchJobManagerBuildJsonl:
    """Tests for JSONL building."""

    def test_build_jsonl_produces_valid_openai_format(self, manager):
        """JSONL should contain OpenAI batch format entries."""
        requests = _make_requests(2)
        jsonl = manager.build_jsonl(requests, model="gpt-4o")

        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        entry = json.loads(lines[0])
        assert entry["custom_id"] == "req_0"
        assert entry["method"] == "POST"
        assert entry["url"] == "/v1/chat/completions"
        assert entry["body"]["model"] == "gpt-4o"
        assert len(entry["body"]["messages"]) == 2

    def test_build_jsonl_with_max_tokens(self, manager):
        """max_tokens should be included in the body."""
        requests = _make_requests(1)
        jsonl = manager.build_jsonl(requests, model="gpt-4o", max_tokens=4096)

        entry = json.loads(jsonl.strip())
        assert entry["body"]["max_tokens"] == 4096

    def test_build_jsonl_without_max_tokens(self, manager):
        """Without max_tokens, body should not include it."""
        requests = _make_requests(1)
        jsonl = manager.build_jsonl(requests, model="gpt-4o")

        entry = json.loads(jsonl.strip())
        assert "max_tokens" not in entry["body"]

    def test_build_jsonl_with_schema(self, manager):
        """Structured output schema should produce response_format."""
        requests = [
            BatchRequest(
                custom_id="req_0",
                record_id="rec_0",
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4o",
                response_schema=SAMPLE_SCHEMA,
            )
        ]
        jsonl = manager.build_jsonl(requests, model="gpt-4o")

        entry = json.loads(jsonl.strip())
        assert "response_format" in entry["body"]
        assert entry["body"]["response_format"]["type"] == "json_schema"
        assert entry["body"]["response_format"]["json_schema"]["strict"] is True

    def test_build_jsonl_model_fallback(self, manager):
        """When request.model is None, converter model should be used."""
        requests = [
            BatchRequest(
                custom_id="req_0",
                record_id="rec_0",
                messages=[{"role": "user", "content": "Test"}],
                model=None,
            )
        ]
        jsonl = manager.build_jsonl(requests, model="gpt-4o-mini")

        entry = json.loads(jsonl.strip())
        assert entry["body"]["model"] == "gpt-4o-mini"


# =============================================================================
# Submit Batch Tests
# =============================================================================


class TestOpenAIBatchJobManagerSubmit:
    """Tests for batch submission."""

    @pytest.mark.anyio
    async def test_submit_batch_uploads_file_and_creates_batch(self, manager, mock_openai_client):
        """submit_batch should upload file and create batch."""
        # Mock file upload
        mock_file = MagicMock()
        mock_file.id = "file-abc123"
        mock_openai_client.files.create.return_value = mock_file

        # Mock batch creation
        mock_batch = MagicMock()
        mock_batch.id = "batch_xyz789"
        mock_batch.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch

        requests = _make_requests(2)

        with patch.object(manager, "_resolve_batch_dir", return_value="/tmp/test_batch"), patch.object(manager, "_save_text"):
            result = await manager.submit_batch(
                model="gpt-4o",
                requests=requests,
            )

        assert result["job_id"].startswith("oai_batch_")
        assert result["openai_batch_id"] == "batch_xyz789"
        assert result["input_file_id"] == "file-abc123"
        assert result["status"] == "validating"
        assert result["model"] == "gpt-4o"
        assert result["request_count"] == 2

        # Verify file was uploaded
        mock_openai_client.files.create.assert_called_once()
        call_kwargs = mock_openai_client.files.create.call_args
        assert call_kwargs.kwargs["purpose"] == "batch"

        # Verify batch was created
        mock_openai_client.batches.create.assert_called_once_with(
            input_file_id="file-abc123",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=None,
        )

    @pytest.mark.anyio
    async def test_submit_batch_with_metadata(self, manager, mock_openai_client):
        """submit_batch should pass metadata to OpenAI."""
        mock_file = MagicMock()
        mock_file.id = "file-abc123"
        mock_openai_client.files.create.return_value = mock_file

        mock_batch = MagicMock()
        mock_batch.id = "batch_xyz789"
        mock_batch.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch

        requests = _make_requests(1)
        metadata = {"description": "nightly eval", "project": "osb"}

        with patch.object(manager, "_resolve_batch_dir", return_value="/tmp/test"), patch.object(manager, "_save_text"):
            await manager.submit_batch(
                model="gpt-4o",
                requests=requests,
                metadata=metadata,
            )

        mock_openai_client.batches.create.assert_called_once_with(
            input_file_id="file-abc123",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )

    @pytest.mark.anyio
    async def test_submit_batch_raises_on_failure(self, manager, mock_openai_client):
        """submit_batch should raise RuntimeError on failure."""
        mock_openai_client.files.create.side_effect = Exception("Upload failed")

        requests = _make_requests(1)

        with pytest.raises(RuntimeError, match="Failed to submit OpenAI batch job"):
            await manager.submit_batch(model="gpt-4o", requests=requests)


# =============================================================================
# Batch Status Tests
# =============================================================================


class TestOpenAIBatchJobManagerStatus:
    """Tests for batch status checking."""

    def test_get_batch_status_running(self, manager, mock_openai_client):
        """Status should report running batch with progress."""
        mock_batch = MagicMock()
        mock_batch.status = "in_progress"
        mock_batch.request_counts = SimpleNamespace(completed=5, failed=0, total=10)
        mock_batch.output_file_id = None
        mock_batch.error_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch

        status = manager.get_batch_status("batch_xyz789")

        assert status["status"] == "in_progress"
        assert status["is_complete"] is False
        assert status["is_success"] is False
        assert status["completed"] == 5
        assert status["total"] == 10

    def test_get_batch_status_completed(self, manager, mock_openai_client):
        """Status should report completed batch."""
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.request_counts = SimpleNamespace(completed=10, failed=0, total=10)
        mock_batch.output_file_id = "file-output123"
        mock_batch.error_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch

        status = manager.get_batch_status("batch_xyz789")

        assert status["status"] == "completed"
        assert status["is_complete"] is True
        assert status["is_success"] is True
        assert status["output_file_id"] == "file-output123"

    def test_get_batch_status_failed(self, manager, mock_openai_client):
        """Status should report failed batch."""
        mock_batch = MagicMock()
        mock_batch.status = "failed"
        mock_batch.request_counts = SimpleNamespace(completed=0, failed=0, total=10)
        mock_batch.output_file_id = None
        mock_batch.error_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch

        status = manager.get_batch_status("batch_xyz789")

        assert status["is_complete"] is True
        assert status["is_success"] is False

    def test_get_batch_status_api_error(self, manager, mock_openai_client):
        """Status should handle API errors gracefully."""
        mock_openai_client.batches.retrieve.side_effect = Exception("Network error")

        status = manager.get_batch_status("batch_xyz789")

        assert status["status"] == "error"
        assert "Network error" in status["error"]


# =============================================================================
# Download Results Tests
# =============================================================================


class TestOpenAIBatchJobManagerDownload:
    """Tests for result downloading."""

    def test_download_results_success(self, manager, mock_openai_client):
        """download_results should parse output JSONL correctly."""
        # Mock batch status
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output123"
        mock_batch.error_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch

        # Mock file content
        output_lines = "\n".join(
            [
                _make_openai_result_line("req_0", "Response 0"),
                _make_openai_result_line("req_1", "Response 1"),
                _make_openai_result_line("req_2", "Response 2"),
            ]
        )
        mock_content = MagicMock()
        mock_content.content = output_lines.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_content

        requests = _make_requests(3)
        results = manager.download_results("batch_xyz789", requests)

        assert len(results) == 3
        assert results[0].response == "Response 0"
        assert results[0].record_id == "rec_0"
        assert results[0].custom_id == "req_0"
        assert results[0].usage is not None
        assert results[0].usage["prompt_tokens"] == 50

    def test_download_results_with_errors(self, manager, mock_openai_client):
        """download_results should handle error entries."""
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output123"
        mock_batch.error_file_id = "file-error123"
        mock_openai_client.batches.retrieve.return_value = mock_batch

        # Output file has 2 successes
        output_lines = "\n".join(
            [
                _make_openai_result_line("req_0", "Response 0"),
                _make_openai_result_line("req_1", "Response 1"),
            ]
        )
        mock_output_content = MagicMock()
        mock_output_content.content = output_lines.encode("utf-8")

        # Error file has 1 error
        error_lines = _make_openai_error_line("req_2", "Content policy violation")
        mock_error_content = MagicMock()
        mock_error_content.content = error_lines.encode("utf-8")

        mock_openai_client.files.content.side_effect = [
            mock_output_content,
            mock_error_content,
        ]

        requests = _make_requests(3)
        results = manager.download_results("batch_xyz789", requests)

        assert len(results) == 3
        success_results = [r for r in results if not r.error]
        error_results = [r for r in results if r.error]
        assert len(success_results) == 2
        assert len(error_results) == 1
        assert "Content policy violation" in error_results[0].error

    def test_download_results_not_completed_raises(self, manager, mock_openai_client):
        """download_results should raise if batch is not completed."""
        mock_batch = MagicMock()
        mock_batch.status = "in_progress"
        mock_openai_client.batches.retrieve.return_value = mock_batch

        requests = _make_requests(1)

        with pytest.raises(RuntimeError, match="batch status is 'in_progress'"):
            manager.download_results("batch_xyz789", requests)

    def test_download_results_no_output_file_raises(self, manager, mock_openai_client):
        """download_results should raise if no output_file_id."""
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch

        requests = _make_requests(1)

        with pytest.raises(RuntimeError, match="no output_file_id"):
            manager.download_results("batch_xyz789", requests)

    def test_download_results_http_error_in_response(self, manager, mock_openai_client):
        """download_results should capture HTTP error status in response."""
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output123"
        mock_batch.error_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch

        # Response with 400 status
        error_entry = {
            "id": "resp_req_0",
            "custom_id": "req_0",
            "response": {
                "status_code": 400,
                "body": {"error": {"message": "Invalid request parameters"}},
            },
            "error": None,
        }
        mock_content = MagicMock()
        mock_content.content = json.dumps(error_entry).encode("utf-8")
        mock_openai_client.files.content.return_value = mock_content

        requests = _make_requests(1)
        results = manager.download_results("batch_xyz789", requests)

        assert len(results) == 1
        assert results[0].error == "Invalid request parameters"
        assert results[0].response is None


# =============================================================================
# Wait for Completion Tests
# =============================================================================


class TestOpenAIBatchJobManagerWait:
    """Tests for wait_for_completion."""

    @pytest.mark.anyio
    async def test_wait_for_completion_returns_on_completed(self, manager, mock_openai_client):
        """wait_for_completion should return when batch completes."""
        mock_batch_running = MagicMock()
        mock_batch_running.status = "in_progress"
        mock_batch_running.request_counts = SimpleNamespace(completed=5, failed=0, total=10)

        mock_batch_done = MagicMock()
        mock_batch_done.status = "completed"
        mock_batch_done.request_counts = SimpleNamespace(completed=10, failed=0, total=10)

        mock_openai_client.batches.retrieve.side_effect = [
            mock_batch_running,
            mock_batch_done,
        ]

        manager.poll_interval = 1

        with patch("asyncio.sleep", return_value=None):
            result = await manager.wait_for_completion("batch_xyz789")
        assert result.status == "completed"

    @pytest.mark.anyio
    async def test_wait_for_completion_raises_on_failure(self, manager, mock_openai_client):
        """wait_for_completion should raise on failed batch."""
        mock_batch = MagicMock()
        mock_batch.status = "failed"
        mock_batch.request_counts = SimpleNamespace(completed=0, failed=10, total=10)
        mock_openai_client.batches.retrieve.return_value = mock_batch

        manager.poll_interval = 1

        with patch("asyncio.sleep", return_value=None), pytest.raises(RuntimeError, match="failed"):
            await manager.wait_for_completion("batch_xyz789")

    @pytest.mark.anyio
    async def test_wait_for_completion_raises_on_expired(self, manager, mock_openai_client):
        """wait_for_completion should raise on expired batch."""
        mock_batch = MagicMock()
        mock_batch.status = "expired"
        mock_batch.request_counts = SimpleNamespace(completed=5, failed=0, total=10)
        mock_openai_client.batches.retrieve.return_value = mock_batch

        manager.poll_interval = 1

        with patch("asyncio.sleep", return_value=None), pytest.raises(RuntimeError, match="expired"):
            await manager.wait_for_completion("batch_xyz789")


# =============================================================================
# Manifest Tests
# =============================================================================


class TestOpenAIBatchManifest:
    """Tests for OpenAIBatchManifest model."""

    def test_manifest_serialization(self):
        """Manifest should serialize and deserialize cleanly."""
        requests = _make_requests(2)
        manifest = OpenAIBatchManifest(
            job_id="oai_batch_abc123",
            openai_batch_id="batch_xyz789",
            model="gpt-4o",
            input_file_id="file-input123",
            request_count=2,
            requests=requests,
            metadata={"description": "test run"},
        )

        json_str = manifest.model_dump_json()
        loaded = OpenAIBatchManifest.model_validate_json(json_str)

        assert loaded.job_id == "oai_batch_abc123"
        assert loaded.openai_batch_id == "batch_xyz789"
        assert loaded.model == "gpt-4o"
        assert loaded.input_file_id == "file-input123"
        assert loaded.request_count == 2
        assert len(loaded.requests) == 2
        assert loaded.requests[0].custom_id == "req_0"
        assert loaded.metadata == {"description": "test run"}

    def test_manifest_submitted_at_auto_set(self):
        """submitted_at should be auto-set to current time."""
        manifest = OpenAIBatchManifest(
            job_id="oai_batch_abc123",
            openai_batch_id="batch_xyz789",
            model="gpt-4o",
            input_file_id="file-input123",
            request_count=0,
            requests=[],
        )

        assert manifest.submitted_at is not None
        # Should be a valid ISO timestamp
        import datetime

        datetime.datetime.fromisoformat(manifest.submitted_at)


# =============================================================================
# Integration Test: run_batch_and_wait
# =============================================================================


class TestOpenAIBatchJobManagerIntegration:
    """Integration tests combining submit, wait, and download."""

    @pytest.mark.anyio
    async def test_run_batch_and_wait(self, manager, mock_openai_client):
        """run_batch_and_wait should combine submit, wait, and download."""
        # Mock file upload
        mock_file = MagicMock()
        mock_file.id = "file-abc123"
        mock_openai_client.files.create.return_value = mock_file

        # Mock batch creation
        mock_batch_created = MagicMock()
        mock_batch_created.id = "batch_xyz789"
        mock_batch_created.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch_created

        # Mock polling - returns completed immediately
        mock_batch_done = MagicMock()
        mock_batch_done.status = "completed"
        mock_batch_done.request_counts = SimpleNamespace(completed=2, failed=0, total=2)
        mock_batch_done.output_file_id = "file-output123"
        mock_batch_done.error_file_id = None
        mock_openai_client.batches.retrieve.return_value = mock_batch_done

        # Mock result download
        output_lines = "\n".join(
            [
                _make_openai_result_line("req_0", "Answer 0"),
                _make_openai_result_line("req_1", "Answer 1"),
            ]
        )
        mock_content = MagicMock()
        mock_content.content = output_lines.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_content

        requests = _make_requests(2)
        manager.poll_interval = 1

        with patch("asyncio.sleep", return_value=None), patch.object(manager, "_resolve_batch_dir", return_value="/tmp/test"):
            with patch.object(manager, "_save_text"):
                results = await manager.run_batch_and_wait(
                    model="gpt-4o",
                    requests=requests,
                )

        assert len(results) == 2
        assert results[0].response == "Answer 0"
        assert results[1].response == "Answer 1"
        assert results[0].record_id == "rec_0"
        assert results[1].record_id == "rec_1"
