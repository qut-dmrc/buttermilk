"""Live integration test for OpenAI Batch API via Azure OpenAI.

Runs a real batch job against Azure OpenAI's Batch API endpoint.
No mocks — all API calls are live.

Requires:
- Azure OpenAI credentials in ~/.cache/buttermilk/models/models.json
- A deployment with GlobalBatch SKU (not GlobalStandard)

To create a GlobalBatch deployment in Azure Portal:
1. Go to your Azure OpenAI resource
2. Model deployments -> Deploy model
3. Choose a model (e.g., gpt-4o-mini)
4. Set Deployment Type to "Global-Batch"
5. Name it (e.g., "gpt-4o-mini-batch")

Then set AZURE_BATCH_DEPLOYMENT=gpt-4o-mini-batch before running.

Run with:
    AZURE_BATCH_DEPLOYMENT=gpt-4o-mini-batch uv run python -m pytest \
        tests/processors/test_openai_batch_live.py -o "addopts=" -m slow -v -s --timeout=600
"""

import io
import json
import os
from pathlib import Path

import pytest

from buttermilk.batch.managers.openai import OpenAIBatchJobManager
from buttermilk.batch.types import BatchRequest, BatchResult

pytestmark = [pytest.mark.slow, pytest.mark.anyio, pytest.mark.timeout(600)]

# Path to the cached model registry
MODELS_JSON = Path.home() / ".cache" / "buttermilk" / "models" / "models.json"

# Azure model keys to try for credentials (picks first azure model found)
AZURE_MODEL_KEYS = ["gpt-nano", "gpt-mini", "gpt-4o", "gpt-chat"]


def _find_azure_credentials():
    """Find Azure OpenAI credentials from models.json.

    Returns the first Azure model's api_key, base_url, and api_version.
    The actual deployment name for batch must come from AZURE_BATCH_DEPLOYMENT env var.
    """
    if not MODELS_JSON.exists():
        pytest.skip(f"models.json not found at {MODELS_JSON}")

    registry = json.loads(MODELS_JSON.read_text())

    for model_key in AZURE_MODEL_KEYS:
        entry = registry.get(model_key)
        if not entry or entry.get("client_type") != "azure":
            continue

        api_key = entry.get("api_key")
        base_url = entry.get("base_url")
        configs = entry.get("configs", {})
        api_version = configs.get("api_version", "2024-12-01-preview")

        if api_key and base_url:
            return {
                "api_key": api_key,
                "base_url": base_url,
                "api_version": api_version,
            }

    pytest.skip("No Azure OpenAI model found in models.json")


def _get_batch_deployment():
    """Get the batch deployment name from env or try auto-detection.

    Azure Batch API requires a deployment with GlobalBatch SKU.
    Standard deployments (GlobalStandard) won't work.
    """
    # Check explicit env var first
    deployment = os.environ.get("AZURE_BATCH_DEPLOYMENT")
    if deployment:
        return deployment

    # Try to auto-detect by attempting a batch create with known deployments
    # and looking for one that doesn't return 'invalid_deployment_type'
    return None


@pytest.fixture(scope="module")
def azure_batch_setup():
    """Provide an AzureOpenAI client and batch-capable deployment name.

    Skips if no batch-capable deployment is available.
    """
    from openai import AzureOpenAI

    creds = _find_azure_credentials()

    client = AzureOpenAI(
        api_key=creds["api_key"],
        azure_endpoint=creds["base_url"],
        api_version=creds["api_version"],
    )

    deployment = _get_batch_deployment()
    if not deployment:
        # Try to auto-detect by probing deployments
        deployment = _probe_batch_deployment(client)

    if not deployment:
        pytest.skip(
            "No GlobalBatch deployment found. Set AZURE_BATCH_DEPLOYMENT env var "
            "or create a GlobalBatch deployment in Azure Portal. "
            "See test docstring for instructions."
        )

    return client, deployment


def _probe_batch_deployment(client):
    """Try to find a batch-capable deployment by probing the API.

    Uploads a minimal JSONL and tries to create a batch with various
    deployment names. Returns the first one that works, or None.
    """
    # Common deployment names to try (batch deployments often have -batch suffix)
    candidates = [
        "gpt-4o-mini-batch",
        "gpt-4o-batch",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5-chat",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
    ]

    for candidate in candidates:
        jsonl = json.dumps(
            {
                "custom_id": "probe-001",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": candidate,
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_tokens": 5,
                },
            }
        )

        try:
            uploaded = client.files.create(
                file=("probe.jsonl", io.BytesIO(jsonl.encode())),
                purpose="batch",
            )

            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/chat/completions",
                completion_window="24h",
            )

            # If we get here, this deployment works for batch!
            # Cancel the probe batch immediately
            try:
                client.batches.cancel(batch.id)
            except Exception:
                pass

            return candidate

        except Exception as e:
            error_str = str(e)
            if "invalid_deployment_type" in error_str:
                continue  # Wrong SKU, try next
            if "model_not_found" in error_str:
                continue  # Deployment doesn't exist, try next
            # Some other error — log and continue
            continue

    return None


class TestOpenAIBatchLive:
    """Live integration tests for OpenAI Batch API via Azure."""

    async def test_batch_submit_poll_download(self, azure_batch_setup):
        """Submit a small batch, wait for completion, validate results.

        This is the core end-to-end proof: real file upload, real batch
        creation, real polling, real result download and parsing.
        """
        client, deployment = azure_batch_setup

        requests = [
            BatchRequest(
                custom_id="live-test-001",
                record_id="rec-001",
                messages=[
                    {"role": "system", "content": "Reply with exactly one word."},
                    {"role": "user", "content": "What color is the sky?"},
                ],
                model=deployment,
            ),
            BatchRequest(
                custom_id="live-test-002",
                record_id="rec-002",
                messages=[
                    {"role": "system", "content": "Reply with exactly one word."},
                    {"role": "user", "content": "What color is grass?"},
                ],
                model=deployment,
            ),
            BatchRequest(
                custom_id="live-test-003",
                record_id="rec-003",
                messages=[
                    {"role": "system", "content": "Reply with exactly one word."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                model=deployment,
            ),
        ]

        manager = OpenAIBatchJobManager(
            client=client,
            endpoint="/chat/completions",  # Azure uses no /v1/ prefix
            poll_interval=10,
            max_wait_hours=1,
        )

        # Step 1: Build JSONL and verify structure
        jsonl = manager.build_jsonl(requests, model=deployment)
        lines = jsonl.strip().split("\n")
        assert len(lines) == 3, f"Expected 3 JSONL lines, got {len(lines)}"

        for line in lines:
            entry = json.loads(line)
            assert entry["method"] == "POST"
            assert entry["url"] == "/v1/chat/completions"
            assert "body" in entry
            assert entry["body"]["model"] == deployment

        # Step 2: Submit batch (real API call)
        submit_result = await manager.submit_batch(
            model=deployment,
            requests=requests,
            metadata={"test": "live-integration"},
        )

        assert "openai_batch_id" in submit_result
        assert submit_result["request_count"] == 3
        openai_batch_id = submit_result["openai_batch_id"]
        print(f"\nSubmitted batch: {openai_batch_id}")

        # Step 3: Wait for completion (real polling)
        completed_batch = await manager.wait_for_completion(openai_batch_id)
        print(f"Batch completed with status: {completed_batch.status}")
        assert completed_batch.status == "completed"

        # Step 4: Download and parse results
        results = manager.download_results(openai_batch_id, requests)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert all(isinstance(r, BatchResult) for r in results)

        # Verify each result has a response
        results_by_id = {r.record_id: r for r in results}
        for record_id in ["rec-001", "rec-002", "rec-003"]:
            result = results_by_id[record_id]
            assert result.response is not None, f"No response for {record_id}"
            assert len(result.response.strip()) > 0, f"Empty response for {record_id}"
            assert result.error is None, f"Error for {record_id}: {result.error}"
            print(f"  {record_id}: {result.response.strip()}")

    async def test_batch_status_check(self, azure_batch_setup):
        """Submit a batch and check its status without waiting for completion."""
        client, deployment = azure_batch_setup

        requests = [
            BatchRequest(
                custom_id="status-test-001",
                record_id="status-rec-001",
                messages=[{"role": "user", "content": "Say hello."}],
                model=deployment,
            ),
        ]

        manager = OpenAIBatchJobManager(
            client=client,
            endpoint="/chat/completions",
            poll_interval=10,
            max_wait_hours=1,
        )

        submit_result = await manager.submit_batch(
            model=deployment,
            requests=requests,
        )

        openai_batch_id = submit_result["openai_batch_id"]

        # Immediately check status — should be validating or in_progress
        status = manager.get_batch_status(openai_batch_id)
        assert status["openai_batch_id"] == openai_batch_id
        assert status["status"] in {
            "validating",
            "in_progress",
            "finalizing",
            "completed",
        }
        assert status["total"] >= 0
        print(f"\nBatch {openai_batch_id} status: {status['status']}")

        # Wait for it to finish so we don't leave orphaned batches
        try:
            await manager.wait_for_completion(openai_batch_id)
        except (TimeoutError, RuntimeError):
            pass  # Best-effort cleanup
