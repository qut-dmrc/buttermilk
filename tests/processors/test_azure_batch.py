"""Tests for Azure OpenAI Batch support."""

import json
from unittest.mock import MagicMock

import pytest
from buttermilk.batch.managers.openai import OpenAIBatchJobManager
from buttermilk.batch.types import BatchRequest

def test_azure_jsonl_url_configuration():
    """Test that jsonl_url is correctly used in JSONL generation."""
    mock_client = MagicMock()
    manager = OpenAIBatchJobManager(
        client=mock_client,
        jsonl_url="/chat/completions"
    )

    requests = [
        BatchRequest(
            custom_id="req_0",
            record_id="rec_0",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
        )
    ]

    jsonl = manager.build_jsonl(requests, model="gpt-4o")

    entry = json.loads(jsonl.strip())
    assert entry["url"] == "/chat/completions"
    assert entry["method"] == "POST"

def test_azure_default_url():
    """Test that default url is still /v1/chat/completions."""
    mock_client = MagicMock()
    manager = OpenAIBatchJobManager(client=mock_client)

    requests = [
        BatchRequest(
            custom_id="req_0",
            record_id="rec_0",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
        )
    ]

    jsonl = manager.build_jsonl(requests, model="gpt-4o")

    entry = json.loads(jsonl.strip())
    assert entry["url"] == "/v1/chat/completions"
