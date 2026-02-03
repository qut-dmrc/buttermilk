"""Tests for Vertex AI batch processor and caching components.

These tests focus on unit-testable components that don't require
actual GCP connections. Separate integration tests use live Vertex AI.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.vertex_batch import (
    BatchJobManager,
    BatchRequest,
    BatchResult,
)
from buttermilk._core.vertex_caching import CriteriaCacheManager


class TestBatchRequest:
    """Tests for BatchRequest model."""

    def test_create_batch_request(self):
        """Test creating a batch request."""
        request = BatchRequest(
            custom_id="record_001",
            record_id="record_001",
            messages=[{"role": "user", "content": "Test content"}],
        )

        assert request.custom_id == "record_001"
        assert request.record_id == "record_001"
        assert request.messages == [{"role": "user", "content": "Test content"}]

    def test_batch_request_with_system_message(self):
        """Test batch request with system and user messages."""
        request = BatchRequest(
            custom_id="test",
            record_id="rec1",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[1]["role"] == "user"


class TestBatchResult:
    """Tests for BatchResult model."""

    def test_create_success_result(self):
        """Test creating a successful batch result."""
        result = BatchResult(
            custom_id="record_001",
            record_id="record_001",
            response="This is the LLM response",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        assert result.response == "This is the LLM response"
        assert result.error is None
        assert result.usage["input_tokens"] == 100

    def test_create_error_result(self):
        """Test creating an error batch result."""
        result = BatchResult(
            custom_id="record_001",
            record_id="record_001",
            error="Rate limit exceeded",
        )

        assert result.response is None
        assert result.error == "Rate limit exceeded"


class TestBatchJobManager:
    """Tests for BatchJobManager JSONL building."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock genai client."""
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_client):
        """Create a BatchJobManager with mock client."""
        return BatchJobManager(client=mock_client)

    def test_build_gemini_request(self, manager):
        """Test building a Gemini batch request entry."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this content"}],
        )

        entry = manager._build_gemini_request(request)

        assert entry["custom_id"] == "test_001"
        assert "request" in entry
        assert entry["request"]["contents"][0]["role"] == "user"
        assert entry["request"]["contents"][0]["parts"][0]["text"] == "Evaluate this content"

    def test_build_gemini_request_with_system(self, manager):
        """Test Gemini request with system instruction."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
        )

        entry = manager._build_gemini_request(request)

        assert "system_instruction" in entry["request"]
        assert entry["request"]["system_instruction"]["parts"][0]["text"] == "You are a helpful assistant"
        assert entry["request"]["contents"][0]["role"] == "user"

    def test_build_claude_request(self, manager):
        """Test building a Claude batch request entry."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Record content here"}],
        )

        entry = manager._build_claude_request(request, max_tokens=2048)

        assert entry["custom_id"] == "test_001"
        assert entry["request"]["anthropic_version"] == "vertex-2023-10-16"
        assert entry["request"]["max_tokens"] == 2048

        # Check message structure
        messages = entry["request"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Record content here"

    def test_build_claude_request_with_system(self, manager):
        """Test building a Claude batch request with system message."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
        )

        entry = manager._build_claude_request(request)

        assert entry["request"]["system"] == "You are a helpful assistant"
        assert entry["request"]["messages"][0]["role"] == "user"
        assert entry["request"]["messages"][0]["content"] == "Hello"

    def test_build_jsonl_gemini(self, manager):
        """Test building complete JSONL for Gemini."""
        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Content 1"}],
            ),
            BatchRequest(
                custom_id="rec2",
                record_id="rec2",
                messages=[{"role": "user", "content": "Content 2"}],
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gemini-2.5-flash")

        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["custom_id"] == "rec1"

        entry2 = json.loads(lines[1])
        assert entry2["custom_id"] == "rec2"

    def test_build_jsonl_claude(self, manager):
        """Test building complete JSONL for Claude."""
        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Content 1"}],
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="claude-sonnet-4")

        lines = jsonl.strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert "anthropic_version" in entry["request"]

    def test_get_vertex_model_path_gemini(self, manager):
        """Test model path for Gemini models."""
        assert manager._get_vertex_model_path("gemini-2.5-flash") == "gemini-2.5-flash"
        assert manager._get_vertex_model_path("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_get_vertex_model_path_claude(self, manager):
        """Test model path for Claude models."""
        assert manager._get_vertex_model_path("claude-sonnet-4") == "publishers/anthropic/models/claude-sonnet-4"
        assert manager._get_vertex_model_path("claude-opus-4") == "publishers/anthropic/models/claude-opus-4"

    def test_extract_response_gemini_format(self, manager):
        """Test extracting response from Gemini batch result."""
        entry = {"response": {"candidates": [{"content": {"parts": [{"text": "The response text"}]}}]}}

        response = manager._extract_response(entry)
        assert response == "The response text"

    def test_extract_response_claude_format(self, manager):
        """Test extracting response from Claude batch result."""
        entry = {"response": {"content": [{"type": "text", "text": "Claude response here"}]}}

        response = manager._extract_response(entry)
        assert response == "Claude response here"

    def test_generate_job_id(self, manager):
        """Test job ID generation."""
        job_id1 = manager._generate_job_id()
        job_id2 = manager._generate_job_id()

        assert job_id1.startswith("batch_")
        assert job_id2.startswith("batch_")
        assert job_id1 != job_id2
        assert len(job_id1) == 18  # "batch_" + 12 hex chars


class TestCriteriaCacheManager:
    """Tests for CriteriaCacheManager."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock genai client."""
        client = MagicMock()
        client.caches = MagicMock()
        return client

    @pytest.fixture
    def cache_manager(self, mock_client):
        """Create a CriteriaCacheManager with mock client."""
        return CriteriaCacheManager(
            client=mock_client,
            ttl="3600s",
            model="gemini-2.5-flash",
        )

    def test_compute_cache_key_deterministic(self, cache_manager):
        """Test that cache keys are deterministic."""
        key1 = cache_manager._compute_cache_key(
            "gemini-2.5-flash",
            "You are a judge",
            "Evaluate for hate speech",
        )

        key2 = cache_manager._compute_cache_key(
            "gemini-2.5-flash",
            "You are a judge",
            "Evaluate for hate speech",
        )

        assert key1 == key2
        assert len(key1) == 16  # Truncated SHA256

    def test_compute_cache_key_different_content(self, cache_manager):
        """Test that different content produces different keys."""
        key1 = cache_manager._compute_cache_key(
            "gemini-2.5-flash",
            "System A",
            "Criteria A",
        )

        key2 = cache_manager._compute_cache_key(
            "gemini-2.5-flash",
            "System A",
            "Criteria B",  # Different
        )

        assert key1 != key2

    def test_compute_cache_key_different_model(self, cache_manager):
        """Test that different models produce different keys."""
        key1 = cache_manager._compute_cache_key(
            "gemini-2.5-flash",
            "System",
            "Criteria",
        )

        key2 = cache_manager._compute_cache_key(
            "gemini-2.5-pro",  # Different model
            "System",
            "Criteria",
        )

        assert key1 != key2

    def test_get_or_create_cache_returns_cached(self, cache_manager, mock_client):
        """Test that second call returns cached value."""
        # Setup mock
        mock_cache = MagicMock()
        mock_cache.name = "projects/123/cachedContents/abc"
        mock_client.caches.create.return_value = mock_cache

        # First call creates cache
        name1 = cache_manager.get_or_create_cache(
            criteria_content="Test criteria",
            display_name="test_cache",
        )

        # Second call should return from local registry
        name2 = cache_manager.get_or_create_cache(
            criteria_content="Test criteria",
            display_name="test_cache",
        )

        assert name1 == name2
        # Should only call create once
        assert mock_client.caches.create.call_count == 1

    def test_list_caches(self, cache_manager, mock_client):
        """Test listing caches."""
        mock_cache1 = MagicMock()
        mock_cache1.name = "cache1"
        mock_cache1.display_name = "Test 1"
        mock_cache1.model = "gemini-2.5-flash"
        mock_cache1.expire_time = None

        mock_client.caches.list.return_value = [mock_cache1]

        caches = cache_manager.list_caches()

        assert len(caches) == 1
        assert caches[0]["name"] == "cache1"
        assert caches[0]["display_name"] == "Test 1"
