"""Tests for Vertex AI batch processor and caching components.

These tests focus on unit-testable components that don't require
actual GCP connections. Integration tests would require live Vertex AI.
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
            custom_id="record_001_criteria_A",
            record_id="record_001",
            criteria_key="criteria_A",
            content="Test content",
            cache_name="projects/123/cachedContents/abc",
        )

        assert request.custom_id == "record_001_criteria_A"
        assert request.record_id == "record_001"
        assert request.criteria_key == "criteria_A"
        assert request.content == "Test content"
        assert request.cache_name == "projects/123/cachedContents/abc"

    def test_batch_request_optional_cache(self):
        """Test batch request without cache name."""
        request = BatchRequest(
            custom_id="test",
            record_id="rec1",
            criteria_key="crit1",
            content="content",
        )

        assert request.cache_name is None


class TestBatchResult:
    """Tests for BatchResult model."""

    def test_create_success_result(self):
        """Test creating a successful batch result."""
        result = BatchResult(
            custom_id="record_001_criteria_A",
            record_id="record_001",
            criteria_key="criteria_A",
            response="This is the LLM response",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        assert result.response == "This is the LLM response"
        assert result.error is None
        assert result.usage["input_tokens"] == 100

    def test_create_error_result(self):
        """Test creating an error batch result."""
        result = BatchResult(
            custom_id="record_001_criteria_A",
            record_id="record_001",
            criteria_key="criteria_A",
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
            criteria_key="criteria_A",
            content="Evaluate this content",
            cache_name="projects/123/cachedContents/abc",
        )

        entry = manager._build_gemini_request(request)

        assert entry["custom_id"] == "test_001"
        assert "request" in entry
        assert entry["request"]["cached_content"] == "projects/123/cachedContents/abc"
        assert entry["request"]["contents"][0]["role"] == "user"
        assert entry["request"]["contents"][0]["parts"][0]["text"] == "Evaluate this content"

    def test_build_gemini_request_no_cache(self, manager):
        """Test Gemini request without cache."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            criteria_key="criteria_A",
            content="Content",
        )

        entry = manager._build_gemini_request(request)

        assert "cached_content" not in entry["request"]

    def test_build_claude_request(self, manager):
        """Test building a Claude batch request entry."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            criteria_key="criteria_A",
            content="Record content here",
        )

        entry = manager._build_claude_request(
            request,
            criteria_content="Evaluate for hate speech...",
            max_tokens=2048,
        )

        assert entry["custom_id"] == "test_001"
        assert entry["request"]["anthropic_version"] == "vertex-2023-10-16"
        assert entry["request"]["max_tokens"] == 2048

        # Check message structure
        messages = entry["request"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        assert len(content) == 2
        # First block: criteria with cache_control
        assert content[0]["text"] == "Evaluate for hate speech..."
        assert content[0]["cache_control"]["type"] == "ephemeral"
        # Second block: record content
        assert content[1]["text"] == "Record content here"

    def test_build_jsonl_gemini(self, manager):
        """Test building complete JSONL for Gemini."""
        requests = [
            BatchRequest(
                custom_id="rec1_critA",
                record_id="rec1",
                criteria_key="critA",
                content="Content 1",
                cache_name="cache/123",
            ),
            BatchRequest(
                custom_id="rec2_critA",
                record_id="rec2",
                criteria_key="critA",
                content="Content 2",
                cache_name="cache/123",
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gemini-2.5-flash")

        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["custom_id"] == "rec1_critA"

        entry2 = json.loads(lines[1])
        assert entry2["custom_id"] == "rec2_critA"

    def test_build_jsonl_claude(self, manager):
        """Test building complete JSONL for Claude with inline caching."""
        requests = [
            BatchRequest(
                custom_id="rec1_critA",
                record_id="rec1",
                criteria_key="critA",
                content="Content 1",
            ),
        ]

        criteria_contents = {"critA": "Criteria text here"}

        jsonl = manager.build_jsonl(
            requests,
            model="claude-sonnet-4",
            criteria_contents=criteria_contents,
        )

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
        entry = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "The response text"}]
                        }
                    }
                ]
            }
        }

        response = manager._extract_response(entry)
        assert response == "The response text"

    def test_extract_response_claude_format(self, manager):
        """Test extracting response from Claude batch result."""
        entry = {
            "response": {
                "content": [
                    {"type": "text", "text": "Claude response here"}
                ]
            }
        }

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
