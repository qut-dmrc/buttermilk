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


class TestVertexBatchProcessorConfigInheritance:
    """Tests for VertexBatchProcessor inheriting config from model registry."""

    @pytest.fixture
    def mock_model_config(self):
        """Create mock model config with max_output_tokens and region."""
        mock_config = MagicMock()
        mock_config.configs = {
            "model": "claude-sonnet-4-20250514",
            "max_output_tokens": 64000,
            "region": "us-east5",
            "project_id": "test-project",
        }
        return mock_config

    def test_inherits_max_tokens_from_model_config(self, mock_model_config):
        """VertexBatchProcessor should read max_output_tokens from model config.

        When max_tokens is not explicitly set (None), the processor should
        read max_output_tokens from bm.llms.connections[model].configs.
        """
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        # Create processor with max_tokens=None
        processor = VertexBatchProcessor(
            model="claude-sonnet",
            template="test_template",
            max_tokens=None,  # Should inherit from config
        )

        # Mock get_bm() to return a mock BM with llms.connections
        mock_bm_instance = MagicMock()
        mock_bm_instance.llms.connections = {"claude-sonnet": mock_model_config}

        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_instance):
            # Get the resolved max_tokens value
            resolved_max_tokens = processor._get_resolved_max_tokens()

            assert resolved_max_tokens == 64000

    def test_explicit_max_tokens_overrides_config(self, mock_model_config):
        """Explicit max_tokens should override model config.

        When max_tokens is explicitly set, it should take precedence over
        the max_output_tokens value from model config.
        """
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        # Create processor with explicit max_tokens
        processor = VertexBatchProcessor(
            model="claude-sonnet",
            template="test_template",
            max_tokens=4096,  # Explicit override
        )

        # Mock get_bm() to return a mock BM with llms.connections
        mock_bm_instance = MagicMock()
        mock_bm_instance.llms.connections = {"claude-sonnet": mock_model_config}

        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_instance):
            # Get the resolved max_tokens value
            resolved_max_tokens = processor._get_resolved_max_tokens()

            # Explicit value should override config value
            assert resolved_max_tokens == 4096

    def test_inherits_region_from_model_config(self, mock_model_config):
        """VertexBatchProcessor should read region from model config.

        The processor should read region from bm.llms.connections[model].configs
        and use it for all models, not just Gemini 3.
        """
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        processor = VertexBatchProcessor(
            model="claude-sonnet",
            template="test_template",
        )

        # Mock get_bm() to return a mock BM with llms.connections
        mock_bm_instance = MagicMock()
        mock_bm_instance.llms.connections = {"claude-sonnet": mock_model_config}

        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_instance):
            # Get the resolved region value
            resolved_region = processor._get_resolved_region()

            assert resolved_region == "us-east5"

    def test_build_jsonl_passes_max_tokens_to_claude(self):
        """build_jsonl should pass max_tokens for Claude models.

        When building JSONL for Claude batch requests, the max_tokens
        parameter should be passed through to the message converter.
        """
        from buttermilk._core.vertex_batch import BatchJobManager, BatchRequest

        mock_client = MagicMock()
        manager = BatchJobManager(client=mock_client)

        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test content"}],
            ),
        ]

        # Build JSONL with explicit max_tokens
        jsonl = manager.build_jsonl(requests, model="claude-sonnet-4", max_tokens=64000)

        lines = jsonl.strip().split("\n")
        entry = json.loads(lines[0])

        # Verify max_tokens is in the request
        assert entry["request"]["max_tokens"] == 64000

    def test_build_jsonl_uses_default_max_tokens_when_not_specified(self):
        """build_jsonl should use default max_tokens when not specified.

        When max_tokens is None, the Claude converter should use its default value.
        """
        from buttermilk._core.vertex_batch import BatchJobManager, BatchRequest

        mock_client = MagicMock()
        manager = BatchJobManager(client=mock_client)

        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test content"}],
            ),
        ]

        # Build JSONL without explicit max_tokens
        jsonl = manager.build_jsonl(requests, model="claude-sonnet-4")

        lines = jsonl.strip().split("\n")
        entry = json.loads(lines[0])

        # Verify default max_tokens is present (4096 is the default)
        assert entry["request"]["max_tokens"] == 4096


# =============================================================================
# Structured Output Tests
# =============================================================================

SAMPLE_SCHEMA = {
    "title": "JudgeReasons",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["compliant", "non_compliant", "partial"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
}


class TestGeminiStructuredOutput:
    """Tests for Gemini batch converter with structured output."""

    def test_build_request_without_schema(self):
        """Gemini request without schema should not include generationConfig."""
        from buttermilk._core.vertex_batch import GeminiMessageConverter

        converter = GeminiMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this"}],
        )

        entry = converter.build_request(request)

        assert "generationConfig" not in entry["request"]

    def test_build_request_with_schema(self):
        """Gemini request with schema should include generationConfig."""
        from buttermilk._core.vertex_batch import GeminiMessageConverter

        converter = GeminiMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this"}],
            response_schema=SAMPLE_SCHEMA,
        )

        entry = converter.build_request(request)

        assert "generationConfig" in entry["request"]
        gen_config = entry["request"]["generationConfig"]
        assert gen_config["responseMimeType"] == "application/json"
        assert gen_config["responseSchema"] == SAMPLE_SCHEMA

    def test_build_request_schema_preserves_messages(self):
        """Schema should not interfere with message structure."""
        from buttermilk._core.vertex_batch import GeminiMessageConverter

        converter = GeminiMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[
                {"role": "system", "content": "You are a judge"},
                {"role": "user", "content": "Evaluate this"},
            ],
            response_schema=SAMPLE_SCHEMA,
        )

        entry = converter.build_request(request)

        assert "system_instruction" in entry["request"]
        assert entry["request"]["contents"][0]["role"] == "user"
        assert "generationConfig" in entry["request"]


class TestClaudeStructuredOutput:
    """Tests for Claude batch converter with structured output."""

    def test_build_request_without_schema(self):
        """Claude request without schema should not include tools."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this"}],
        )

        entry = converter.build_request(request)

        assert "tools" not in entry["request"]
        assert "tool_choice" not in entry["request"]

    def test_build_request_with_schema(self):
        """Claude request with schema should include tools and tool_choice."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this"}],
            response_schema=SAMPLE_SCHEMA,
        )

        entry = converter.build_request(request)

        assert "tools" in entry["request"]
        tools = entry["request"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "create_judgereasons"
        assert tools[0]["input_schema"] == SAMPLE_SCHEMA

        assert "tool_choice" in entry["request"]
        assert entry["request"]["tool_choice"]["type"] == "tool"
        assert entry["request"]["tool_choice"]["name"] == "create_judgereasons"

    def test_extract_response_tool_use(self):
        """Claude extract_response should handle tool_use blocks."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        entry = {
            "response": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "create_judgereasons",
                        "input": {"verdict": "compliant", "confidence": 0.95, "reasoning": "Meets all criteria"},
                    }
                ]
            }
        }

        response = converter.extract_response(entry)

        assert response is not None
        parsed = json.loads(response)
        assert parsed["verdict"] == "compliant"
        assert parsed["confidence"] == 0.95

    def test_extract_response_text_fallback(self):
        """Claude extract_response should still handle plain text responses."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        entry = {
            "response": {
                "content": [{"type": "text", "text": "Plain text response"}]
            }
        }

        response = converter.extract_response(entry)
        assert response == "Plain text response"

    def test_build_request_schema_preserves_system(self):
        """Schema should not interfere with system message handling."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter(max_tokens=8192)
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[
                {"role": "system", "content": "You are a judge"},
                {"role": "user", "content": "Evaluate this"},
            ],
            response_schema=SAMPLE_SCHEMA,
        )

        entry = converter.build_request(request)

        assert entry["request"]["system"] == "You are a judge"
        assert entry["request"]["max_tokens"] == 8192
        assert "tools" in entry["request"]


class TestBuildJsonlStructuredOutput:
    """Tests for build_jsonl with structured output schema."""

    def test_build_jsonl_gemini_with_schema(self):
        """build_jsonl should pass schema through for Gemini."""
        mock_client = MagicMock()
        manager = BatchJobManager(client=mock_client)

        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test"}],
                response_schema=SAMPLE_SCHEMA,
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gemini-2.5-flash")

        entry = json.loads(jsonl.strip())
        assert "generationConfig" in entry["request"]
        assert entry["request"]["generationConfig"]["responseMimeType"] == "application/json"

    def test_build_jsonl_claude_with_schema(self):
        """build_jsonl should pass schema through for Claude."""
        mock_client = MagicMock()
        manager = BatchJobManager(client=mock_client)

        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test"}],
                response_schema=SAMPLE_SCHEMA,
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="claude-sonnet-4")

        entry = json.loads(jsonl.strip())
        assert "tools" in entry["request"]
        assert entry["request"]["tool_choice"]["type"] == "tool"

    def test_build_jsonl_mixed_schema_and_no_schema(self):
        """Requests with and without schema should coexist."""
        mock_client = MagicMock()
        manager = BatchJobManager(client=mock_client)

        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test 1"}],
                response_schema=SAMPLE_SCHEMA,
            ),
            BatchRequest(
                custom_id="rec2",
                record_id="rec2",
                messages=[{"role": "user", "content": "Test 2"}],
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gemini-2.5-flash")

        lines = jsonl.strip().split("\n")
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert "generationConfig" in entry1["request"]
        assert "generationConfig" not in entry2["request"]


class TestJsonSchemaUtilities:
    """Tests for JSON schema transform utilities."""

    def test_convert_enum_values_to_strings(self):
        """Integer enum values should be converted to strings."""
        from buttermilk._core.json_schema import convert_enum_values_to_strings

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "integer", "enum": [0, 1, 2]},
                "name": {"type": "string"},
            },
        }

        result = convert_enum_values_to_strings(schema)

        assert result["properties"]["status"]["enum"] == ["0", "1", "2"]
        assert result["properties"]["name"]["type"] == "string"

    def test_convert_enum_nested(self):
        """Nested enum values should also be converted."""
        from buttermilk._core.json_schema import convert_enum_values_to_strings

        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer", "enum": [1, 2, 3]},
                    },
                },
            },
        }

        result = convert_enum_values_to_strings(schema)

        assert result["properties"]["nested"]["properties"]["level"]["enum"] == ["1", "2", "3"]

    def test_prepare_schema_for_vertex_gemini(self):
        """prepare_schema_for_vertex with is_gemini should convert enums."""
        from pydantic import BaseModel as PydanticBaseModel

        from buttermilk._core.json_schema import prepare_schema_for_vertex

        class SimpleModel(PydanticBaseModel):
            verdict: str
            confidence: float

        schema = prepare_schema_for_vertex(SimpleModel, is_gemini=True)

        assert "$defs" not in schema
        assert "verdict" in schema.get("required", [])
        assert "confidence" in schema.get("required", [])

    def test_convert_enum_no_mutation(self):
        """convert_enum_values_to_strings should not mutate the input dict."""
        from buttermilk._core.json_schema import convert_enum_values_to_strings

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "integer", "enum": [0, 1, 2]},
            },
        }
        original_enum = schema["properties"]["status"]["enum"].copy()

        convert_enum_values_to_strings(schema)

        # Original should be unchanged
        assert schema["properties"]["status"]["enum"] == original_enum


class TestClaudeToolNameSanitization:
    """Tests for tool name sanitization in Claude converter."""

    def test_tool_name_with_spaces(self):
        """Tool name should sanitize spaces to underscores."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        schema_with_spaces = {
            "title": "Judge Reasons",
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
        }
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this"}],
            response_schema=schema_with_spaces,
        )

        entry = converter.build_request(request)

        tool_name = entry["request"]["tools"][0]["name"]
        assert " " not in tool_name
        assert tool_name == "create_judge_reasons"
        assert entry["request"]["tool_choice"]["name"] == tool_name

    def test_tool_name_with_special_chars(self):
        """Tool name should sanitize special characters."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        schema_with_special = {
            "title": "My.Schema/v2",
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            response_schema=schema_with_special,
        )

        entry = converter.build_request(request)

        tool_name = entry["request"]["tools"][0]["name"]
        # Only [a-z0-9_-] should remain
        import re
        assert re.match(r"^[a-z0-9_\-]+$", tool_name)

    def test_tool_name_missing_title_uses_fallback(self):
        """Missing title should use structured_response as fallback."""
        from buttermilk._core.vertex_batch import ClaudeMessageConverter

        converter = ClaudeMessageConverter()
        schema_no_title = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            response_schema=schema_no_title,
        )

        entry = converter.build_request(request)

        assert entry["request"]["tools"][0]["name"] == "create_structured_response"


class TestBatchRequestSerialization:
    """Tests for BatchRequest serialization behavior."""

    def test_response_schema_excluded_from_serialization(self):
        """response_schema should be excluded from model_dump to avoid manifest bloat."""
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            response_schema=SAMPLE_SCHEMA,
        )

        dumped = request.model_dump()
        assert "response_schema" not in dumped

        # But the attribute should still be accessible for build_request
        assert request.response_schema == SAMPLE_SCHEMA
