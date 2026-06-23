"""Tests for OpenAI batch message converter and response extraction.

These tests cover the OpenAI-specific batch format:
- Input: {"custom_id": ..., "method": "POST", "url": "/v1/chat/completions", "body": {...}}
- Output: {"id": ..., "custom_id": ..., "response": {"status_code": 200, "body": {...}}, "error": null}
"""

import json
from unittest.mock import MagicMock

import pytest

from buttermilk.batch.managers import (
    BatchJobManager,
    BatchRequest,
)

# =============================================================================
# Sample fixtures
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


# =============================================================================
# OpenAI Message Converter Tests
# =============================================================================


class TestOpenAIMessageConverter:
    """Tests for OpenAI batch message converter."""

    def test_import_converter(self):
        """OpenAIMessageConverter should be importable from vertex_batch."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        assert converter is not None

    def test_build_request_basic(self):
        """Build a basic OpenAI batch request with user message."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this content"}],
            model="gpt-4o",
        )

        entry = converter.build_request(request)

        assert entry["custom_id"] == "test_001"
        assert entry["method"] == "POST"
        assert entry["url"] == "/v1/chat/completions"
        assert "body" in entry
        assert entry["body"]["model"] == "gpt-4o"
        assert entry["body"]["messages"] == [{"role": "user", "content": "Evaluate this content"}]

    def test_build_request_with_system_message(self):
        """OpenAI format passes system messages through directly (not extracted)."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            model="gpt-4o",
        )

        entry = converter.build_request(request)

        # OpenAI keeps system messages in the messages array
        messages = entry["body"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_request_with_assistant_message(self):
        """Multi-turn conversation should preserve all roles."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "Evaluate this"},
            ],
            model="gpt-4o",
        )

        entry = converter.build_request(request)

        messages = entry["body"]["messages"]
        assert len(messages) == 4
        assert messages[2]["role"] == "assistant"

    def test_build_request_with_max_tokens(self):
        """max_tokens should be included in the request body."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter(max_tokens=2048)
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
        )

        entry = converter.build_request(request)

        assert entry["body"]["max_tokens"] == 2048

    def test_build_request_without_max_tokens(self):
        """Without max_tokens, body should not include it (let API use default)."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
        )

        entry = converter.build_request(request)

        assert "max_tokens" not in entry["body"]

    def test_build_request_with_structured_output(self):
        """Structured output should use response_format with json_schema."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Evaluate this"}],
            model="gpt-4o",
            response_schema=SAMPLE_SCHEMA,
        )

        entry = converter.build_request(request)

        assert "response_format" in entry["body"]
        rf = entry["body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert "json_schema" in rf
        assert rf["json_schema"]["schema"] == SAMPLE_SCHEMA
        assert rf["json_schema"]["name"] == "JudgeReasons"
        assert rf["json_schema"]["strict"] is True

    def test_build_request_without_structured_output(self):
        """Without schema, no response_format should be set."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
        )

        entry = converter.build_request(request)

        assert "response_format" not in entry["body"]

    def test_build_request_schema_missing_title_uses_fallback(self):
        """Schema without title should use fallback name."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        schema_no_title = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
            response_schema=schema_no_title,
        )

        entry = converter.build_request(request)

        rf = entry["body"]["response_format"]
        assert rf["json_schema"]["name"] == "structured_response"

    def test_build_request_model_from_request(self):
        """Model should come from BatchRequest.model field."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o-mini",
        )

        entry = converter.build_request(request)

        assert entry["body"]["model"] == "gpt-4o-mini"

    def test_build_request_no_model_on_request(self):
        """When model is None on request, body should still have model key (may be None)."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter(model="gpt-4o")
        request = BatchRequest(
            custom_id="test_001",
            record_id="rec_001",
            messages=[{"role": "user", "content": "Test"}],
            model=None,
        )

        entry = converter.build_request(request)

        # Should use the converter's default model
        assert entry["body"]["model"] == "gpt-4o"

    def test_extract_response_success(self):
        """Extract response from successful OpenAI batch result."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        entry = {
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "This is the response.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            }
        }

        response = converter.extract_response(entry)
        assert response == "This is the response."

    def test_extract_response_empty_choices(self):
        """Empty choices should return None."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        entry = {"response": {"status_code": 200, "body": {"choices": []}}}

        response = converter.extract_response(entry)
        assert response is None

    def test_extract_response_no_response(self):
        """Missing response key should return None."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        entry = {"error": {"message": "Something went wrong"}}

        response = converter.extract_response(entry)
        assert response is None

    def test_extract_response_error_status(self):
        """Non-200 status code should still try to extract."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        entry = {
            "response": {
                "status_code": 400,
                "body": {"error": {"message": "Bad request"}},
            }
        }

        # No choices, should return None
        response = converter.extract_response(entry)
        assert response is None

    def test_extract_response_tool_calls(self):
        """Extract response with tool_calls (structured output via function calling)."""
        from buttermilk.batch.managers import OpenAIMessageConverter

        converter = OpenAIMessageConverter()
        entry = {
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "create_judgereasons",
                                            "arguments": '{"verdict": "compliant", "confidence": 0.95}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            }
        }

        response = converter.extract_response(entry)
        # Should return the function arguments as the response
        assert response is not None
        parsed = json.loads(response)
        assert parsed["verdict"] == "compliant"


# =============================================================================
# Model Detection Tests
# =============================================================================


class TestOpenAIModelDetection:
    """Tests for OpenAI/Azure model detection."""

    def test_is_openai_model_gpt(self):
        """GPT models should be detected as OpenAI."""
        from buttermilk.batch.managers import _is_openai_model

        assert _is_openai_model("gpt-4o") is True
        assert _is_openai_model("gpt-4o-mini") is True
        assert _is_openai_model("gpt-3.5-turbo") is True
        assert _is_openai_model("gpt-5-chat") is True

    def test_is_openai_model_negative(self):
        """Non-GPT models should not be detected as OpenAI."""
        from buttermilk.batch.managers import _is_openai_model

        assert _is_openai_model("gemini-2.5-flash") is False
        assert _is_openai_model("claude-sonnet-4") is False
        assert _is_openai_model("meta/llama-4-maverick-17b-128e-instruct-maas") is False

    def test_is_openai_model_case_insensitive(self):
        """Model detection should be case-insensitive."""
        from buttermilk.batch.managers import _is_openai_model

        assert _is_openai_model("GPT-4o") is True
        assert _is_openai_model("Gpt-4o-Mini") is True


# =============================================================================
# Converter Factory Tests
# =============================================================================


class TestConverterFactory:
    """Tests for get_message_converter with OpenAI models."""

    def test_factory_returns_openai_converter_for_gpt(self):
        """Factory should return OpenAIMessageConverter for GPT models."""
        from buttermilk.batch.managers import OpenAIMessageConverter, get_message_converter

        converter = get_message_converter("gpt-4o")
        assert isinstance(converter, OpenAIMessageConverter)

    def test_factory_returns_openai_converter_for_gpt_mini(self):
        """Factory should return OpenAIMessageConverter for gpt-4o-mini."""
        from buttermilk.batch.managers import OpenAIMessageConverter, get_message_converter

        converter = get_message_converter("gpt-4o-mini")
        assert isinstance(converter, OpenAIMessageConverter)

    def test_factory_still_returns_gemini_for_gemini(self):
        """Factory should still return GeminiMessageConverter for Gemini models."""
        from buttermilk.batch.managers import GeminiMessageConverter, get_message_converter

        converter = get_message_converter("gemini-2.5-flash")
        assert isinstance(converter, GeminiMessageConverter)

    def test_factory_still_returns_claude_for_claude(self):
        """Factory should still return ClaudeMessageConverter for Claude models."""
        from buttermilk.batch.managers import ClaudeMessageConverter, get_message_converter

        converter = get_message_converter("claude-sonnet-4")
        assert isinstance(converter, ClaudeMessageConverter)

    def test_factory_passes_max_tokens(self):
        """Factory should pass max_tokens to OpenAI converter."""
        from buttermilk.batch.managers import OpenAIMessageConverter, get_message_converter

        converter = get_message_converter("gpt-4o", max_tokens=8192)
        assert isinstance(converter, OpenAIMessageConverter)
        assert converter.max_tokens == 8192


# =============================================================================
# BatchJobManager Integration Tests
# =============================================================================


class TestBatchJobManagerOpenAI:
    """Tests for BatchJobManager with OpenAI models."""

    @pytest.fixture
    def manager(self):
        mock_client = MagicMock()
        return BatchJobManager(client=mock_client)

    def test_build_jsonl_openai(self, manager):
        """build_jsonl for GPT models should produce OpenAI batch format."""
        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Content 1"}],
                model="gpt-4o",
            ),
            BatchRequest(
                custom_id="rec2",
                record_id="rec2",
                messages=[{"role": "user", "content": "Content 2"}],
                model="gpt-4o",
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gpt-4o")

        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["custom_id"] == "rec1"
        assert entry1["method"] == "POST"
        assert entry1["url"] == "/v1/chat/completions"
        assert entry1["body"]["model"] == "gpt-4o"

    def test_build_jsonl_openai_with_max_tokens(self, manager):
        """build_jsonl for GPT with max_tokens should include it in body."""
        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4o",
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gpt-4o", max_tokens=4096)

        entry = json.loads(jsonl.strip())
        assert entry["body"]["max_tokens"] == 4096

    def test_build_jsonl_openai_with_schema(self, manager):
        """build_jsonl for GPT with schema should include response_format."""
        requests = [
            BatchRequest(
                custom_id="rec1",
                record_id="rec1",
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4o",
                response_schema=SAMPLE_SCHEMA,
            ),
        ]

        jsonl = manager.build_jsonl(requests, model="gpt-4o")

        entry = json.loads(jsonl.strip())
        assert "response_format" in entry["body"]
        assert entry["body"]["response_format"]["type"] == "json_schema"

    def test_extract_response_openai_format(self, manager):
        """_extract_response should handle OpenAI batch output format."""
        entry = {
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "OpenAI response text",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            }
        }

        response = manager._extract_response(entry)
        assert response == "OpenAI response text"
