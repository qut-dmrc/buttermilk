"""Batch message converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from buttermilk.batch.types import BatchRequest

# =============================================================================
# Message Format Converters
# =============================================================================

# Model name patterns that indicate Claude/Anthropic models
_CLAUDE_MODEL_PATTERNS = ("claude", "anthropic")

# Model name patterns that indicate OpenAI/GPT models
_OPENAI_MODEL_PATTERNS = ("gpt", "grok")

# Model name patterns that indicate Llama/Meta models
_LLAMA_MODEL_PATTERNS = ("llama", "meta/")



# Model name patterns that indicate DeepSeek models
_DEEPSEEK_MODEL_PATTERNS = ("deepseek", "deepseek-ai")
class BatchMessageConverter(ABC):
    """Abstract base for converting LiteLLM messages to provider-specific batch format."""

    @abstractmethod
    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Convert a BatchRequest to provider-specific JSONL entry.

        Args:
            request: BatchRequest with messages in LiteLLM format

        Returns:
            Dictionary ready for JSONL serialization
        """

    @abstractmethod
    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response text from provider-specific batch result.

        Args:
            entry: Result entry from batch output

        Returns:
            Extracted text response or None
        """


class GeminiMessageConverter(BatchMessageConverter):
    """Convert messages to/from Gemini batch format.

    Gemini format:
    - System messages → "system_instruction": {"parts": [{"text": ...}]}
    - User/Assistant → "contents": [{"role": "user"|"model", "parts": [{"text": ...}]}]
    - Role mapping: "assistant" → "model"
    """

    # Gemini uses "model" instead of "assistant"
    ROLE_MAP = {"assistant": "model"}

    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Build a Gemini batch request entry.

        When request.response_schema is set, includes generationConfig with
        responseMimeType and responseSchema for native structured output.
        """
        contents = []
        system_parts = []

        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_parts.append({"text": content})
            else:
                # Map role and wrap content in Gemini's parts structure
                gemini_role = self.ROLE_MAP.get(role, role)
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        entry: dict[str, Any] = {
            "custom_id": request.custom_id,
            "request": {"contents": contents},
        }

        if system_parts:
            entry["request"]["system_instruction"] = {"parts": system_parts}

        if request.response_schema:
            entry["request"]["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": request.response_schema,
            }

        return entry

    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response from Gemini batch result.

        Path: response.candidates[0].content.parts[0].text
        """
        response = entry.get("response", {})
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text")
        return None


class ClaudeMessageConverter(BatchMessageConverter):
    """Convert messages to/from Claude batch format.

    Claude format:
    - System messages → "system": "concatenated text"
    - User/Assistant → "messages": [{"role": "user"|"assistant", "content": ...}]
    - Requires: anthropic_version, max_tokens
    """

    DEFAULT_MAX_TOKENS = 4096
    ANTHROPIC_VERSION = "vertex-2023-10-16"

    def __init__(self, max_tokens: int | None = None):
        self.max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Build a Claude batch request entry.

        When request.response_schema is set, includes a tool definition and
        tool_choice to force structured output via the fake-tool pattern
        (same approach as the sync path in LiteLLMWrapper).
        """
        messages = []
        system_parts = []

        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_parts.append(content)
            else:
                # Claude keeps role names as-is, content is direct string
                messages.append({"role": role, "content": content})

        request_body: dict[str, Any] = {
            "anthropic_version": self.ANTHROPIC_VERSION,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if system_parts:
            # Claude concatenates multiple system messages
            request_body["system"] = "\n\n".join(system_parts)

        if request.response_schema:
            import re

            schema = request.response_schema
            schema_name = schema.get("title", "structured_response").lower()
            # Sanitize to valid Anthropic tool name: only [a-z0-9_-]
            schema_name = re.sub(r"[^a-z0-9_\-]", "_", schema_name)
            tool_name = f"create_{schema_name}"
            request_body["tools"] = [
                {
                    "name": tool_name,
                    "description": f"Create a {schema_name} object with the specified fields",
                    "input_schema": schema,
                }
            ]
            request_body["tool_choice"] = {"type": "tool", "name": tool_name}

        return {
            "custom_id": request.custom_id,
            "request": request_body,
        }

    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response from Claude batch result.

        Handles both text responses and tool_use responses (from structured output).
        - Text path: response.content[0].text (where type=="text")
        - Tool path: response.content[0].input (where type=="tool_use"), serialized as JSON
        """
        import json as _json

        response = entry.get("response", {})
        content = response.get("content", [])
        if content and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use":
                    # Structured output via tool use — return the input as JSON string
                    return _json.dumps(block.get("input", {}))
                if block.get("type") == "text":
                    return block.get("text")
        return None


class OpenAIMessageConverter(BatchMessageConverter):
    """Convert messages to/from OpenAI Batch API format.

    OpenAI batch format:
    - Input: {"custom_id": ..., "method": "POST", "url": "/v1/chat/completions",
              "body": {"model": ..., "messages": [...], ...}}
    - Output: {"id": ..., "custom_id": ..., "response": {"status_code": 200,
              "body": {"choices": [...], "usage": {...}}}, "error": null}

    Messages are passed through directly (OpenAI uses the same format as LiteLLM).
    Structured output uses native response_format with json_schema.
    """

    def __init__(self, max_tokens: int | None = None, model: str | None = None, url: str = "/v1/chat/completions"):
        self.max_tokens = max_tokens
        self.model = model
        self.url = url

    def build_request(self, request: BatchRequest) -> dict[str, Any]:
        """Build an OpenAI batch request entry.

        When request.response_schema is set, includes response_format with
        json_schema for native structured output.
        """
        body: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": request.messages,
        }

        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        if request.response_schema:
            schema = request.response_schema
            schema_name = schema.get("title", "structured_response")
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            }

        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": self.url,
            "body": body,
        }

    def extract_response(self, entry: dict[str, Any]) -> str | None:
        """Extract response text from OpenAI batch result.

        Handles both text responses and tool_calls (function calling).
        Path: response.body.choices[0].message.content
        Tool path: response.body.choices[0].message.tool_calls[0].function.arguments
        """
        response = entry.get("response", {})
        body = response.get("body", {}) if isinstance(response, dict) else {}
        choices = body.get("choices", [])

        if not choices:
            return None

        message = choices[0].get("message", {})

        # Check for tool_calls first (structured output via function calling)
        tool_calls = message.get("tool_calls")
        if tool_calls and len(tool_calls) > 0:
            function = tool_calls[0].get("function", {})
            arguments = function.get("arguments")
            if arguments:
                return arguments

        # Standard text content
        content = message.get("content")
        return content


def _is_claude_model(model: str) -> bool:
    """Check if model identifier indicates a Claude/Anthropic model."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in _CLAUDE_MODEL_PATTERNS)


def _is_openai_model(model: str) -> bool:
    """Check if model identifier indicates an OpenAI/GPT model."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in _OPENAI_MODEL_PATTERNS)


def _is_llama_model(model: str) -> bool:
    """Check if model identifier indicates a Llama/Meta model."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in _LLAMA_MODEL_PATTERNS)


def _is_deepseek_model(model: str) -> bool:
    """Check if model identifier indicates a DeepSeek model."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in _DEEPSEEK_MODEL_PATTERNS)


def get_message_converter(model: str, **kwargs: Any) -> BatchMessageConverter:
    """Factory function to get the appropriate converter for a model.

    Args:
        model: Model identifier (e.g., "gemini-2.5-flash", "claude-sonnet-4", "gpt-4o",
               "meta/llama-4-maverick-17b-128e-instruct-maas")
        **kwargs: Provider-specific options (e.g., max_tokens for Claude/OpenAI)

    Returns:
        Appropriate BatchMessageConverter instance
    """
    if _is_claude_model(model):
        return ClaudeMessageConverter(max_tokens=kwargs.get("max_tokens"))
    if _is_openai_model(model) or _is_llama_model(model) or _is_deepseek_model(model):
        return OpenAIMessageConverter(max_tokens=kwargs.get("max_tokens"), model=model)
    return GeminiMessageConverter()
