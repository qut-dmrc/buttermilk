"""Vertex AI batch prediction utilities (Deprecated).

This module is deprecated. Use `buttermilk.batch` submodules instead.
"""

from __future__ import annotations

from buttermilk._core.log import logger
from buttermilk.batch.converters import (
    _CLAUDE_MODEL_PATTERNS,
    _DEEPSEEK_MODEL_PATTERNS,
    _LLAMA_MODEL_PATTERNS,
    _OPENAI_MODEL_PATTERNS,
    BatchMessageConverter,
    ClaudeMessageConverter,
    GeminiMessageConverter,
    OpenAIMessageConverter,
    _is_claude_model,
    _is_deepseek_model,
    _is_llama_model,
    _is_openai_model,
    get_message_converter,
)
from buttermilk.batch.managers.openai import OpenAIBatchJobManager
from buttermilk.batch.managers.vertex import BatchJobManager
from buttermilk.batch.manifests import BatchJobManifest, OpenAIBatchManifest

# Re-export from new locations for backward compatibility
from buttermilk.batch.types import BatchRequest, BatchResult

__all__ = [
    "BatchRequest",
    "BatchResult",
    "BatchMessageConverter",
    "GeminiMessageConverter",
    "ClaudeMessageConverter",
    "OpenAIMessageConverter",
    "BatchJobManager",
    "OpenAIBatchJobManager",
    "BatchJobManifest",
    "OpenAIBatchManifest",
    "get_message_converter",
    "_is_claude_model",
    "_is_openai_model",
    "_is_llama_model",
    "_is_deepseek_model",
    "_CLAUDE_MODEL_PATTERNS",
    "_OPENAI_MODEL_PATTERNS",
    "_LLAMA_MODEL_PATTERNS",
    "_DEEPSEEK_MODEL_PATTERNS",
    "logger",
]
