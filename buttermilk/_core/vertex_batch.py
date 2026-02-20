"""Vertex AI batch prediction utilities (Deprecated).

This module is deprecated. Use `buttermilk.batch.managers.vertex` and other submodules in `buttermilk.batch` instead.
"""

from __future__ import annotations

import warnings

# Issue a deprecation warning when this module is imported
warnings.warn(
    "The 'buttermilk._core.vertex_batch' module is deprecated and will be removed in a future version. "
    "Please use 'buttermilk.batch' submodules instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export symbols from new locations for backward compatibility
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
from buttermilk.batch.types import BatchRequest, BatchResult

__all__ = [
    "BatchRequest",
    "BatchResult",
    "BatchMessageConverter",
    "GeminiMessageConverter",
    "ClaudeMessageConverter",
    "OpenAIMessageConverter",
    "_CLAUDE_MODEL_PATTERNS",
    "_OPENAI_MODEL_PATTERNS",
    "_LLAMA_MODEL_PATTERNS",
    "_DEEPSEEK_MODEL_PATTERNS",
    "_is_claude_model",
    "_is_openai_model",
    "_is_llama_model",
    "_is_deepseek_model",
    "get_message_converter",
    "BatchJobManifest",
    "BatchJobManager",
    "OpenAIBatchManifest",
    "OpenAIBatchJobManager",
]
