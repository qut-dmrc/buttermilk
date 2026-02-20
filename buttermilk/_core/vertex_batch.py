"""Vertex AI batch prediction utilities (Deprecated).

This module is deprecated. Use `buttermilk.batch.managers.vertex` and other submodules in `buttermilk.batch` instead.
"""

from buttermilk.batch.types import BatchRequest, BatchResult
from buttermilk.batch.converters import (
    BatchMessageConverter,
    GeminiMessageConverter,
    ClaudeMessageConverter,
    OpenAIMessageConverter,
    get_message_converter,
    _is_claude_model,
    _is_openai_model,
    _is_llama_model,
)
from buttermilk.batch.manifests import BatchJobManifest, OpenAIBatchManifest
from buttermilk.batch.managers.vertex import BatchJobManager
from buttermilk.batch.managers.openai import OpenAIBatchJobManager
