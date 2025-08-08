"""Shared utilities for loading the model registry and resolving provider-qualified model ids.

Centralizes:
  - Loading and caching models.json (tolerating // comment lines)
  - Mapping internal model keys to litellm-style identifiers
  - Provider prefix normalization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buttermilk import logger
from buttermilk._core.constants import CONFIG_CACHE_PATH

# Path to cached model registry (generated elsewhere by the app)
_MODELS_JSON_PATH = Path(CONFIG_CACHE_PATH)
_MODEL_REGISTRY: dict[str, Any] | None = None

# Known litellm provider prefixes (extend as needed)
_KNOWN_LITELLM_PROVIDERS = {
    "azure",
    "openai",
    "gemini",
    "huggingface",
    "vertex_ai",
    "anthropic",
}


def load_model_registry(force: bool = False) -> dict[str, Any]:
    """Load (and cache) the models.json registry. Tolerates // comment lines."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is not None and not force:
        return _MODEL_REGISTRY

    try:
        text = _MODELS_JSON_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug(f"Model registry not found at {_MODELS_JSON_PATH}")
        _MODEL_REGISTRY = {}
        return _MODEL_REGISTRY
    except Exception as e:
        logger.warning(f"Failed reading model registry {_MODELS_JSON_PATH}: {e}")
        _MODEL_REGISTRY = {}
        return _MODEL_REGISTRY

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    try:
        _MODEL_REGISTRY = json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Failed parsing model registry {_MODELS_JSON_PATH}: {e}")
        _MODEL_REGISTRY = {}
    return _MODEL_REGISTRY


def provider_prefix_for_client_type(client_type: str) -> str:
    """Normalize internal client_type to a litellm provider prefix."""
    match client_type:
        case "azure":
            return "azure"
        case "openai":
            return "openai"
        case "gemini" | "gemini_vertex":
            # litellm uses 'gemini' for Gemini API; vertex-hosted Gemini still routes differently upstream
            return "gemini"
        case "huggingface":
            return "huggingface"
        case "vertex_openai" | "anthropic_vertex":
            # Vertex OpenAI-compatible & Anthropic-on-Vertex
            return "vertex_ai"
        case "anthropic":
            return "anthropic"
        case _:
            return client_type  # fallback / extension


def is_already_litellm_identifier(model_name: str, registry: dict[str, Any] | None = None) -> bool:
    """Heuristic: treat as already-qualified if first segment is a known provider and not an internal key."""
    if "/" not in model_name:
        return False
    first = model_name.split("/", 1)[0]
    keys = set(registry.keys()) if registry is not None else get_model_registry_keys()
    return first in _KNOWN_LITELLM_PROVIDERS and model_name not in keys


def resolve_litellm_model_name(internal_name: str) -> str:
    """Resolve an internal model key to a litellm-compatible identifier.

    Order:
      1. If already a provider-qualified litellm id -> return unchanged.
      2. Lookup internal_name in registry; if missing -> return as-is.
      3. If configs.litellm_model present -> return it (assumed fully-qualified or accepted by litellm).
      4. Base id = configs.model or model_info.family
      5. Prefix with normalized provider prefix derived from client_type.
      6. Fallback: original internal_name.

    Safe for absent / partial entries.
    """
    registry = load_model_registry()

    if is_already_litellm_identifier(internal_name, registry):
        return internal_name

    entry = registry.get(internal_name)
    if not entry:
        return internal_name

    client_type = entry.get("client_type")
    configs: dict[str, Any] = entry.get("configs", {}) or {}
    model_info: dict[str, Any] = entry.get("model_info", {}) or {}

    # Explicit override key (optional)
    explicit = configs.get("litellm_model")
    if explicit:
        return explicit

    raw_model = configs.get("model") or model_info.get("family")
    if not raw_model or not client_type:
        return internal_name

    prefix = provider_prefix_for_client_type(client_type)
    return f"{prefix}/{raw_model}"


def list_internal_models() -> list[str]:
    """List internal model keys declared in models.json."""
    return list(load_model_registry().keys())


def get_model_entry(name: str) -> dict[str, Any] | None:
    """Return raw registry entry for inspection."""
    return load_model_registry().get(name)
