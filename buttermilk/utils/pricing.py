"""Token cost utilities for calculating LLM usage costs."""

from __future__ import annotations

from typing import Any

from buttermilk._core.log import logger

# Lazy-load litellm cost calculator to avoid slow import at module load time
_cost_per_token = None
_cost_per_token_loaded = False


def _get_cost_per_token() -> Any:
    """Get cost_per_token function, lazy-loading litellm on first use."""
    global _cost_per_token, _cost_per_token_loaded
    if not _cost_per_token_loaded:
        try:
            from litellm.cost_calculator import cost_per_token

            _cost_per_token = cost_per_token
        except ImportError:
            logger.warning("litellm not installed. Token cost tracking will be disabled.")
            _cost_per_token = None
        _cost_per_token_loaded = True
    return _cost_per_token


# Simple model mappings for backward compatibility
# This is a fallback for when LLMs class resolution is not available
_MODEL_MAPPINGS = {
    "gpt41": "azure/gpt-4.1",
    "o4mini": "azure/o4-mini",
    "gemini25flash": "gemini/gemini-2.5-flash-preview-05-20",
    "sonnet": "vertex_ai/claude-sonnet-4@20250514",
    # Claude model aliases used in batch manifests
    "claude-sonnet-4-6": "vertex_ai/claude-sonnet-4-6@20250514",
    "claude-haiku-4-5@20251001": "vertex_ai/claude-3-5-haiku@20241022",
    "claude-opus-4-1": "vertex_ai/claude-opus-4@20250514",
}


def _simple_model_resolution(model_name: str | None) -> str:
    """Normalize model names to litellm-compatible format for pricing lookups.

    Handles various model name formats:
    - Direct mappings from _MODEL_MAPPINGS
    - Double-prefixed names like "openai/google/gemini-2.5-flash" → "gemini/gemini-2.5-flash"
    - Names with incorrect provider prefixes
    """
    if model_name is None:
        return "unknown"

    # Check direct mappings first
    if model_name in _MODEL_MAPPINGS:
        return _MODEL_MAPPINGS[model_name]

    # Handle double-prefixed model names (e.g., "openai/google/gemini-3.1-flash-lite")
    # This occurs when Vertex models are incorrectly prefixed for pricing
    if model_name.count("/") >= 2:
        parts = model_name.split("/")
        # Check for pattern: provider/google/gemini-*
        if len(parts) >= 3 and parts[1] == "google" and parts[2].startswith("gemini"):
            # Extract base model name (e.g., "gemini-2.5-flash-lite")
            base_model = "/".join(parts[2:])
            # Return with gemini/ prefix for litellm pricing
            return f"gemini/{base_model}"
        # Check for pattern: openai/meta/llama-* (VertexAI MaaS models via OpenAI API)
        if len(parts) >= 3 and parts[0] == "openai" and parts[1] == "meta":
            # Return with vertex_ai/ prefix for litellm pricing: vertex_ai/meta/llama-*
            return "vertex_ai/" + "/".join(parts[1:])

    # Handle single wrong prefix (e.g., "openai/gemini-2.5-flash")
    if "/" in model_name:
        prefix, base = model_name.split("/", 1)
        # If it's a gemini model with wrong prefix, fix it
        if base.startswith("gemini") and prefix not in {"gemini", "vertex_ai"}:
            return f"gemini/{base}"

    return model_name


def extract_cached_tokens(usage: Any) -> int:
    """Extract the cache-read (cached prompt) token count from a usage object or dict.

    For a live litellm ``Usage`` object this reduces to reading litellm's normalised
    ``prompt_tokens_details.cached_tokens`` (litellm folds Anthropic
    ``cache_read_input_tokens``, DeepSeek ``prompt_cache_hit_tokens`` etc. into that
    field — verified 2026-06-22: azure_ai returned cached_tokens via this path).

    The remaining flat/camelCase branches are KEPT-BECAUSE-LITELLM-GAP: this helper is
    ALSO called on raw, NON-litellm usage dicts produced by the batch path
    (``_core/vertex_batch.py`` reads ``response.body.usage`` straight from the
    provider's batch-result JSONL and passes it as ``usage_dict``). Those raw Vertex/
    Gemini batch shapes are not normalised by litellm, so the
    ``cachedContentTokenCount`` / ``cached_content_token_count`` / flat
    ``cache_read_input_tokens`` / ``cached_tokens`` fallbacks are retained
    (fail-safe; the exact Vertex batch usage schema was not live-submitted here).

    Checked in order:
    - ``prompt_tokens_details.cached_tokens`` (object attr or nested dict) — litellm-normalised.
    - Anthropic-style flat field: ``cache_read_input_tokens``.
    - Gemini native batch schema: ``cachedContentTokenCount`` / ``cached_content_token_count``.
    - Flat ``cached_tokens``.

    Returns 0 when no cache hit is reported (the common no-cache case).
    """
    if usage is None:
        return 0

    def _coerce(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    # Nested prompt_tokens_details.cached_tokens (object or dict)
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details") or usage.get("promptTokensDetails")
    if details is not None:
        if isinstance(details, dict):
            cached = details.get("cached_tokens") or details.get("cachedTokens")
        else:
            cached = getattr(details, "cached_tokens", None)
        if cached:
            return _coerce(cached)

    # Flat fields across providers
    for attr in ("cache_read_input_tokens", "cachedContentTokenCount", "cached_content_token_count", "cached_tokens"):
        if isinstance(usage, dict):
            if usage.get(attr):
                return _coerce(usage.get(attr))
        else:
            value = getattr(usage, attr, None)
            if value:
                return _coerce(value)

    return 0


def calculate_token_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    usage_dict: dict[str, Any] | None = None,
    cached_tokens: int = 0,
) -> tuple[int, int, float]:
    """Compute USD token cost for a model, applying any cache-read discount.

    Args:
        model: Model name (used for logging)
        prompt_tokens: Number of prompt tokens (TOTAL input, including cached)
        completion_tokens: Number of completion tokens
        usage_dict: Optional usage dictionary with token counts. If it carries a
            cached-token count (any supported schema), it is extracted and the
            cache-read discount applied.
        cached_tokens: Subset of ``prompt_tokens`` served from cache. Billed at the
            provider's reduced cache-read rate. Overridden by usage_dict if that
            carries a cached count.

    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_cost). ``total_cost``
        reflects the cache-read discount when cached_tokens > 0.
    """
    cost_per_token = _get_cost_per_token()
    if cost_per_token is None:
        return prompt_tokens, completion_tokens, 0.0

    # Accept usage dict variants
    if usage_dict:
        if "prompt_tokens" in usage_dict:
            prompt_tokens = usage_dict.get("prompt_tokens", 0)
            completion_tokens = usage_dict.get("completion_tokens", 0)
        elif "input_tokens" in usage_dict:
            prompt_tokens = usage_dict.get("input_tokens", 0)
            completion_tokens = usage_dict.get("output_tokens", 0)
        # Prefer cached count carried in the usage dict over the explicit arg
        usage_cached = extract_cached_tokens(usage_dict)
        if usage_cached:
            cached_tokens = usage_cached

    # Cached tokens are a subset of prompt_tokens; guard against bad inputs.
    cached_tokens = max(0, min(int(cached_tokens or 0), int(prompt_tokens or 0)))

    # Resolve model name if necessary
    cost_model = _simple_model_resolution(model)

    try:
        # litellm bills prompt_tokens at the full rate then re-prices the
        # cache_read_input_tokens subset at the model's cache-read rate.
        prompt_cost, completion_cost_val = cost_per_token(
            model=cost_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
        )
        total_cost = prompt_cost + completion_cost_val
        logger.debug(
            f"Token cost for {model} (cost model: {cost_model}): {prompt_tokens} prompt "
            f"({cached_tokens} cached) + {completion_tokens} completion = ${total_cost:.6f}",
        )
        return prompt_tokens, completion_tokens, total_cost
    except Exception as e:
        logger.warning(
            f"Could not calculate token cost for model {model} (cost model: {cost_model}): {e}",
            model=model,
            cost_model=cost_model,
            error=str(e),
        )
        return prompt_tokens, completion_tokens, 0.0


def extract_usage_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract usage dict from heterogeneous metadata structures."""
    if "usage" in metadata:
        usage = metadata["usage"]
        if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "cached_tokens": extract_cached_tokens(usage),
            }
        if isinstance(usage, dict):
            return usage
    if "outputs" in metadata and isinstance(metadata["outputs"], dict):
        outputs = metadata["outputs"]
        if "token_usage" in outputs:
            return outputs["token_usage"]
        if "usage" in outputs:
            return outputs["usage"]
    return None
