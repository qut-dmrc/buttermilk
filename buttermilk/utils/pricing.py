"""Token cost utilities for calculating LLM usage costs."""

from __future__ import annotations

from typing import Any

from buttermilk._core.log import logger

try:
    from litellm.cost_calculator import completion_cost, cost_per_token
except ImportError:
    logger.warning("litellm not installed. Token cost tracking will be disabled.")
    completion_cost = None
    cost_per_token = None


# Simple model mappings for backward compatibility
# This is a fallback for when LLMs class resolution is not available
_MODEL_MAPPINGS = {
    "gpt41": "azure/gpt-4.1",
    "o4mini": "azure/o4-mini",
    "gemini25flash": "gemini/gemini-2.5-flash-preview-05-20",
    "sonnet": "vertex_ai/claude-sonnet-4@20250514",
}


def _simple_model_resolution(model_name: str) -> str:
    """Simple model name resolution for backward compatibility.

    This is a fallback when the full LLMs resolution is not available.
    """
    return _MODEL_MAPPINGS.get(model_name, model_name)


def calculate_token_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    usage_dict: dict[str, Any] | None = None,
) -> tuple[int, int, float]:
    """Compute USD token cost for a model.

    Args:
        model: Model name (used for logging)
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        usage_dict: Optional usage dictionary with token counts
        litellm_model: Optional pre-resolved litellm model name for cost calculation

    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_cost)
    """
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

    # Resolve model name if necessary
    cost_model = _simple_model_resolution(model)

    try:
        prompt_cost, completion_cost_val = cost_per_token(
            model=cost_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost_val
        logger.debug(
            f"Token cost for {model} (cost model: {cost_model}): {prompt_tokens} prompt + {completion_tokens} completion = ${total_cost:.6f}",
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
