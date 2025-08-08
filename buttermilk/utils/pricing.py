"""Token cost utilities using shared model registry resolution."""
from __future__ import annotations

from typing import Any

from buttermilk import logger

from .model_registry import resolve_litellm_model_name  # unified resolver

try:
    from litellm.cost_calculator import completion_cost, cost_per_token
except ImportError:
    logger.warning("litellm not installed. Token cost tracking will be disabled.")
    completion_cost = None
    cost_per_token = None


def calculate_token_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    usage_dict: dict[str, Any] | None = None,
) -> tuple[int, int, float]:
    """Compute USD token cost for a model (internal or already-qualified)."""
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

    litellm_model = resolve_litellm_model_name(model)

    try:
        prompt_cost, completion_cost_val = cost_per_token(
            model=litellm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost_val
        logger.debug(
            f"Token cost for {model} (resolved {litellm_model}): "
            f"{prompt_tokens} prompt + {completion_tokens} completion = ${total_cost:.6f}",
        )
        return prompt_tokens, completion_tokens, total_cost
    except Exception as e:
        logger.warning(
            f"Could not calculate token cost for model {model} (resolved {litellm_model}): {e}",
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
