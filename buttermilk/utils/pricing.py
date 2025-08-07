"""Utility module for calculating LLM token usage costs.

This module provides functions to calculate the cost of LLM API calls based on
token usage, leveraging the litellm library for pricing data.
"""

from typing import Any, Dict, Optional, Tuple

from buttermilk import logger

try:
    from litellm.cost_calculator import completion_cost, cost_per_token
except ImportError:
    logger.warning("litellm not installed. Token cost tracking will be disabled.")
    completion_cost = None
    cost_per_token = None


# Mapping from buttermilk model names to litellm-compatible model names
MODEL_MAPPING = {
    # Azure OpenAI models
    "o4mini": "azure/o4-mini",
    "gpt41": "azure/gpt-4.1",
    "gpt41nano": "azure/gpt-4.1-nano", 
    "gpt41mini": "azure/gpt-4.1-mini",
    
    # Anthropic models (via Vertex)
    "sonnet": "vertex_ai/claude-sonnet-4@20250514",
    "opus": "vertex_ai/claude-opus-4-1-20250805",
    "haiku": "vertex_ai/claude-3-5-haiku",
    
    # Google Gemini models
    "gemini25pro": "gemini/gemini-2.5-pro-preview-03-25",
    "gemini25flash": "gemini/gemini-2.5-flash-preview-05-20",
    
    # Meta Llama models (via Vertex)
    "llama4maverick": "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
    "llama33_70b": "vertex_ai/meta/llama-3.3-70b-instruct-maas",
    "llama32_90b": "vertex_ai/meta/llama-3.2-90b-vision-instruct-maas",
}


def calculate_token_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    usage_dict: Optional[Dict[str, Any]] = None
) -> Tuple[int, int, float]:
    """Calculate the cost of tokens for a specific model.
    
    Args:
        model: The model name (e.g., "gpt-4", "claude-3-sonnet")
        prompt_tokens: Number of prompt/input tokens
        completion_tokens: Number of completion/output tokens
        usage_dict: Optional usage dictionary from LLM response
            Can be in OpenAI format (prompt_tokens/completion_tokens)
            or Anthropic format (input_tokens/output_tokens)
    
    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_cost_usd)
    """
    if cost_per_token is None:
        return prompt_tokens, completion_tokens, 0.0
    
    # Extract tokens from usage dict if provided
    if usage_dict:
        # OpenAI format
        if "prompt_tokens" in usage_dict:
            prompt_tokens = usage_dict.get("prompt_tokens", 0)
            completion_tokens = usage_dict.get("completion_tokens", 0)
        # Anthropic format
        elif "input_tokens" in usage_dict:
            prompt_tokens = usage_dict.get("input_tokens", 0)
            completion_tokens = usage_dict.get("output_tokens", 0)
    
    # Map buttermilk model names to litellm-compatible names
    litellm_model = MODEL_MAPPING.get(model, model)
    
    try:
        # Get cost per token for this model
        prompt_cost, completion_cost = cost_per_token(
            model=litellm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        total_cost = prompt_cost + completion_cost
        
        logger.debug(
            f"Token cost for {model} (mapped to {litellm_model}): {prompt_tokens} prompt + {completion_tokens} completion = ${total_cost:.6f}"
        )
        
        return prompt_tokens, completion_tokens, total_cost
        
    except Exception as e:
        logger.warning(f"Could not calculate token cost for model {model} (mapped to {litellm_model}): {e}")
        return prompt_tokens, completion_tokens, 0.0


def extract_usage_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract usage information from agent output metadata.
    
    Args:
        metadata: The metadata dictionary from AgentOutput
    
    Returns:
        Usage dictionary if found, None otherwise
    """
    # Direct usage field
    if "usage" in metadata:
        usage = metadata["usage"]
        # If it's a RequestUsage object, convert to dict
        if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0)
            }
        # If it's already a dict, return it
        elif isinstance(usage, dict):
            return usage
    
    # Nested in outputs
    if "outputs" in metadata:
        outputs = metadata["outputs"]
        if isinstance(outputs, dict):
            # Check for token_usage field (OpenAI style)
            if "token_usage" in outputs:
                return outputs["token_usage"]
            # Check for usage field (Anthropic style)
            if "usage" in outputs:
                return outputs["usage"]
    
    return None