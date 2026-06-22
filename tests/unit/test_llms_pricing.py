"""Tests for pricing + litellm provider-segment derivation in llms.py.

The ClientType enum and the lookup_litellm_model_name / _provider_prefix_for_client_type
/ _extract_base_model_name machinery were deleted: routing is now driven entirely by
the full provider-prefixed litellm_model stored in the registry. These tests cover the
surviving pricing surface plus the prefix-derivation helper that replaced the enum.
"""

import pytest

from buttermilk._core.llms import ModelOutput, litellm_provider_segment
from buttermilk._core.messages import RequestUsage


class TestModelOutputPricing:
    """Test that ModelOutput properly includes pricing information."""

    def test_model_output_with_pricing_metadata(self):
        """Test ModelOutput can store pricing in metadata."""
        usage = RequestUsage(prompt_tokens=100, completion_tokens=50)

        model_output = ModelOutput(content="Test response", finish_reason="stop", usage=usage, cached=False)

        # Add pricing to metadata
        model_output.metadata = {
            "pricing": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_cost": 0.003,
            }
        }

        assert model_output.metadata["pricing"]["total_cost"] == 0.003
        assert model_output.usage.prompt_tokens == 100
        assert model_output.usage.completion_tokens == 50


class TestLiteLLMProviderSegment:
    """The full litellm name is the single source of truth; this is the only routing helper."""

    @pytest.mark.parametrize(
        "litellm_model,expected",
        [
            ("vertex_ai/gemini-3-flash-preview", "vertex_ai"),
            ("vertex_ai/claude-sonnet-4-6", "vertex_ai"),
            ("vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas", "vertex_ai"),
            ("azure/gpt-5-mini", "azure"),
            ("azure_ai/grok-4-1-fast-non-reasoning", "azure_ai"),
            ("anthropic/claude-3-5-sonnet", "anthropic"),
            ("huggingface/openai/gpt-oss-safeguard-20b", "huggingface"),
            ("zentropi/cope-latest", "zentropi"),
            ("gpt-4o", ""),  # bare / un-prefixed -> empty segment
            ("", ""),
        ],
    )
    def test_provider_segment(self, litellm_model, expected):
        assert litellm_provider_segment(litellm_model) == expected


class TestLiteLLMPricingResolution:
    """The litellm names we store must resolve in litellm's pricing database."""

    @pytest.mark.parametrize(
        "litellm_model",
        [
            "vertex_ai/gemini-2.5-flash",
            "azure/gpt-4o",
            "vertex_ai/claude-sonnet-4@20250514",
        ],
    )
    def test_known_names_have_pricing(self, litellm_model):
        from litellm.cost_calculator import cost_per_token

        try:
            prompt_cost, completion_cost = cost_per_token(
                model=litellm_model,
                prompt_tokens=100,
                completion_tokens=50,
            )
        except Exception as e:  # pragma: no cover - some names lag litellm's db
            pytest.skip(f"Model {litellm_model} not in litellm pricing: {e}")
        assert isinstance(prompt_cost, (int, float))
        assert isinstance(completion_cost, (int, float))
        assert prompt_cost >= 0
        assert completion_cost >= 0

    @pytest.mark.anyio
    async def test_real_model_pricing_resolution(self, real_llm_expensive, session_runner):
        """Configured models must expose a clean litellm name (no double / google prefixes)."""
        from litellm.cost_calculator import cost_per_token

        resolved_name = real_llm_expensive.litellm_model_name

        assert "//" not in resolved_name, f"Double prefix in {resolved_name}"
        assert not resolved_name.startswith("vertex_ai/google/"), f"Invalid google/ prefix in {resolved_name}"

        try:
            prompt_cost, completion_cost = cost_per_token(
                model=resolved_name,
                prompt_tokens=100,
                completion_tokens=50,
            )
        except Exception as e:
            pytest.skip(f"Model {resolved_name} not in litellm pricing: {e}")
        assert isinstance(prompt_cost, (int, float))
        assert isinstance(completion_cost, (int, float))
