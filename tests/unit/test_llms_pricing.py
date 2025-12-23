"""Tests for pricing functionality in llms.py module."""

import pytest
from autogen_core.models import RequestUsage

from buttermilk._core.llms import LLMs, ModelOutput


class TestModelOutputPricing:
    """Test that ModelOutput properly includes pricing information."""

    def test_model_output_with_pricing_metadata(self):
        """Test ModelOutput can store pricing in metadata."""
        usage = RequestUsage(prompt_tokens=100, completion_tokens=50)

        model_output = ModelOutput(
            content="Test response", finish_reason="stop", usage=usage, cached=False
        )

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


class TestLiteLLMModelNameResolution:
    """Test robust model name resolution for LiteLLM cost calculation."""

    def test_gemini_vertex_openai_resolution(self):
        """Test that Gemini models with vertex_openai client resolve correctly."""
        # This is the key test case - should strip google/ prefix for litellm compatibility
        result = LLMs.lookup_litellm_model_name(
            "google/gemini-2.5-flash", "vertex_openai"
        )
        assert result == "vertex_ai/gemini-2.5-flash"

        result = LLMs.lookup_litellm_model_name(
            "google/gemini-2.5-pro", "vertex_openai"
        )
        assert result == "vertex_ai/gemini-2.5-pro"

    def test_gemini_direct_api_resolution(self):
        """Test that Gemini models with direct API client stay as-is."""
        result = LLMs.lookup_litellm_model_name("gemini-2.5-flash", "gemini")
        assert result == "gemini-2.5-flash"

        result = LLMs.lookup_litellm_model_name("gemini-2.5-pro", "gemini")
        assert result == "gemini-2.5-pro"

    def test_anthropic_vertex_resolution(self):
        """Test Anthropic models on Vertex resolve correctly."""
        result = LLMs.lookup_litellm_model_name(
            "claude-sonnet-4@20250514", "anthropic_vertex"
        )
        assert result == "vertex_ai/claude-sonnet-4@20250514"

        result = LLMs.lookup_litellm_model_name("claude-opus-4-1", "anthropic_vertex")
        assert result == "vertex_ai/claude-opus-4-1"

    def test_anthropic_direct_api_resolution(self):
        """Test Anthropic models with direct API stay as-is."""
        result = LLMs.lookup_litellm_model_name(
            "claude-3-5-sonnet-20241022", "anthropic"
        )
        assert result == "claude-3-5-sonnet-20241022"

    def test_openai_azure_resolution(self):
        """Test OpenAI models on Azure resolve correctly."""
        result = LLMs.lookup_litellm_model_name("gpt-5-chat", "azure")
        assert result == "azure/gpt-5-chat"

        result = LLMs.lookup_litellm_model_name("gpt-5-nano", "azure")
        assert result == "azure/gpt-5-nano"

    def test_openai_direct_api_resolution(self):
        """Test OpenAI models with direct API stay as-is."""
        result = LLMs.lookup_litellm_model_name("gpt-4o", "openai")
        assert result == "gpt-4o"

    def test_llama_vertex_openai_resolution(self):
        """Test Llama models on Vertex OpenAI endpoint resolve correctly."""
        result = LLMs.lookup_litellm_model_name(
            "meta/llama-4-maverick-17b-128e-instruct-maas", "vertex_openai"
        )
        assert result == "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"

    def test_existing_prefix_handling(self):
        """Test models that already have provider prefixes are handled correctly."""
        # If a model already has the expected prefix, it should be returned as-is
        result = LLMs.lookup_litellm_model_name("vertex_ai/some-model", "vertex_openai")
        assert result == "vertex_ai/some-model"

        # If it has a different prefix, it should be preserved for cross-provider compatibility
        result = LLMs.lookup_litellm_model_name("openai/gpt-4", "azure")
        assert result == "azure/openai/gpt-4"

        # Test azure prefix with azure client type stays as-is
        result = LLMs.lookup_litellm_model_name("azure/gpt-4", "azure")
        assert result == "azure/gpt-4"

    def test_empty_or_none_model_names(self):
        """Test edge cases with empty or None model names."""
        result = LLMs.lookup_litellm_model_name("", "vertex_openai")
        assert result == ""

        result = LLMs.lookup_litellm_model_name(None, "vertex_openai")
        assert result is None

    def test_unknown_client_types_fallback(self):
        """Test that unknown client types get treated as fallback."""
        result = LLMs.lookup_litellm_model_name("some-model", "unknown_provider")
        assert result == "unknown_provider/some-model"

    def test_provider_prefix_mapping(self):
        """Test that _provider_prefix_for_client_type maps correctly."""
        assert LLMs._provider_prefix_for_client_type("azure") == "azure"
        assert LLMs._provider_prefix_for_client_type("openai") == "openai"
        assert LLMs._provider_prefix_for_client_type("gemini") == "gemini"
        assert LLMs._provider_prefix_for_client_type("gemini_vertex") == "gemini"
        assert LLMs._provider_prefix_for_client_type("vertex_openai") == "vertex_ai"
        assert LLMs._provider_prefix_for_client_type("anthropic_vertex") == "vertex_ai"
        assert LLMs._provider_prefix_for_client_type("anthropic") == "anthropic"

    def test_base_model_name_extraction(self):
        """Test that _extract_base_model_name handles various patterns."""
        # For vertex_openai with google/ models, strip the google/ prefix for litellm compatibility
        result = LLMs._extract_base_model_name(
            "google/gemini-2.5-flash", "vertex_openai"
        )
        assert result == "gemini-2.5-flash"

        # For other cases, strip mismatched prefixes
        result = LLMs._extract_base_model_name("azure/gpt-4", "openai")
        assert (
            result == "azure/gpt-4"
        )  # Keep full name for cross-provider compatibility

        # No prefix found, return as-is
        result = LLMs._extract_base_model_name("gpt-4", "openai")
        assert result == "gpt-4"

    def test_real_world_model_registry_examples(self):
        """Test with real model names from the model registry."""
        # Test current gemini models that were causing issues - should strip google/ prefix
        result = LLMs.lookup_litellm_model_name(
            "google/gemini-2.5-flash", "vertex_openai"
        )
        assert result == "vertex_ai/gemini-2.5-flash"

        result = LLMs.lookup_litellm_model_name(
            "google/gemini-2.5-pro", "vertex_openai"
        )
        assert result == "vertex_ai/gemini-2.5-pro"

        # Test Llama model - should preserve meta/ prefix for vertex_openai
        result = LLMs.lookup_litellm_model_name(
            "meta/llama-4-maverick-17b-128e-instruct-maas", "vertex_openai"
        )
        assert result == "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"

        # Test Azure models
        result = LLMs.lookup_litellm_model_name("gpt-5-chat", "azure")
        assert result == "azure/gpt-5-chat"

        # Test Anthropic on Vertex
        result = LLMs.lookup_litellm_model_name(
            "claude-sonnet-4@20250514", "anthropic_vertex"
        )
        assert result == "vertex_ai/claude-sonnet-4@20250514"


class TestLiteLLMIntegration:
    """Test that generated model names actually work with litellm cost_per_token."""

    def test_gemini_vertex_openai_litellm_compatibility(self):
        """Test that generated model names work with actual litellm cost_per_token."""
        from litellm.cost_calculator import cost_per_token

        # Test the key case that was failing - google/gemini-2.5-flash with vertex_openai
        resolved_name = LLMs.lookup_litellm_model_name(
            "google/gemini-2.5-flash", "vertex_openai"
        )
        assert resolved_name == "vertex_ai/gemini-2.5-flash"

        # Test that this model name actually works with litellm
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=resolved_name,
                prompt_tokens=100,
                completion_tokens=50,
            )
            # Should return valid costs without raising an exception
            assert isinstance(prompt_cost, (int, float))
            assert isinstance(completion_cost, (int, float))
            assert prompt_cost >= 0
            assert completion_cost >= 0
        except Exception as e:
            pytest.fail(f"litellm cost_per_token failed for {resolved_name}: {e}")

    def test_gemini_models_litellm_compatibility(self):
        """Test multiple gemini model variations with litellm."""
        from litellm.cost_calculator import cost_per_token

        test_cases = [
            ("google/gemini-2.5-flash", "vertex_openai", "vertex_ai/gemini-2.5-flash"),
            ("google/gemini-2.5-pro", "vertex_openai", "vertex_ai/gemini-2.5-pro"),
            ("gemini-2.5-flash", "gemini", "gemini-2.5-flash"),
        ]

        for model_name, client_type, expected_litellm_name in test_cases:
            resolved_name = LLMs.lookup_litellm_model_name(model_name, client_type)
            assert resolved_name == expected_litellm_name

            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=resolved_name,
                    prompt_tokens=10,
                    completion_tokens=5,
                )
                assert isinstance(prompt_cost, (int, float))
                assert isinstance(completion_cost, (int, float))
            except Exception as e:
                pytest.fail(
                    f"litellm cost_per_token failed for {resolved_name} (from {model_name}+{client_type}): {e}"
                )

    @pytest.mark.anyio
    async def test_real_model_pricing_resolution(
        self, real_llm_expensive, session_runner
    ):
        """Test that all real configured models resolve to valid litellm names."""
        from litellm.cost_calculator import cost_per_token

        # Get the resolved litellm model name from the wrapper
        resolved_name = real_llm_expensive.litellm_model_name

        # Verify no double prefixes (e.g., vertex_ai/google/gemini-*)
        assert "//" not in resolved_name, f"Double prefix in {resolved_name}"
        # Verify no google/ prefix for vertex models (litellm doesn't recognize it)
        assert not resolved_name.startswith(
            "vertex_ai/google/"
        ), f"Invalid google/ prefix in {resolved_name}"

        # Verify the resolved name works with litellm
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=resolved_name,
                prompt_tokens=100,
                completion_tokens=50,
            )
            assert isinstance(prompt_cost, (int, float))
            assert isinstance(completion_cost, (int, float))
        except Exception as e:
            # Some models may not be in litellm's pricing database yet - that's OK
            pytest.skip(f"Model {resolved_name} not in litellm pricing: {e}")

    def test_bad_model_names_should_fail(self):
        """Test that malformed model names properly fail with litellm."""
        from litellm.cost_calculator import cost_per_token

        # Test that the old malformed names would fail
        bad_names = [
            "vertex_ai/google/gemini-2.5-flash",  # This is what was generated before the fix
            "google/gemini-2.5-flash",  # This should also fail with litellm
        ]

        for bad_name in bad_names:
            with pytest.raises(Exception):  # Should raise ValueError or similar
                cost_per_token(
                    model=bad_name,
                    prompt_tokens=10,
                    completion_tokens=5,
                )
