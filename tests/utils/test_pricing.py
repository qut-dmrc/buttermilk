"""Tests for the pricing utility module."""

from unittest.mock import MagicMock, patch

import pytest

from buttermilk.utils.pricing import calculate_token_cost, extract_cached_tokens, extract_usage_from_metadata


class TestCalculateTokenCost:
    """Test token cost calculation functionality."""

    @patch("buttermilk.utils.pricing._get_cost_per_token")
    def test_calculate_token_cost_with_usage_dict_openai(self, mock_get_cost_per_token):
        """Test token cost calculation with OpenAI format usage dict."""
        # Mock the _get_cost_per_token function to return a mock cost_per_token
        from unittest.mock import MagicMock

        mock_cost_per_token = MagicMock(return_value=(0.001, 0.002))  # $0.001 prompt, $0.002 completion
        mock_get_cost_per_token.return_value = mock_cost_per_token

        usage_dict = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(model="gpt41", usage_dict=usage_dict)

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert total_cost == 0.003  # mock returns total cost: 0.001 (prompt) + 0.002 (completion)
        mock_cost_per_token.assert_called_once_with(
            model="azure/gpt-4.1",  # Should map to azure model
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_input_tokens=0,
        )

    @patch("buttermilk.utils.pricing._get_cost_per_token")
    def test_calculate_token_cost_with_usage_dict_anthropic(self, mock_get_cost_per_token):
        """Test token cost calculation with Anthropic format usage dict."""
        from unittest.mock import MagicMock

        mock_cost_per_token = MagicMock(return_value=(0.0015, 0.0025))
        mock_get_cost_per_token.return_value = mock_cost_per_token

        usage_dict = {"input_tokens": 200, "output_tokens": 75}

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(model="sonnet", usage_dict=usage_dict)

        assert prompt_tokens == 200
        assert completion_tokens == 75
        assert total_cost == 0.004  # 0.0015 + 0.0025 = 0.004
        mock_cost_per_token.assert_called_once_with(
            model="vertex_ai/claude-sonnet-4@20250514",  # Should map to vertex AI model
            prompt_tokens=200,
            completion_tokens=75,
            cache_read_input_tokens=0,
        )

    @patch("buttermilk.utils.pricing._get_cost_per_token")
    def test_calculate_token_cost_with_explicit_tokens(self, mock_get_cost_per_token):
        """Test token cost calculation with explicitly provided tokens."""
        from unittest.mock import MagicMock

        mock_cost_per_token = MagicMock(return_value=(0.001, 0.002))
        mock_get_cost_per_token.return_value = mock_cost_per_token

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(model="gpt-3.5-turbo", prompt_tokens=150, completion_tokens=100)

        assert prompt_tokens == 150
        assert completion_tokens == 100
        assert total_cost == 0.003
        mock_cost_per_token.assert_called_once_with(model="gpt-3.5-turbo", prompt_tokens=150, completion_tokens=100, cache_read_input_tokens=0)

    @patch("buttermilk.utils.pricing._get_cost_per_token")
    def test_calculate_token_cost_with_error(self, mock_get_cost_per_token):
        """Test token cost calculation handles errors gracefully."""
        from unittest.mock import MagicMock

        mock_cost_per_token = MagicMock(side_effect=Exception("Model not found"))
        mock_get_cost_per_token.return_value = mock_cost_per_token

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(model="unknown-model", prompt_tokens=100, completion_tokens=50)

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert total_cost == 0.0  # Returns 0 on error

    def test_calculate_token_cost_no_litellm(self):
        """Test behavior when litellm is not available."""
        with patch("buttermilk.utils.pricing._get_cost_per_token", return_value=None):
            prompt_tokens, completion_tokens, total_cost = calculate_token_cost(model="gpt-4", prompt_tokens=100, completion_tokens=50)

            assert prompt_tokens == 100
            assert completion_tokens == 50
            assert total_cost == 0.0

    @patch("buttermilk.utils.pricing._get_cost_per_token")
    def test_model_mapping(self, mock_get_cost_per_token):
        """Test that buttermilk model names are properly mapped to litellm names."""
        from unittest.mock import MagicMock

        mock_cost_per_token = MagicMock(return_value=(0.001, 0.002))
        mock_get_cost_per_token.return_value = mock_cost_per_token

        # Test different model mappings
        test_cases = [
            ("o4mini", "azure/o4-mini"),
            ("gemini25flash", "gemini/gemini-2.5-flash-preview-05-20"),
            ("sonnet", "vertex_ai/claude-sonnet-4@20250514"),
            ("unknown-model", "unknown-model"),  # Should pass through unmapped
            # VertexAI MaaS models via OpenAI API - must resolve to vertex_ai/ prefix
            (
                "openai/meta/llama-4-maverick-17b-128e-instruct-maas",
                "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
            ),
        ]

        for buttermilk_model, expected_litellm_model in test_cases:
            mock_cost_per_token.reset_mock()

            calculate_token_cost(model=buttermilk_model, prompt_tokens=100, completion_tokens=50)

            mock_cost_per_token.assert_called_once_with(model=expected_litellm_model, prompt_tokens=100, completion_tokens=50, cache_read_input_tokens=0)


class TestExtractUsageFromMetadata:
    """Test usage extraction from metadata."""

    def test_extract_direct_usage(self):
        """Test extracting usage from direct usage field."""
        metadata = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        usage = extract_usage_from_metadata(metadata)
        assert usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_extract_nested_token_usage(self):
        """Test extracting usage from nested outputs.token_usage field."""
        metadata = {"outputs": {"token_usage": {"prompt_tokens": 200, "completion_tokens": 100}}}

        usage = extract_usage_from_metadata(metadata)
        assert usage == {"prompt_tokens": 200, "completion_tokens": 100}

    def test_extract_nested_usage(self):
        """Test extracting usage from nested outputs.usage field."""
        metadata = {"outputs": {"usage": {"input_tokens": 150, "output_tokens": 75}}}

        usage = extract_usage_from_metadata(metadata)
        assert usage == {"input_tokens": 150, "output_tokens": 75}

    def test_extract_no_usage(self):
        """Test when no usage data is found."""
        metadata = {"some_other_field": "value"}

        usage = extract_usage_from_metadata(metadata)
        assert usage is None

    def test_extract_outputs_not_dict(self):
        """Test when outputs is not a dictionary."""
        metadata = {"outputs": "string_value"}

        usage = extract_usage_from_metadata(metadata)
        assert usage is None

    def test_extract_includes_cached_tokens_from_object(self):
        """Usage objects carrying prompt_tokens_details.cached_tokens surface the cached count."""

        class _Details:
            cached_tokens = 700

        class _Usage:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_tokens_details = _Details()

        metadata = {"usage": _Usage()}
        usage = extract_usage_from_metadata(metadata)
        assert usage == {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 700}


class TestExtractCachedTokens:
    """Cache-read token counts arrive under several provider schemas; all must be handled."""

    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            # OpenAI / Vertex-OpenAI / Gemini implicit-cache (nested dict)
            ({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 800}}, 800),
            # camelCase nested variant
            ({"promptTokensDetails": {"cachedTokens": 600}}, 600),
            # Anthropic-style flat field
            ({"cache_read_input_tokens": 500}, 500),
            # Gemini native batch schema
            ({"cachedContentTokenCount": 400}, 400),
            ({"cached_content_token_count": 300}, 300),
            # flat cached_tokens
            ({"cached_tokens": 200}, 200),
            # no-cache case (the pre-fix default)
            ({"prompt_tokens": 1000, "completion_tokens": 50}, 0),
            (None, 0),
            ({}, 0),
        ],
    )
    def test_extract_cached_tokens_schemas(self, usage, expected):
        assert extract_cached_tokens(usage) == expected

    def test_extract_cached_tokens_from_object_attr(self):
        """A usage object exposing prompt_tokens_details.cached_tokens as an attribute."""

        class _Details:
            cached_tokens = 1234

        class _Usage:
            prompt_tokens_details = _Details()

        assert extract_cached_tokens(_Usage()) == 1234


class TestCacheReadDiscount:
    """The cache-read discount must reach litellm and reduce the total cost.

    These use the REAL litellm pricing tables (local, deterministic — no network)
    rather than mocks, so they verify the actual billing contract end to end.
    """

    def test_cached_tokens_reduce_cost(self):
        """Pricing an identical prompt with cached tokens costs strictly less."""
        _, _, full = calculate_token_cost("gemini/gemini-2.5-flash", prompt_tokens=10_000, completion_tokens=100)
        _, _, cached = calculate_token_cost(
            "gemini/gemini-2.5-flash", prompt_tokens=10_000, completion_tokens=100, cached_tokens=8_000,
        )
        assert full > 0
        assert cached < full, (full, cached)

    def test_cached_count_from_usage_dict_applies_discount(self):
        """A cached count carried inside usage_dict (OpenAI schema) is honoured."""
        _, _, baseline = calculate_token_cost("gemini/gemini-2.5-flash", prompt_tokens=10_000, completion_tokens=100)
        _, _, discounted = calculate_token_cost(
            "gemini/gemini-2.5-flash",
            usage_dict={
                "prompt_tokens": 10_000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 8_000},
            },
        )
        assert discounted < baseline, (baseline, discounted)

    def test_cached_tokens_clamped_to_prompt_tokens(self):
        """cached_tokens > prompt_tokens is clamped, not forwarded raw to litellm."""
        mock_cost = MagicMock(return_value=(0.001, 0.002))
        with patch("buttermilk.utils.pricing._get_cost_per_token", return_value=mock_cost):
            calculate_token_cost("gpt-4", prompt_tokens=100, completion_tokens=50, cached_tokens=99999)
        # cache_read_input_tokens must be clamped to prompt_tokens (100), never exceed it
        assert mock_cost.call_args.kwargs["cache_read_input_tokens"] == 100
