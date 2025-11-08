"""Tests for the pricing utility module."""

from unittest.mock import patch

from buttermilk.utils.pricing import calculate_token_cost, extract_usage_from_metadata


class TestCalculateTokenCost:
    """Test token cost calculation functionality."""

    @patch("buttermilk.utils.pricing.cost_per_token")
    def test_calculate_token_cost_with_usage_dict_openai(self, mock_cost_per_token):
        """Test token cost calculation with OpenAI format usage dict."""
        # Mock the cost_per_token function
        mock_cost_per_token.return_value = (
            0.001,
            0.002,
        )  # $0.001 prompt, $0.002 completion

        usage_dict = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(
            model="gpt41", usage_dict=usage_dict
        )

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert (
            total_cost == 0.003
        )  # mock returns total cost: 0.001 (prompt) + 0.002 (completion)
        mock_cost_per_token.assert_called_once_with(
            model="azure/gpt-4.1",  # Should map to azure model
            prompt_tokens=100,
            completion_tokens=50,
        )

    @patch("buttermilk.utils.pricing.cost_per_token")
    def test_calculate_token_cost_with_usage_dict_anthropic(self, mock_cost_per_token):
        """Test token cost calculation with Anthropic format usage dict."""
        mock_cost_per_token.return_value = (0.0015, 0.0025)

        usage_dict = {"input_tokens": 200, "output_tokens": 75}

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(
            model="sonnet", usage_dict=usage_dict
        )

        assert prompt_tokens == 200
        assert completion_tokens == 75
        assert total_cost == 0.004  # 0.0015 + 0.0025 = 0.004
        mock_cost_per_token.assert_called_once_with(
            model="vertex_ai/claude-sonnet-4@20250514",  # Should map to vertex AI model
            prompt_tokens=200,
            completion_tokens=75,
        )

    @patch("buttermilk.utils.pricing.cost_per_token")
    def test_calculate_token_cost_with_explicit_tokens(self, mock_cost_per_token):
        """Test token cost calculation with explicitly provided tokens."""
        mock_cost_per_token.return_value = (0.001, 0.002)

        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(
            model="gpt-3.5-turbo", prompt_tokens=150, completion_tokens=100
        )

        assert prompt_tokens == 150
        assert completion_tokens == 100
        assert total_cost == 0.003
        mock_cost_per_token.assert_called_once_with(
            model="gpt-3.5-turbo", prompt_tokens=150, completion_tokens=100
        )

    @patch(
        "buttermilk.utils.pricing.cost_per_token",
        side_effect=Exception("Model not found"),
    )
    def test_calculate_token_cost_with_error(self, mock_cost_per_token):
        """Test token cost calculation handles errors gracefully."""
        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(
            model="unknown-model", prompt_tokens=100, completion_tokens=50
        )

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert total_cost == 0.0  # Returns 0 on error

    def test_calculate_token_cost_no_litellm(self):
        """Test behavior when litellm is not available."""
        with patch("buttermilk.utils.pricing.cost_per_token", None):
            prompt_tokens, completion_tokens, total_cost = calculate_token_cost(
                model="gpt-4", prompt_tokens=100, completion_tokens=50
            )

            assert prompt_tokens == 100
            assert completion_tokens == 50
            assert total_cost == 0.0

    @patch("buttermilk.utils.pricing.cost_per_token")
    def test_model_mapping(self, mock_cost_per_token):
        """Test that buttermilk model names are properly mapped to litellm names."""
        mock_cost_per_token.return_value = (0.001, 0.002)

        # Test different model mappings
        test_cases = [
            ("o4mini", "azure/o4-mini"),
            ("gemini25flash", "gemini/gemini-2.5-flash-preview-05-20"),
            ("sonnet", "vertex_ai/claude-sonnet-4@20250514"),
            ("unknown-model", "unknown-model"),  # Should pass through unmapped
        ]

        for buttermilk_model, expected_litellm_model in test_cases:
            mock_cost_per_token.reset_mock()

            calculate_token_cost(
                model=buttermilk_model, prompt_tokens=100, completion_tokens=50
            )

            mock_cost_per_token.assert_called_once_with(
                model=expected_litellm_model, prompt_tokens=100, completion_tokens=50
            )


class TestExtractUsageFromMetadata:
    """Test usage extraction from metadata."""

    def test_extract_direct_usage(self):
        """Test extracting usage from direct usage field."""
        metadata = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        usage = extract_usage_from_metadata(metadata)
        assert usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_extract_nested_token_usage(self):
        """Test extracting usage from nested outputs.token_usage field."""
        metadata = {
            "outputs": {"token_usage": {"prompt_tokens": 200, "completion_tokens": 100}}
        }

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
