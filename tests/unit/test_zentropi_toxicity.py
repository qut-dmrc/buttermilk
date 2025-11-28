"""Unit tests for Zentropi toxicity model.

Tests the Zentropi class implementation following ToxicityModel interface.
"""

import pytest

# Skip if torch not available (required by llamaguard imports)
torch = pytest.importorskip("torch")

from buttermilk.toxicity.types import EvalRecord, Score


class TestZentropi:
    """Test Zentropi toxicity model."""

    def test_zentropi_init_with_credentials(self):
        """Test that Zentropi can be instantiated with api_key and base_url."""
        from buttermilk.toxicity.toxicity import Zentropi

        # Should be able to create instance with required credentials
        model = Zentropi(
            model="zentropi",
            process_chain="api",
            standard="zentropi",
            api_key="test_api_key",
            base_url="https://api.zentropi.com/v1",
        )

        assert model.api_key == "test_api_key"
        assert model.base_url == "https://api.zentropi.com/v1"
        assert model.model == "zentropi"
        assert model.process_chain == "api"
        assert model.standard == "zentropi"

    def test_interpret_converts_zentropi_response_to_evalrecord(self):
        """Test that interpret() correctly converts Zentropi API response to EvalRecord.

        Zentropi API returns:
        {
            "toxic": true/false,
            "scores": {
                "toxicity": 0.85,
                "severity": 0.72,
                "confidence": 0.91
            },
            "labels": ["profanity", "hate_speech"]
        }
        """
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="zentropi",
            process_chain="api",
            standard="zentropi",
            api_key="test_key",
            base_url="https://api.zentropi.com/v1",
        )

        # Mock Zentropi API response
        mock_response = {
            "toxic": True,
            "scores": {
                "toxicity": 0.85,
                "severity": 0.72,
                "confidence": 0.91,
            },
            "labels": ["profanity", "hate_speech"],
        }

        # interpret() should convert this to EvalRecord
        result = model.interpret(mock_response)

        # Validate result type and structure
        assert isinstance(result, EvalRecord)
        assert result.prediction is True

        # Validate scores were converted properly
        assert len(result.scores) == 3
        score_measures = {s.measure for s in result.scores}
        assert "toxicity" in score_measures
        assert "severity" in score_measures
        assert "confidence" in score_measures

        # Find toxicity score and verify value
        toxicity_score = next(s for s in result.scores if s.measure == "toxicity")
        assert toxicity_score.score == 0.85

        # Validate labels were copied
        assert set(result.labels) == {"profanity", "hate_speech"}

    def test_interpret_non_toxic_response(self):
        """Test interpret() with non-toxic response."""
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="zentropi",
            process_chain="api",
            standard="zentropi",
            api_key="test_key",
            base_url="https://api.zentropi.com/v1",
        )

        # Non-toxic response
        mock_response = {
            "toxic": False,
            "scores": {
                "toxicity": 0.12,
                "severity": 0.05,
                "confidence": 0.95,
            },
            "labels": [],
        }

        result = model.interpret(mock_response)

        assert isinstance(result, EvalRecord)
        assert result.prediction is False
        assert len(result.labels) == 0
        assert len(result.scores) == 3

    def test_interpret_raises_on_missing_toxic_field(self):
        """Test interpret() raises ValueError when 'toxic' field is missing."""
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="zentropi",
            process_chain="api",
            standard="zentropi",
            api_key="test_key",
            base_url="https://api.zentropi.com/v1",
        )

        # Invalid response missing 'toxic' field
        invalid_response = {
            "scores": {"toxicity": 0.5},
            "labels": [],
        }

        # Should raise ValueError (fail-fast, no defaults)
        with pytest.raises(ValueError, match="toxic"):
            model.interpret(invalid_response)
