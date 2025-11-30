"""Unit tests for Zentropi toxicity model.

Tests the Zentropi class implementation following ToxicityModel interface.
"""

import pytest

# Skip if torch not available (required by llamaguard imports)
torch = pytest.importorskip("torch")

from buttermilk.toxicity.types import EvalRecord


class TestZentropi:
    """Test Zentropi toxicity model."""

    def test_zentropi_init_with_credentials(self):
        """Test that Zentropi can be instantiated with credentials dict."""
        from buttermilk.toxicity.toxicity import Zentropi

        # Should be able to create instance with required credentials and criteria
        model = Zentropi(
            model="cope-latest",
            process_chain="api",
            standard="zentropi",
            criteria="Classify the content as toxic or not toxic.",
            credentials={
                "ZENTROPI_API_KEY": "test_api_key",
                "ZENTROPI_BASE_URL": "https://api.zentropi.com/v1",
            },
        )

        assert model.credentials["ZENTROPI_API_KEY"] == "test_api_key"
        assert model.credentials["ZENTROPI_BASE_URL"] == "https://api.zentropi.com/v1"
        assert model.model == "cope-latest"
        assert model.process_chain == "api"
        assert model.standard == "zentropi"
        assert model.criteria == "Classify the content as toxic or not toxic."

    def test_interpret_converts_zentropi_response_to_evalrecord(self):
        """Test that interpret() correctly converts Zentropi API response to EvalRecord.

        Zentropi API returns:
        {
            "label": "1",
            "confidence": 0.87,
            "compute_time": 0.324
        }
        """
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="cope-latest",
            process_chain="api",
            standard="zentropi",
            criteria="Classify as 1 (toxic) or 0 (not toxic).",
            credentials={
                "ZENTROPI_API_KEY": "test_key",
                "ZENTROPI_BASE_URL": "https://api.zentropi.com/v1",
            },
        )

        # Mock Zentropi API response - positive classification
        mock_response = {
            "label": "1",
            "confidence": 0.87,
            "compute_time": 0.324,
        }

        # interpret() should convert this to EvalRecord
        result = model.interpret(mock_response)

        # Validate result type and structure
        assert isinstance(result, EvalRecord)
        assert result.prediction is True

        # Validate scores were converted properly
        assert len(result.scores) == 2
        score_measures = {s.measure for s in result.scores}
        assert "confidence" in score_measures
        assert "compute_time" in score_measures

        # Find confidence score and verify value
        confidence_score = next(s for s in result.scores if s.measure == "confidence")
        assert confidence_score.score == 0.87

        # Validate label was captured
        assert result.labels == ["1"]

    def test_interpret_negative_label_response(self):
        """Test interpret() with negative label response."""
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="cope-latest",
            process_chain="api",
            standard="zentropi",
            criteria="Classify as 1 (toxic) or 0 (not toxic).",
            credentials={
                "ZENTROPI_API_KEY": "test_key",
                "ZENTROPI_BASE_URL": "https://api.zentropi.com/v1",
            },
        )

        # Negative label response
        mock_response = {
            "label": "0",
            "confidence": 0.95,
            "compute_time": 0.215,
        }

        result = model.interpret(mock_response)

        assert isinstance(result, EvalRecord)
        assert result.prediction is False
        assert result.labels == ["0"]
        assert len(result.scores) == 2

    def test_interpret_raises_on_missing_label_field(self):
        """Test interpret() raises ValueError when 'label' field is missing."""
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="cope-latest",
            process_chain="api",
            standard="zentropi",
            criteria="Classify content.",
            credentials={
                "ZENTROPI_API_KEY": "test_key",
                "ZENTROPI_BASE_URL": "https://api.zentropi.com/v1",
            },
        )

        # Invalid response missing 'label' field
        invalid_response = {
            "confidence": 0.5,
            "compute_time": 0.1,
        }

        # Should raise ValueError (fail-fast, no defaults)
        with pytest.raises(ValueError, match="label"):
            model.interpret(invalid_response)

    def test_call_client_raises_without_criteria(self):
        """Test call_client() raises ValueError when criteria is not set."""
        from buttermilk.toxicity.toxicity import Zentropi

        model = Zentropi(
            model="cope-latest",
            process_chain="api",
            standard="zentropi",
            criteria="",  # Empty criteria
            credentials={
                "ZENTROPI_API_KEY": "test_key",
                "ZENTROPI_BASE_URL": "https://api.zentropi.com/v1",
            },
        )

        # Should raise ValueError when criteria is empty
        with pytest.raises(ValueError, match="criteria"):
            model.call_client("test content")


class TestZentropiClassifier:
    """Test ZentropiClassifier implementation following ClassifierCore pattern."""

    def test_zentropi_classifier_loads_criteria_from_template(self):
        """Test that ZentropiClassifier can load criteria from template file.

        ZentropiClassifier should follow the ClassifierCore pattern where:
        1. It accepts a 'template' parameter (required)
        2. It loads criteria from the template file via load_template()
        3. It does NOT require 'criteria' as a positional argument

        This test verifies that ZentropiClassifier can be instantiated with
        a template parameter like other classifiers (HuggingFaceClassifier, etc.)

        Expected to FAIL currently because ZentropiClassifier.__init__()
        requires 'criteria' as a positional argument.
        """
        from pydantic import BaseModel

        from buttermilk.agents.classifier import ZentropiClassifier

        # Define output model for structured classification
        class ToxicityClassification(BaseModel):
            label: int
            confidence: float

        # This should work - template provides criteria via load_template()
        # Currently FAILS with: TypeError: __init__() missing 1 required positional argument: 'criteria'
        classifier = ZentropiClassifier(
            template="test/classify",  # Template file in buttermilk/templates/
            output_model=ToxicityClassification,
        )

        # Verify template was stored
        assert classifier.template == "test/classify"
        assert classifier.output_model == ToxicityClassification
