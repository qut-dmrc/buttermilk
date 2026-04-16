"""Unit tests for ClassifierCore validation."""

import pytest
from pydantic import BaseModel, ValidationError

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.agents.classifier import ClassifierCore

pytestmark = pytest.mark.slow


class ClassificationOutput(BaseModel):
    """Output model for classifier tests."""

    label: int
    confidence: float


class ConcreteClassifier(ClassifierCore):
    """Concrete implementation for testing abstract ClassifierCore."""

    async def _classify(self, text: str) -> dict:
        return {"label": "1", "confidence": 0.9}

    def _map_to_schema(self, response: dict, schema: type[BaseModel]) -> BaseModel:
        return schema(label=int(response["label"]), confidence=response["confidence"])


class TestClassifierCoreValidation:
    """Test suite for ClassifierCore initialization validation."""

    def test_raises_validationerror_when_template_is_empty_string(self):
        """ClassifierCore should raise ValidationError when template is empty string."""
        with pytest.raises(ValidationError):
            ConcreteClassifier(
                template="",
                output_model=ClassificationOutput,
            )

    def test_raises_validationerror_when_template_is_none(self):
        """ClassifierCore should raise ValidationError when template is None."""
        with pytest.raises(ValidationError):
            ConcreteClassifier(
                template=None,
                output_model=ClassificationOutput,
            )

    def test_raises_validationerror_when_output_model_is_none(self):
        """ClassifierCore should raise ValidationError when output_model is None."""
        with pytest.raises(ValidationError):
            ConcreteClassifier(
                template="test/classify",
                output_model=None,
            )

    @pytest.mark.anyio
    async def test_raises_error_when_template_file_not_found(self):
        """ClassifierCore should raise error when template file cannot be found.

        The error may be ProcessingError or RuntimeError depending on whether
        BM singleton is initialized (for error trace logging).
        """
        classifier = ConcreteClassifier(
            template="nonexistent/template/that/does/not/exist",
            output_model=ClassificationOutput,
        )

        record = BaseRecord(record_id="test-1", content="test content")

        # Either ProcessingError (if BM initialized) or RuntimeError (if not)
        # Both indicate the template loading failed as expected
        with pytest.raises((ProcessingError, RuntimeError)):
            async for _ in classifier.process(ProcessingContext(session_id="test", record=record)):
                pass
