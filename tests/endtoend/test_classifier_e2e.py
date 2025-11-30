"""TRUE end-to-end tests for Classifier module.

This test uses REAL components:
- Real HuggingFaceClassifier: Calls HuggingFace models via LiteLLM
- Real ZentropiClassifier: Calls Zentropi API (requires ZENTROPI_API_KEY)
- Real templates: Loads from buttermilk/templates/
- Real text_record: Uses toxic content from conftest.py fixtures

NO mocks except: None - all classifiers call real APIs.

This validates the complete classification workflow from record → template → API → structured output.
"""

import os

import pytest
from pydantic import BaseModel

from buttermilk._core.types import BaseRecord
from buttermilk.agents.classifier import HuggingFaceClassifier, ZentropiClassifier


class ClassificationOutput(BaseModel):
    """Output model for classification results."""

    model_config = {"extra": "forbid"}
    label: int  # 0=safe, 1=toxic
    confidence: float
    categories: list[str] = []


@pytest.mark.anyio
@pytest.mark.parametrize("model", ["gpt-oss-safeguard-20b"])
async def test_huggingface_classifier_e2e(
    real_bm, session_runner, text_record: BaseRecord, model: str
):
    """Test HuggingFaceClassifier with real model via LiteLLM.

    Uses:
    - Real HuggingFace model: gpt-oss-safeguard-20b
    - Real text_record fixture: Contains toxic content
    - Real template: test/classify
    - Real LiteLLM API call

    Validates:
    - Classification returns structured output with label, confidence
    - Result metadata contains processor stage info
    - Template metadata is included
    """
    # ARRANGE: Create classifier with real model
    classifier = HuggingFaceClassifier(
        model=model,
        template="test/classify",
        output_model=ClassificationOutput,
    )

    # ACT: Process text record through real classifier
    results = []
    async for result in classifier.process(
        text_record, processor_stage="test_hf_classify"
    ):
        results.append(result)

    # ASSERT: Verify complete classification workflow
    assert len(results) == 1, "Should yield exactly one result"

    result = results[0]
    assert isinstance(result, BaseRecord), "Result should be BaseRecord"

    # Check output structure
    assert hasattr(result, "output"), "Result should have output field"
    output = getattr(result, "output")
    assert "label" in output, "Output should contain label"
    assert "confidence" in output, "Output should contain confidence"
    assert isinstance(output["label"], int), "Label should be integer"
    assert isinstance(output["confidence"], float), "Confidence should be float"
    assert 0 <= output["confidence"] <= 1, "Confidence should be in [0, 1]"

    # Check metadata contains processing info
    assert result.metadata is not None, "Result should have metadata"
    assert "test_hf_classify" in result.metadata, "Should have processor stage metadata"

    stage_metadata = result.metadata["test_hf_classify"]
    assert stage_metadata["classifier"] == "HuggingFaceClassifier"
    assert stage_metadata["template"] == "test/classify"
    assert "template_hash" in stage_metadata
    assert "trace_id" in stage_metadata
    assert "processing_time_ms" in stage_metadata
    assert isinstance(stage_metadata["processing_time_ms"], int)


@pytest.mark.anyio
@pytest.mark.endtoend
@pytest.mark.skipif(
    not os.environ.get("ZENTROPI_API_KEY"),
    reason="ZENTROPI_API_KEY not set",
)
async def test_zentropi_classifier_e2e(real_bm, session_runner, text_record: BaseRecord):
    """Test ZentropiClassifier with real Zentropi API.

    Uses:
    - Real Zentropi API: Requires ZENTROPI_API_KEY env var
    - Real text_record fixture: Contains toxic content
    - Real template: test/classify
    - Real HTTP POST to Zentropi endpoint

    Validates:
    - Classification returns structured output with label, confidence, categories
    - Result metadata contains processor stage info
    - API response is included in metadata
    """
    # ARRANGE: Create classifier with Zentropi API
    classifier = ZentropiClassifier(
        criteria="Classify if the content is toxic or safe based on harmful language, threats, or offensive material.",
        template="test/classify",
        output_model=ClassificationOutput,
    )

    # ACT: Process text record through real classifier
    results = []
    async for result in classifier.process(
        text_record, processor_stage="test_zentropi_classify"
    ):
        results.append(result)

    # ASSERT: Verify complete classification workflow
    assert len(results) == 1, "Should yield exactly one result"

    result = results[0]
    assert isinstance(result, BaseRecord), "Result should be BaseRecord"

    # Check output structure
    assert hasattr(result, "output"), "Result should have output field"
    output = getattr(result, "output")
    assert "label" in output, "Output should contain label"
    assert "confidence" in output, "Output should contain confidence"
    assert "categories" in output, "Output should contain categories"
    assert isinstance(output["label"], int), "Label should be integer"
    assert isinstance(output["confidence"], float), "Confidence should be float"
    assert isinstance(output["categories"], list), "Categories should be list"
    assert 0 <= output["confidence"] <= 1, "Confidence should be in [0, 1]"

    # Check metadata contains processing info
    assert result.metadata is not None, "Result should have metadata"
    assert (
        "test_zentropi_classify" in result.metadata
    ), "Should have processor stage metadata"

    stage_metadata = result.metadata["test_zentropi_classify"]
    assert stage_metadata["classifier"] == "ZentropiClassifier"
    assert stage_metadata["template"] == "test/classify"
    assert "template_hash" in stage_metadata
    assert "trace_id" in stage_metadata
    assert "processing_time_ms" in stage_metadata
    assert isinstance(stage_metadata["processing_time_ms"], int)
    assert "api_response" in stage_metadata, "Should include raw API response"
