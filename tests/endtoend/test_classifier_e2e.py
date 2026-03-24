"""TRUE end-to-end tests for Classifier module.

This test uses REAL components:
- Real HuggingFaceClassifier: Calls HuggingFace models via LiteLLM
- Real ZentropiClassifier: Calls Zentropi API (requires ZENTROPI_API_KEY)
- Real templates: Loads from buttermilk/templates/
- Real text_record: Uses toxic content from conftest.py fixtures

NO mocks except: None - all classifiers call real APIs.

This validates the complete classification workflow from record → template → API → structured output.
"""

import asyncio
import datetime
import json
import os

import pytest

pytestmark = pytest.mark.slow
from pydantic import BaseModel

from buttermilk import logger
from buttermilk._core.types import BaseRecord
from buttermilk.agents.classifier import HuggingFaceClassifier, ZentropiClassifier
from buttermilk.utils.templating import load_template
from buttermilk.utils.trace_writer import get_trace_writer


class ClassificationOutput(BaseModel):
    """Output model for classification results."""

    model_config = {"extra": "forbid"}
    label: int  # 0=safe, 1=toxic
    confidence: float
    categories: list[str] = []


@pytest.mark.anyio
@pytest.mark.parametrize("model", ["gpt-oss-safeguard-20b"])
async def test_huggingface_classifier_e2e(real_bm, session_runner, text_record: BaseRecord, model: str):
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
    async for result in classifier.process(text_record, processor_stage="test_hf_classify"):
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
    async for result in classifier.process(text_record, processor_stage="test_zentropi_classify"):
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
    assert "test_zentropi_classify" in result.metadata, "Should have processor stage metadata"

    stage_metadata = result.metadata["test_zentropi_classify"]
    assert stage_metadata["classifier"] == "ZentropiClassifier"
    assert stage_metadata["template"] == "test/classify"
    assert "template_hash" in stage_metadata
    assert "trace_id" in stage_metadata
    assert "processing_time_ms" in stage_metadata
    assert isinstance(stage_metadata["processing_time_ms"], int)
    assert "api_response" in stage_metadata, "Should include raw API response"


@pytest.mark.anyio
@pytest.mark.parametrize("model", ["gpt-oss-safeguard-20b"])
async def test_huggingface_classifier_input_tracing(real_bm, session_runner, text_record: BaseRecord, model: str):
    """Test that HuggingFaceClassifier correctly stores rendered_prompt in ExecutionTrace.inputs.

    This validates that:
    1. Template is correctly rendered
    2. rendered_prompt is stored in ExecutionTrace.inputs
    3. Inputs can be retrieved from BigQuery for debugging/analysis

    Uses:
    - Real HuggingFace model: gpt-oss-safeguard-20b
    - Real text_record fixture: Contains toxic content
    - Real template: test/classify
    - Real BigQuery trace storage
    """
    # ARRANGE: Create classifier
    classifier = HuggingFaceClassifier(
        model=model,
        template="test/classify",
        output_model=ClassificationOutput,
    )

    # Record test start time for BigQuery query
    test_start_time = datetime.datetime.now(datetime.timezone.utc)
    processor_stage = f"test_hf_input_trace_{model}"

    # ACT: Process text record
    results = []
    async for result in classifier.process(text_record, processor_stage=processor_stage):
        results.append(result)

    # ASSERT: Basic classification worked
    assert len(results) == 1, "Should yield exactly one result"
    assert hasattr(results[0], "output"), "Result should have output field"

    # Flush traces to BigQuery
    trace_writer = get_trace_writer()
    await asyncio.sleep(1)
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Query BigQuery for the trace with inputs
    query = f"""
        SELECT call_id, inputs, metadata, parameters
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
        ORDER BY timestamp DESC
        LIMIT 1
    """

    logger.info(f"Querying BigQuery for classifier trace: {query}")
    df = real_bm.run_query(query)

    # ASSERT: Trace was uploaded
    assert df.shape[0] > 0, f"Expected at least one trace in BigQuery for processor_stage={processor_stage}"

    trace_row = df.iloc[0]

    # Parse inputs
    inputs = trace_row.inputs
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    # ASSERT: rendered_prompt exists in inputs
    assert inputs is not None, "Inputs should not be None"
    assert "rendered_prompt" in inputs, f"Inputs should contain 'rendered_prompt'. Got keys: {inputs.keys()}"

    rendered_prompt = inputs["rendered_prompt"]
    assert rendered_prompt is not None, "rendered_prompt should not be None"
    assert len(rendered_prompt) > 50, "rendered_prompt should contain substantial template content"

    # ASSERT: Content and record_id are also stored
    assert "content" in inputs, "Inputs should contain 'content'"
    assert "record_id" in inputs, "Inputs should contain 'record_id'"
    # Note: We don't compare record_id to text_record.record_id because BigQuery
    # may have eventual consistency issues in fast-running tests

    logger.info("✅ HuggingFace classifier input tracing validated")
    logger.info(f"rendered_prompt length: {len(rendered_prompt)} chars")

    # Validate template_hash in metadata
    metadata = trace_row.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    assert "template_hash" in metadata, "Metadata should contain template_hash"

    # Recompute template hash and verify it matches
    _, _, expected_template_hash = load_template(
        template="test/classify",
        template_vars={},
    )
    assert metadata["template_hash"] == expected_template_hash, "template_hash should match recomputed hash"
    logger.info(f"✅ template_hash validated: {metadata['template_hash'][:16]}...")


@pytest.mark.anyio
@pytest.mark.skipif(
    not os.environ.get("ZENTROPI_API_KEY"),
    reason="ZENTROPI_API_KEY not set",
)
async def test_zentropi_classifier_input_tracing(real_bm, session_runner, text_record: BaseRecord):
    """Test that ZentropiClassifier correctly stores rendered_prompt in ExecutionTrace.inputs.

    This validates that:
    1. Criteria (from template) is captured as rendered_prompt in inputs
    2. Content from record is stored in inputs
    3. Inputs can be retrieved from BigQuery for debugging/analysis

    Uses:
    - Real Zentropi API: Requires ZENTROPI_API_KEY env var
    - Real text_record fixture: Contains toxic content
    - Real template: test/classify
    - Real BigQuery trace storage
    """
    # ARRANGE: Create classifier
    classifier = ZentropiClassifier(
        template="test/classify",
        output_model=ClassificationOutput,
    )

    # Record test start time for BigQuery query
    test_start_time = datetime.datetime.now(datetime.timezone.utc)
    processor_stage = "test_zentropi_input_trace"

    # ACT: Process text record
    results = []
    async for result in classifier.process(text_record, processor_stage=processor_stage):
        results.append(result)

    # ASSERT: Basic classification worked
    assert len(results) == 1, "Should yield exactly one result"
    assert hasattr(results[0], "output"), "Result should have output field"

    # Flush traces to BigQuery
    trace_writer = get_trace_writer()
    await asyncio.sleep(1)
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Query BigQuery for the trace with inputs
    query = f"""
        SELECT call_id, inputs, metadata, parameters
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
        ORDER BY timestamp DESC
        LIMIT 1
    """

    logger.info(f"Querying BigQuery for classifier trace: {query}")
    df = real_bm.run_query(query)

    # ASSERT: Trace was uploaded
    assert df.shape[0] > 0, f"Expected at least one trace in BigQuery for processor_stage={processor_stage}"

    trace_row = df.iloc[0]

    # Parse inputs
    inputs = trace_row.inputs
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    # ASSERT: rendered_prompt exists in inputs (contains criteria from template)
    assert inputs is not None, "Inputs should not be None"
    assert "rendered_prompt" in inputs, f"Inputs should contain 'rendered_prompt'. Got keys: {inputs.keys()}"

    rendered_prompt = inputs["rendered_prompt"]
    assert rendered_prompt is not None, "rendered_prompt should not be None"
    assert len(rendered_prompt) > 20, "rendered_prompt should contain classification criteria from template"

    # ASSERT: Content and record_id are also stored
    assert "content" in inputs, "Inputs should contain 'content'"
    assert "record_id" in inputs, "Inputs should contain 'record_id'"
    # Note: We don't compare record_id/content to text_record because BigQuery
    # may have eventual consistency issues in fast-running tests

    logger.info("✅ Zentropi classifier input tracing validated")
    logger.info(f"rendered_prompt length: {len(rendered_prompt)} chars")

    # Validate template_hash in metadata
    metadata = trace_row.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    assert "template_hash" in metadata, "Metadata should contain template_hash"

    # Recompute template hash and verify it matches
    _, _, expected_template_hash = load_template(
        template="test/classify",
        template_vars={},
    )
    assert metadata["template_hash"] == expected_template_hash, "template_hash should match recomputed hash"
    logger.info(f"✅ template_hash validated: {metadata['template_hash'][:16]}...")
