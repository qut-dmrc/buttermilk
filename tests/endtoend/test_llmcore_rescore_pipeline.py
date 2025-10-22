"""End-to-end test for LLMCore processor using rescore pipeline configuration.

This test validates the complete LLMCore processor pipeline:
1. LLMCore configured as in rescore.yaml (model, template, output_model)
2. Process a mock record through the processor
3. Validate ExecutionTrace has all required fields:
   - call_id (unique identifier)
   - parent_call_id (for tracing nested operations)
   - agent_info (component configuration)
   - inputs (input data)
   - outputs (QualScore object)
   - messages (LLM messages)
   - metadata (usage, pricing, duration)
   - parameters (configuration)
4. Validate QualScore output structure
5. Verify trace is uploaded to BigQuery
"""

import asyncio
import datetime

import pytest

from buttermilk import logger
from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord
from buttermilk.agents.evaluators.scorer import QualScore, QualScoreCRA
from buttermilk.utils.trace_writer import get_trace_writer


@pytest.fixture
def sample_scoring_record() -> BaseRecord:
    """Create a test record with mock prediction data for scoring.

    This mimics the structure that would come from the rescore pipeline's
    JMESPath transform, which extracts data from ExecutionTrace records.
    """
    return BaseRecord(
        # The JMESPath transform extracts these fields from ExecutionTrace
        text="Sample text for scoring",
        metadata={
            "test": "llmcore_rescore_pipeline",
            "pipeline": "rescore",
        },
        # Additional fields that the scorer template expects
        answers=[
            {
                "agent_id": "test_judge",
                "result": {"prediction": "yes", "reasoning": "This is a test reasoning"},
                "answer_id": "test-answer-123",
                "error": None,
            }
        ],
        criteria=[
            "The reasoning is clear and logical",
            "The prediction is supported by evidence",
            "The response is well-structured",
        ],
    )


@pytest.mark.anyio
async def test_llmcore_rescore_pipeline_full_trace(real_bm, sample_scoring_record: BaseRecord, real_model_name: str):
    """Test LLMCore configured as rescore pipeline processor with complete ExecutionTrace validation.

    This test validates:
    1. LLMCore configuration matching rescore.yaml
    2. Processing produces valid QualScore output
    3. ExecutionTrace has all required fields populated correctly
    4. parent_call_id is set for nested tracing
    5. Trace is uploaded to BigQuery with complete data
    """
    # Step 1: Create LLMCore matching rescore.yaml configuration
    # From rescore.yaml lines 74-79:
    # - _target_: buttermilk._core.llm_core.LLMCore
    #   model: "gemini25pro"
    #   template: "score"
    #   output_model: "buttermilk.agents.evaluators.scorer.QualScore"
    #   output_col: "score_results"
    #   fail_on_unfilled_parameters: true

    parent_trace_id = "test-parent-trace-123"  # Simulate parent pipeline operation

    llm_core = LLMCore(
        model=real_model_name,  # Use real model from fixture
        template="score",  # Scoring template
        output_model=QualScore,  # Structured output
        output_col="score_results",
        fail_on_unfilled_parameters=True,
    )

    # Step 2: Process the record through LLMCore
    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing LLMCore rescore pipeline request...")
    results = []
    async for result in llm_core.process(
        record=sample_scoring_record,
        processor_stage="rescore_processor",
        parent_trace_id=parent_trace_id,  # Link to parent trace
        component_name="LLMCore_Scorer",
    ):
        results.append(result)

    # Step 3: Validate processor output
    assert len(results) == 1, "Expected exactly one result from process()"
    result = results[0]

    assert isinstance(result, BaseRecord), "Result should be a BaseRecord"
    assert hasattr(result, "score_results"), "Result should have score_results field (output_col)"
    assert result.score_results is not None, "score_results should not be None"

    # Step 4: Validate QualScore structure
    qual_score = result.score_results
    assert isinstance(qual_score, QualScore), f"score_results should be QualScore, got {type(qual_score)}"
    assert hasattr(qual_score, "assessments"), "QualScore should have assessments field"
    assert isinstance(qual_score.assessments, list), "assessments should be a list"
    assert len(qual_score.assessments) > 0, "Should have at least one assessment"

    # Validate each assessment
    for idx, assessment in enumerate(qual_score.assessments):
        assert isinstance(assessment, QualScoreCRA), f"Assessment {idx} should be QualScoreCRA"
        assert hasattr(assessment, "correct"), f"Assessment {idx} should have 'correct' field"
        assert isinstance(assessment.correct, bool), f"Assessment {idx} 'correct' should be bool"
        assert hasattr(assessment, "feedback"), f"Assessment {idx} should have 'feedback' field"
        assert isinstance(assessment.feedback, str), f"Assessment {idx} 'feedback' should be str"
        assert len(assessment.feedback) > 0, f"Assessment {idx} feedback should not be empty"

    logger.info(f"✅ QualScore valid with {len(qual_score.assessments)} assessments")

    # Step 5: Get trace writer and flush
    trace_writer = get_trace_writer()
    assert trace_writer is not None, "TraceWriter should be initialized"

    # Give a moment for trace to be queued
    await asyncio.sleep(1)

    # Force flush to ensure trace is uploaded
    logger.info("Flushing traces to BigQuery...")
    await trace_writer.flush()

    # Give BigQuery a moment to process
    await asyncio.sleep(2)

    # Step 6: Query BigQuery for the trace
    query = f"""
        SELECT
            call_id,
            parent_call_id,
            agent_info,
            inputs,
            outputs,
            messages,
            metadata,
            parameters,
            timestamp
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND JSON_VALUE(agent_info, '$.component_name') = 'LLMCore_Scorer'
            AND JSON_VALUE(agent_info, '$.processor_stage') = 'rescore_processor'
        ORDER BY timestamp DESC
        LIMIT 5
    """

    logger.debug(f"Querying BigQuery for uploaded trace:\n{query}")
    df = real_bm.run_query(query)

    # Step 7: Validate trace was uploaded
    assert df.shape[0] > 0, (
        f"Expected at least one trace in BigQuery. "
        f"Query returned {df.shape[0]} rows. "
        f"Test time: {test_start_time.isoformat()}"
    )

    # Get the most recent trace
    trace = df.iloc[0]

    logger.info(f"✅ Trace found with call_id: {trace.call_id}")

    # Step 8: Validate ExecutionTrace structure - Core fields
    assert trace.call_id is not None, "Trace should have call_id"
    assert isinstance(trace.call_id, str), "call_id should be a string"
    assert len(trace.call_id) > 0, "call_id should not be empty"

    # CRITICAL: Validate parent_call_id is set
    assert trace.parent_call_id is not None, "Trace should have parent_call_id for nested operations"
    assert isinstance(trace.parent_call_id, str), "parent_call_id should be a string"
    assert trace.parent_call_id == parent_trace_id, f"parent_call_id should match {parent_trace_id}"

    logger.info(f"✅ parent_call_id correctly set: {trace.parent_call_id}")

    # Step 9: Validate agent_info
    agent_info = trace.agent_info
    if isinstance(agent_info, str):
        import json

        agent_info = json.loads(agent_info)

    assert isinstance(agent_info, dict), "agent_info should be a dict"
    assert "component_name" in agent_info, "agent_info should have component_name"
    assert agent_info["component_name"] == "LLMCore_Scorer", "Should have correct component_name"

    assert "execution_type" in agent_info, "agent_info should have execution_type"
    assert agent_info["execution_type"] == "llm_processing", "Should be llm_processing type"

    assert "processor_stage" in agent_info, "agent_info should have processor_stage"
    assert agent_info["processor_stage"] == "rescore_processor", "Should have correct processor_stage"

    assert "config" in agent_info, "agent_info should have config"
    assert isinstance(agent_info["config"], dict), "config should be a dict"

    logger.info("✅ agent_info structure validated")

    # Step 10: Validate inputs
    inputs = trace.inputs
    if isinstance(inputs, str):
        import json

        inputs = json.loads(inputs)

    assert inputs is not None, "Trace should have inputs"
    assert isinstance(inputs, dict), "inputs should be a dict"
    assert "answers" in inputs, "inputs should contain 'answers' from template mapping"
    assert "criteria" in inputs, "inputs should contain 'criteria' from template mapping"

    logger.info("✅ inputs structure validated")

    # Step 11: Validate outputs - QualScore object
    outputs = trace.outputs
    if isinstance(outputs, str):
        import json

        outputs = json.loads(outputs)

    assert outputs is not None, "Trace should have outputs"
    assert isinstance(outputs, dict), "outputs should be a dict (serialized QualScore)"
    assert "assessments" in outputs, "outputs should have 'assessments' field from QualScore"
    assert isinstance(outputs["assessments"], list), "assessments should be a list"
    assert len(outputs["assessments"]) > 0, "Should have at least one assessment in outputs"

    # Validate assessment structure in outputs
    for idx, assessment in enumerate(outputs["assessments"]):
        assert isinstance(assessment, dict), f"Assessment {idx} should be a dict"
        assert "correct" in assessment, f"Assessment {idx} should have 'correct' field"
        assert isinstance(assessment["correct"], bool), f"Assessment {idx} 'correct' should be bool"
        assert "feedback" in assessment, f"Assessment {idx} should have 'feedback' field"
        assert isinstance(assessment["feedback"], str), f"Assessment {idx} 'feedback' should be str"

    logger.info(f"✅ outputs validated as QualScore with {len(outputs['assessments'])} assessments")

    # Step 12: Validate messages
    messages = trace.messages
    if isinstance(messages, str):
        import json

        messages = json.loads(messages)

    assert messages is not None, "Trace should have messages"
    assert isinstance(messages, list), "messages should be a list"
    assert len(messages) > 0, "Should have at least one message (from LLM interaction)"

    logger.info(f"✅ messages validated with {len(messages)} messages")

    # Step 13: Validate metadata
    metadata = trace.metadata
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)

    assert metadata is not None, "Trace should have metadata"
    assert isinstance(metadata, dict), "metadata should be a dict"

    # Check for expected metadata fields
    assert "model" in metadata, "metadata should contain model"
    assert metadata["model"] == real_model_name, f"Should have correct model: {real_model_name}"

    assert "duration_ms" in metadata, "metadata should contain duration_ms"
    assert isinstance(metadata["duration_ms"], (int, float)), "duration_ms should be numeric"
    assert metadata["duration_ms"] > 0, "duration_ms should be positive"

    assert "usage" in metadata, "metadata should contain usage (token usage)"
    assert isinstance(metadata["usage"], dict), "usage should be a dict"

    # Optional pricing metadata (may not always be present)
    if "pricing" in metadata:
        assert isinstance(metadata["pricing"], dict), "pricing should be a dict"
        logger.info(f"Pricing info available: {metadata['pricing']}")

    logger.info("✅ metadata validated with usage and duration info")

    # Step 14: Validate parameters
    parameters = trace.parameters
    if isinstance(parameters, str):
        import json

        parameters = json.loads(parameters)

    assert parameters is not None, "Trace should have parameters"
    assert isinstance(parameters, dict), "parameters should be a dict"

    logger.info("✅ parameters validated")

    # Step 15: Validate timestamp
    assert trace.timestamp is not None, "Trace should have timestamp"
    assert isinstance(trace.timestamp, (str, datetime.datetime)), "timestamp should be datetime or string"

    # Final success summary
    logger.info("=" * 80)
    logger.info("✅ COMPLETE RESCORE PIPELINE TEST PASSED")
    logger.info("✅ LLMCore configured correctly as pipeline processor")
    logger.info("✅ QualScore output structure validated")
    logger.info("✅ ExecutionTrace has all required fields:")
    logger.info(f"   - call_id: {trace.call_id}")
    logger.info(f"   - parent_call_id: {trace.parent_call_id}")
    logger.info(f"   - agent_info: component_name, execution_type, config")
    logger.info(f"   - inputs: answers, criteria")
    logger.info(f"   - outputs: QualScore with {len(outputs['assessments'])} assessments")
    logger.info(f"   - messages: {len(messages)} LLM messages")
    logger.info(f"   - metadata: model, duration_ms, usage, pricing")
    logger.info(f"   - parameters: configuration dict")
    logger.info("✅ Trace uploaded to BigQuery successfully")
    logger.info("=" * 80)


@pytest.mark.anyio
async def test_llmcore_rescore_without_parent_trace(real_bm, sample_scoring_record: BaseRecord, real_model_name: str):
    """Test LLMCore rescore pipeline processor WITHOUT parent_call_id.

    This validates that parent_call_id can be None for top-level operations.
    """
    llm_core = LLMCore(
        model=real_model_name,
        template="score",
        output_model=QualScore,
        output_col="score_results",
        fail_on_unfilled_parameters=True,
    )

    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing LLMCore without parent_trace_id...")
    results = []
    async for result in llm_core.process(
        record=sample_scoring_record,
        processor_stage="rescore_no_parent",
        parent_trace_id=None,  # No parent trace
        component_name="LLMCore_Scorer_Standalone",
    ):
        results.append(result)

    assert len(results) == 1, "Expected exactly one result"
    result = results[0]
    assert isinstance(result, BaseRecord), "Result should be BaseRecord"
    assert hasattr(result, "score_results"), "Should have score_results"
    assert isinstance(result.score_results, QualScore), "score_results should be QualScore"

    # Flush traces
    trace_writer = get_trace_writer()
    await asyncio.sleep(1)
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Query BigQuery
    query = f"""
        SELECT
            call_id,
            parent_call_id
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND JSON_VALUE(agent_info, '$.component_name') = 'LLMCore_Scorer_Standalone'
        LIMIT 1
    """

    df = real_bm.run_query(query)
    assert df.shape[0] == 1, "Should find exactly one trace"

    trace = df.iloc[0]
    assert trace.call_id is not None, "Should have call_id"

    # parent_call_id should be None or null when not provided
    assert trace.parent_call_id is None or str(trace.parent_call_id).lower() == "none", (
        f"parent_call_id should be None for standalone operations, got: {trace.parent_call_id}"
    )

    logger.info("✅ Standalone operation (no parent_call_id) validated")
