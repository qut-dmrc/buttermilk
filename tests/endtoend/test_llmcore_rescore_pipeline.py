"""End-to-end test for rescore pipeline using real configuration.

This test validates the complete rescore pipeline from buttermilk/conf/run/rescore.yaml:
1. Load real pipeline config from real_bm.cfg.pipeline.rescore
2. Execute full pipeline with both processors:
   - JMESPathTransform (transforms ExecutionTrace → scorer template inputs)
   - LLMCore (runs scoring LLM with QualScore output)
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

Testing Philosophy:
- Uses REAL pipeline config (no fake configs)
- Uses REAL processors from Hydra instantiation
- Uses REAL data from BigQuery
- Tests complete workflow, not isolated components
"""

import asyncio
import datetime

import pytest
from hydra.utils import instantiate

from buttermilk import logger
from buttermilk._core.types import BaseRecord
from buttermilk.agents.evaluators.scorer import QualScore, QualScoreCRA
from buttermilk.utils.trace_writer import get_trace_writer


@pytest.fixture
def sample_scoring_record(real_bm) -> BaseRecord:
    """Create a test record with mock prediction data for scoring.

    This mimics the structure that would come from the rescore pipeline's
    SQL query, which extracts data from ExecutionTrace records and transforms
    it into the format needed for the scorer template.
    """

    # Step 1: Run the ACTUAL SQL query you designed
    query = """
        WITH unscored_predictions AS (
        SELECT DISTINCT t.call_id
        FROM `prosocial-443205.testing.traces` t
        LEFT JOIN `prosocial-443205.testing.traces` scorer
        ON JSON_VALUE(t.inputs, '$.inputs.record_id') = JSON_VALUE(scorer.inputs, '$.inputs.record_id')
        AND JSON_VALUE(scorer.agent_info, '$.role') = 'SCORERS'
        AND JSON_VALUE(scorer.agent_info, '$.config.parameters.model') = JSON_VALUE(t.agent_info, '$.config.parameters.model')
        WHERE JSON_VALUE(t.agent_info, '$.role') IN ('JUDGE', 'SYNTHESISER')
        AND t.outputs.prediction IS NOT NULL
        AND t.error IS NULL
        AND scorer.call_id IS NULL
        AND JSON_VALUE(t.agent_info, '$.config.parameters.model') IN ('gemini25pro', 'claude45sonnet', 'gemini25flash')
    )
    SELECT
        -- answers: array of prediction results to score
        ARRAY(
        SELECT AS STRUCT
            JSON_VALUE(t.agent_info, '$.agent_id') AS agent_id,
            t.outputs AS result,
            t.call_id AS answer_id,
            t.error AS error
        ) AS answers,

        -- criteria: the judge/synth criteria used
        COALESCE(
        JSON_VALUE(t.parameters, '$.criteria'),
        JSON_VALUE(t.agent_info, '$.config.parameters.criteria')
        ) AS criteria,

        -- parent_call_id: link scores back to original prediction
        t.call_id AS parent_call_id,

        -- expected: ground truth from tja_records
        r.ground_truth AS expected,

        -- Additional context (optional, for debugging)
        JSON_VALUE(t.inputs, '$.inputs.record_id') AS record_id,
        JSON_VALUE(t.agent_info, '$.config.parameters.model') AS model

    FROM `prosocial-443205.testing.traces` t
    INNER JOIN unscored_predictions u ON t.call_id = u.call_id
    INNER JOIN `prosocial-443205.bmdev.tja_records` r
        ON JSON_VALUE(t.inputs, '$.inputs.record_id') = r.record_id
    ORDER BY RAND()
    LIMIT 1
    """
    df = real_bm.run_query(query)
    assert df.shape[0] > 0, "SQL query should return at least one record"

    # Step 2: Transform query result into BaseRecord
    row = df.iloc[0]
    record = BaseRecord(
        answers=row.answers,
        criteria=row.criteria,
        parent_call_id=row.parent_call_id,
        expected=row.expected,
    )

    # Step 3: Verify parent_call_id came from SQL correctly
    assert record.parent_call_id is not None
    assert record.parent_call_id == row.parent_call_id  # SQL mapping worked

    return record


@pytest.mark.anyio
async def test_llmcore_rescore_pipeline_full_trace(real_bm, sample_scoring_record: BaseRecord):
    """Test complete rescore pipeline using REAL configuration from testing.yaml.

    This test validates:
    1. Real pipeline config loaded from real_bm.cfg.pipeline.rescore
    2. Both processors execute in sequence (JMESPathTransform → LLMCore)
    3. Processing produces valid QualScore output
    4. ExecutionTrace has all required fields populated correctly
    5. parent_call_id is set for nested tracing
    6. Trace is uploaded to BigQuery with complete data
    """
    # Step 1: Load REAL pipeline configuration
    # This comes from testing.yaml line 5: - run@pipeline.rescore: rescore
    # Which loads buttermilk/conf/run/rescore.yaml
    pipeline_config = real_bm.cfg.pipeline["rescore"]["run"]["pipeline"]

    assert pipeline_config is not None, "Pipeline config should be loaded from testing.yaml"
    assert "processors" in pipeline_config, "Pipeline should have processors"
    assert len(pipeline_config["processors"]) == 2, "Rescore pipeline should have 2 processors (JMESPathTransform + LLMCore)"

    logger.info(f"✅ Loaded real pipeline config with {len(pipeline_config['processors'])} processors")

    # Step 2: Instantiate REAL processors from Hydra config
    processors = [instantiate(proc_config) for proc_config in pipeline_config["processors"]]

    jmespath_processor = processors[0]
    llm_core_processor = processors[1]

    # Verify processor types match rescore.yaml
    assert jmespath_processor.__class__.__name__ == "JMESPathTransform", "First processor should be JMESPathTransform"
    assert llm_core_processor.__class__.__name__ == "LLMCore", "Second processor should be LLMCore"

    logger.info(f"✅ Instantiated processors: {jmespath_processor.__class__.__name__}, {llm_core_processor.__class__.__name__}")

    # Step 3: Extract parent_call_id from record (as pipeline does)
    # This will be used by JMESPathTransform, but LLMCore should use the transformed record's parent_call_id
    parent_trace_id_initial = getattr(sample_scoring_record, "parent_call_id", None)

    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Step 4: Process through FIRST processor (JMESPathTransform)
    logger.info("Processing through JMESPathTransform...")
    transformed_records = []
    async for transformed in jmespath_processor.process(
        record=sample_scoring_record,
        processor_stage="jmespath_transform",
        parent_trace_id=parent_trace_id_initial,
        component_name="JMESPathTransform_Rescore",
    ):
        transformed_records.append(transformed)

    assert len(transformed_records) == 1, "JMESPathTransform should yield exactly one record"
    transformed_record = transformed_records[0]

    # Verify JMESPath transformation worked (should have extracted fields per rescore.yaml lines 89-94)
    assert hasattr(transformed_record, "answers"), "Transformed record should have 'answers' field"
    assert hasattr(transformed_record, "criteria"), "Transformed record should have 'criteria' field"
    assert hasattr(transformed_record, "parent_call_id"), "Transformed record should have 'parent_call_id' field"

    logger.info("✅ JMESPathTransform completed successfully")

    # Step 5: Extract parent_trace_id from TRANSFORMED record
    # JMESPath extracted this from the SQL results - it's the call_id of the judge/synth we're scoring
    parent_trace_id = getattr(transformed_record, "parent_call_id", None)

    # Step 6: Process through SECOND processor (LLMCore scorer)
    logger.info("Processing through LLMCore scorer...")
    final_results = []
    async for result in llm_core_processor.process(
        record=transformed_record,
        processor_stage="rescore_processor",  # Match the query filter
        parent_trace_id=parent_trace_id,
        component_name="LLMCore_Scorer",
    ):
        final_results.append(result)

    # Step 7: Validate final output
    assert len(final_results) == 1, "Expected exactly one result from LLMCore"
    result = final_results[0]

    assert isinstance(result, BaseRecord), "Result should be a BaseRecord"
    assert hasattr(result, "score_results"), "Result should have score_results field (from output_col in config)"
    assert result.score_results is not None, "score_results should not be None"

    # Step 8: Validate QualScore structure
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

    # Step 9: Get trace writer and flush
    trace_writer = get_trace_writer()
    assert trace_writer is not None, "TraceWriter should be initialized"

    # Give a moment for trace to be queued
    await asyncio.sleep(1)

    # Force flush to ensure trace is uploaded
    logger.info("Flushing traces to BigQuery...")
    await trace_writer.flush()

    # Give BigQuery a moment to process
    await asyncio.sleep(2)

    # Step 10: Query BigQuery for the trace
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

    # Step 11: Validate trace was uploaded
    assert df.shape[0] > 0, (
        f"Expected at least one trace in BigQuery. "
        f"Query returned {df.shape[0]} rows. "
        f"Test time: {test_start_time.isoformat()}"
    )

    # Get the most recent trace
    trace = df.iloc[0]

    logger.info(f"✅ Trace found with call_id: {trace.call_id}")

    # Step 12: Validate ExecutionTrace structure - Core fields
    assert trace.call_id is not None, "Trace should have call_id"
    assert isinstance(trace.call_id, str), "call_id should be a string"
    assert len(trace.call_id) > 0, "call_id should not be empty"

    # CRITICAL: Validate parent_call_id is set correctly
    # This is THE KEY FIX - parent_call_id should link scorer back to judge
    assert trace.parent_call_id is not None, "Trace should have parent_call_id for nested operations"
    assert isinstance(trace.parent_call_id, str), "parent_call_id should be a string"
    assert trace.parent_call_id == parent_trace_id, (
        f"parent_call_id should match judge trace ID from transformed record, "
        f"expected '{parent_trace_id}', got '{trace.parent_call_id}'"
    )

    # CRITICAL: Validate call_id is DIFFERENT from parent_call_id
    # This was the bug - scorer was reusing judge's call_id instead of generating new one
    assert trace.call_id != trace.parent_call_id, f"call_id should be NEW (not reuse parent), but both are: {trace.call_id}"

    logger.info(f"✅ parent_call_id correctly set: {trace.parent_call_id}")

    # Step 13: Validate agent_info
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

    # Config is optional - may be empty if no parameters were set
    if "config" in agent_info:
        assert isinstance(agent_info["config"], dict), "config should be a dict if present"
        logger.info(f"Config present with {len(agent_info['config'])} parameters")

    logger.info("✅ agent_info structure validated")

    # Step 14: Validate inputs
    inputs = trace.inputs
    if isinstance(inputs, str):
        import json

        inputs = json.loads(inputs)

    assert inputs is not None, "Trace should have inputs"
    assert isinstance(inputs, dict), "inputs should be a dict"
    assert "answers" in inputs, "inputs should contain 'answers' from template mapping"
    assert "criteria" in inputs, "inputs should contain 'criteria' from template mapping"

    logger.info("✅ inputs structure validated")

    # Step 15: Validate outputs - QualScore object
    outputs = trace.outputs
    if isinstance(outputs, str):
        import json

        outputs = json.loads(outputs)

    assert outputs is not None, "Trace should have outputs"
    assert isinstance(outputs, dict), "outputs should be a dict (serialized QualScore)"
    assert "assessments" in outputs, "outputs should have 'assessments' field from QualScore"

    # BigQuery returns arrays as numpy ndarrays - convert to list
    import numpy as np
    if isinstance(outputs["assessments"], np.ndarray):
        outputs["assessments"] = outputs["assessments"].tolist()

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

    # Step 16: Validate messages
    messages = trace.messages
    if isinstance(messages, str):
        import json

        messages = json.loads(messages)

    # BigQuery returns arrays as numpy ndarrays - convert to list
    import numpy as np
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()

    assert messages is not None, "Trace should have messages"
    assert isinstance(messages, list), f"messages should be a list, got {type(messages).__name__}: {str(messages)[:200]}"
    assert len(messages) > 0, "Should have at least one message (from LLM interaction)"

    logger.info(f"✅ messages validated with {len(messages)} messages")

    # Step 17: Validate metadata
    metadata = trace.metadata
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)

    assert metadata is not None, "Trace should have metadata"
    assert isinstance(metadata, dict), "metadata should be a dict"

    # Check for expected metadata fields
    assert "model" in metadata, "metadata should contain model"
    # Model name comes from real config (rescore.yaml line 98: model: "gemini25pro")
    expected_model = llm_core_processor.model
    assert metadata["model"] == expected_model, f"Should have correct model: {expected_model}"

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

    # Step 18: Validate parameters (optional - may be empty)
    parameters = trace.parameters
    if isinstance(parameters, str):
        import json

        parameters = json.loads(parameters)

    # Parameters may be None or empty dict - both are acceptable
    if parameters is not None:
        assert isinstance(parameters, dict), "parameters should be a dict if present"
        logger.info(f"✅ parameters validated ({len(parameters)} params)")
    else:
        logger.info("✅ parameters field is None (acceptable for empty config)")

    # Step 19: Validate timestamp
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
    logger.info("   - agent_info: component_name, execution_type, config")
    logger.info("   - inputs: answers, criteria")
    logger.info(f"   - outputs: QualScore with {len(outputs['assessments'])} assessments")
    logger.info(f"   - messages: {len(messages)} LLM messages")
    logger.info("   - metadata: model, duration_ms, usage, pricing")
    logger.info("   - parameters: configuration dict")
    logger.info("✅ Trace uploaded to BigQuery successfully")
    logger.info("=" * 80)


@pytest.mark.anyio
async def test_llmcore_rescore_without_parent_trace(real_bm, sample_scoring_record: BaseRecord):
    """Test rescore pipeline WITHOUT parent_call_id using REAL configuration.

    This validates that parent_call_id can be None for top-level operations,
    using the actual pipeline processors from testing.yaml.
    """
    # Load REAL pipeline configuration
    pipeline_config = real_bm.cfg.pipeline["rescore"]["run"]["pipeline"]
    processors = [instantiate(proc_config) for proc_config in pipeline_config["processors"]]

    jmespath_processor = processors[0]
    llm_core_processor = processors[1]

    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing through pipeline without parent_trace_id...")

    # Step 1: JMESPathTransform
    transformed_records = []
    async for transformed in jmespath_processor.process(
        record=sample_scoring_record,
        processor_stage="jmespath_no_parent",
        parent_trace_id=None,  # No parent trace
        component_name="JMESPathTransform_Standalone",
    ):
        transformed_records.append(transformed)

    assert len(transformed_records) == 1, "Should get one transformed record"
    transformed_record = transformed_records[0]

    # Step 2: LLMCore scorer
    results = []
    async for result in llm_core_processor.process(
        record=transformed_record,
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
