"""End-to-end test for LLMCore with ExecutionTrace upload to BigQuery.

This test validates the complete tracing pipeline:
1. LLMCore processes a request with a live LLM
2. ExecutionTrace is created with metadata
3. TraceWriter queues the trace via AsyncDataUploader
4. Trace is flushed to BigQuery
5. Trace can be retrieved from BigQuery
"""

import asyncio
import datetime

import pytest

from buttermilk import logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.trace_writer import get_trace_writer


@pytest.fixture
def sample_record() -> BaseRecord:
    """Create a simple test record."""
    return BaseRecord(text="What is the capital of France?", metadata={"test": "llmcore_tracing"})


@pytest.mark.anyio
async def test_llmcore_with_bigquery_trace(real_bm, sample_record: BaseRecord, real_model_name: str):
    """Test LLMCore processes request and uploads ExecutionTrace to BigQuery.

    This test:
    1. Creates an LLMCore instance with a cheap model
    2. Processes a simple request
    3. Verifies the LLM response is valid
    4. Forces trace writer to flush
    5. Queries BigQuery to verify trace was uploaded
    6. Validates trace structure and metadata
    """
    # Step 1: Create LLMCore with cheap model and simple template
    llm_core = LLMCore(
        model=real_model_name,  # Use a real model from the fixture
        template="ra",  # Simple research assistant template
        fail_on_unfilled_parameters=False,
    )

    # Step 2: Process the request
    test_start_time = datetime.datetime.now(datetime.timezone.utc)
    trace_id = None

    logger.info("Processing LLMCore request...")
    results = []
    async for result in llm_core.process(
        record=sample_record, processor_stage="test_stage", component_name="test_llmcore", prompt="What is the capital of France?"
    ):
        results.append(result)

    # Step 3: Validate LLM response
    assert len(results) == 1, "Expected exactly one result from process()"
    result = results[0]

    assert isinstance(result, BaseRecord), "Result should be a BaseRecord"
    assert hasattr(result, "output"), "Result should have output field"
    assert result.output is not None, "Output should not be None"

    # Check if the output contains a reasonable answer (case-insensitive)
    output_str = str(result.output).lower()
    assert "paris" in output_str, f"Expected 'Paris' in output, got: {result.output}"

    logger.info(f"✅ LLM Response valid: {result.output[:100]}...")

    # Step 4: Get the trace writer and force flush
    trace_writer = get_trace_writer()
    assert trace_writer is not None, "TraceWriter should be initialized"

    # Give a moment for trace to be queued
    await asyncio.sleep(1)

    # Force flush to ensure trace is uploaded
    logger.info("Flushing traces to BigQuery...")
    await trace_writer.flush()

    # Give BigQuery a moment to process the upload
    await asyncio.sleep(2)

    # Query for traces created in the last few minutes with our test metadata
    query = f"""
        SELECT
            call_id,
            agent_info,
            inputs,
            outputs,
            metadata,
            parameters,
            timestamp
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND JSON_VALUE(agent_info, '$.component_name') = 'test_llmcore'
            AND JSON_VALUE(agent_info, '$.processor_stage') = 'test_stage'
        ORDER BY timestamp DESC
        LIMIT 5
    """

    logger.debug(f"Querying BigQuery for uploaded trace... Executing query:\n{query}")
    df = real_bm.run_query(query)

    # Step 6: Validate trace was uploaded
    assert df.shape[0] > 0, (
        f"Expected at least one trace in BigQuery. "
        f"Query returned {df.shape[0]} rows. "
        f"Check that:\n"
        f"1. TraceWriter is configured (conf/storage/traces.yaml)\n"
        f"2. AsyncDataUploader is working\n"
        f"3. BigQuery table exists and is accessible\n"
        f"Test time: {test_start_time.isoformat()}"
    )

    # Get the most recent trace
    trace = df.iloc[0]

    logger.info(f"Trace call_id: {trace.call_id}")

    # Validate trace structure
    assert trace.call_id is not None, "Trace should have call_id"
    assert trace.agent_info is not None, "Trace should have agent_info"
    assert trace.inputs is not None, "Trace should have inputs"
    assert trace.outputs is not None, "Trace should have outputs"
    assert trace.metadata is not None, "Trace should have metadata"

    # Validate agent_info
    agent_info = trace.agent_info
    if isinstance(agent_info, str):
        import json

        agent_info = json.loads(agent_info)

    assert agent_info.get("component_name") == "test_llmcore", "Should have correct component_name"
    assert agent_info.get("execution_type") == "llm_processing", "Should be llm_processing type"
    assert agent_info.get("processor_stage") == "test_stage", "Should have correct processor_stage"

    # Validate metadata contains LLM info
    metadata = trace.metadata
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)

    assert "model" in metadata, "Metadata should contain model"
    assert metadata["model"] == real_model_name, "Should have correct model"
    assert "duration_ms" in metadata, "Metadata should contain duration_ms"
    assert metadata["duration_ms"] > 0, "Duration should be positive"

    # Validate inputs contain our test data
    inputs = trace.inputs
    if isinstance(inputs, str):
        import json

        inputs = json.loads(inputs)

    assert "record" in inputs, "Inputs should contain record"
    assert "prompt" in inputs, "Inputs should contain prompt"

    # Validate outputs contain Paris
    outputs = trace.outputs
    if isinstance(outputs, str):
        # Could be JSON string, try to parse
        try:
            import json

            outputs = json.loads(outputs)
        except:
            pass  # outputs is just a string

    output_str = str(outputs).lower()
    assert "paris" in output_str, f"Outputs should contain Paris, got: {outputs}"

    logger.info("✅ All trace validations passed")
    logger.info(f"Trace metadata: {metadata}")

    # Success!
    logger.info("=" * 80)
    logger.info("✅ END-TO-END TEST PASSED")
    logger.info("✅ LLM responded correctly")
    logger.info("✅ ExecutionTrace uploaded to BigQuery")
    logger.info("✅ Trace structure validated")
    logger.info(f"Trace ID: {trace.call_id}")
    logger.info("=" * 80)


@pytest.mark.anyio
async def test_trace_writer_initialization(real_bm):
    """Test that TraceWriter initializes correctly with storage config."""
    trace_writer = get_trace_writer()

    assert trace_writer is not None, "TraceWriter should be initialized"

    # Force initialization
    trace_writer._ensure_initialized()

    assert trace_writer._initialized, "TraceWriter should be marked as initialized"
    assert trace_writer.uploader is not None, (
        "TraceWriter should have an uploader configured. Check that conf/storage/traces.yaml exists and is valid."
    )

    logger.info("✅ TraceWriter initialized successfully")
    logger.info(f"Storage type: {type(trace_writer.uploader.storage).__name__}")


@pytest.mark.anyio
async def test_trace_writer_save(real_bm):
    """Test that TraceWriter can save a dummy trace."""
    trace_writer = get_trace_writer()
    trace_writer._ensure_initialized()

    # Create a dummy trace with unique test identifier
    test_start_time = datetime.datetime.now(datetime.timezone.utc)
    test_call_id = f"test-trace-writer-{test_start_time.timestamp()}"

    dummy_trace = ExecutionTrace(
        call_id=test_call_id,
        agent_info={"component_name": "test_trace_writer", "processor_stage": "save_test", "execution_type": "test"},
        inputs={"prompt": "Hello, world!"},
        outputs={"response": "Hi there!"},
        metadata={"test_key": "test_value", "test_type": "trace_writer_save"},
        parameters={"param1": "value1"},
        timestamp=test_start_time,
    )

    # Emit trace if trace writer is available
    logger.info(f"Adding trace with call_id: {test_call_id}")
    await trace_writer.add(dummy_trace)

    # Force flush to ensure it's uploaded
    logger.info("Flushing trace to BigQuery...")
    await trace_writer.flush()

    # Give BigQuery a moment to process the upload
    await asyncio.sleep(2)

    # Query BigQuery to verify trace was uploaded
    query = f"""
        SELECT
            call_id,
            agent_info,
            inputs,
            outputs,
            metadata,
            parameters,
            timestamp
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND call_id = '{test_call_id}'
        LIMIT 1
    """

    logger.debug(f"Querying BigQuery for uploaded trace:\n{query}")
    df = real_bm.run_query(query)

    # Validate trace was uploaded
    assert df.shape[0] == 1, (
        f"Expected exactly one trace in BigQuery with call_id={test_call_id}. "
        f"Query returned {df.shape[0]} rows. "
        f"Check that TraceWriter and AsyncDataUploader are working correctly."
    )

    # Get the uploaded trace
    trace = df.iloc[0]

    logger.info(f"✅ Trace uploaded successfully: {trace.call_id}")

    # Validate trace structure
    assert trace.call_id == test_call_id, f"Expected call_id={test_call_id}, got {trace.call_id}"

    # Validate agent_info
    agent_info = trace.agent_info
    if isinstance(agent_info, str):
        import json

        agent_info = json.loads(agent_info)

    assert agent_info.get("component_name") == "test_trace_writer", "Should have correct component_name"
    assert agent_info.get("processor_stage") == "save_test", "Should have correct processor_stage"

    # Validate metadata
    metadata = trace.metadata
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)

    assert metadata.get("test_key") == "test_value", "Metadata should contain test_key"
    assert metadata.get("test_type") == "trace_writer_save", "Metadata should contain test_type"

    # Validate inputs and outputs
    inputs = trace.inputs
    if isinstance(inputs, str):
        import json

        inputs = json.loads(inputs)

    assert inputs.get("prompt") == "Hello, world!", "Inputs should contain prompt"

    outputs = trace.outputs
    if isinstance(outputs, str):
        import json

        outputs = json.loads(outputs)

    assert outputs.get("response") == "Hi there!", "Outputs should contain response"

    logger.info("✅ All trace validations passed")
    logger.info(f"Trace metadata: {metadata}")
