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
import json
import re
from typing import Any

import pytest
from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.hashing import compute_record_hash, hash_dict
from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import load_template
from buttermilk.utils.trace_writer import get_trace_writer


class CapitalCityResponse(BaseModel):
    """Structured output for capital city question."""

    model_config = {"extra": "forbid"}  # Required for Azure OpenAI structured outputs

    city: str = Field(description="The capital city")
    country: str = Field(description="The country")
    explanation: str = Field(description="Brief explanation")


@pytest.fixture
def sample_record() -> BaseRecord:
    """Create a simple test record with all key fields populated."""
    return BaseRecord(
        text="What is the capital of France?",
        dataset_name="test_llmcore",
        split_type="test",
        metadata={"test": "llmcore_tracing"},
    )


def _validate_actual_model_name(logged_model: str, alias: str) -> None:
    """Verify that logged model name is the actual API model, not our alias.

    Args:
        logged_model: The model name logged in metadata
        alias: Our shorthand model name (e.g., "gemini-flash")

    Raises:
        AssertionError: If validation fails
    """
    # Extract key components from alias (e.g., "gemini-flash" -> ["gemini", "flash"])
    alias_components = []
    if "gemini" in alias.lower():
        alias_components.append("gemini")
    if "flash" in alias.lower():
        alias_components.append("flash")
    if "pro" in alias.lower():
        alias_components.append("pro")

    # Verify logged model contains these components
    for component in alias_components:
        assert component in logged_model.lower(), f"Model name '{logged_model}' should contain '{component}' from alias '{alias}'"

    # Verify it's NOT exactly our alias (proving we got actual API model name)
    assert logged_model != alias, f"Model should be actual API name (e.g., 'gemini-2.0-flash-exp'), not alias '{alias}'"


# Models that don't reliably support structured JSON output
# Note: Models with function_calling=true can use the fake tool fallback for structured output
# gpt-oss-safeguard models on HuggingFace don't support structured output or function calling
# claude45haiku has function_calling=false so can't use either approach
MODELS_WITHOUT_STRUCTURED_OUTPUT = {
    "gpt-oss-safeguard-20b",
    "gpt-oss-safeguard-120b",
}


@pytest.mark.anyio
async def test_llmcore_with_bigquery_trace(real_bm, sample_record: BaseRecord, real_model_name_expensive: str, llm_wrapper_type):
    """Test LLMCore processes request and uploads ExecutionTrace to BigQuery.

    This test:
    1. Creates an LLMCore instance with a cheap model
    2. Processes a simple request
    3. Verifies the LLM response is valid
    4. Forces trace writer to flush
    5. Queries BigQuery to verify trace was uploaded
    6. Validates trace structure and metadata
    """
    # Skip structured output test for models that don't support it
    if real_model_name_expensive in MODELS_WITHOUT_STRUCTURED_OUTPUT:
        pytest.skip(f"{real_model_name_expensive} doesn't reliably support structured JSON output")

    # Step 1: Create LLMCore with cheap model, simple template, and structured output
    llm_core = LLMCore(
        model=real_model_name_expensive,  # Use a real model from the fixture
        template="ra",  # Simple research assistant template
        output_model=CapitalCityResponse,  # Structured output for maximum E2E coverage
        fail_on_unfilled_parameters=False,
    )

    # Step 2: Process the request
    # Use model-specific processor_stage to isolate parallel test runs in BigQuery queries
    processor_stage = f"test_stage_{real_model_name_expensive}"
    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing LLMCore request...")
    results = []
    async for result in llm_core.process(
        record=sample_record,
        processor_stage=processor_stage,
        component_name="test_llmcore",
        prompt="What is the capital of France?",
    ):
        results.append(result)

    # Step 3: Validate LLM response - should be structured CapitalCityResponse
    assert len(results) == 1, "Expected exactly one result from process()"
    result = results[0]

    assert isinstance(result, BaseRecord), "Result should be a BaseRecord"
    assert hasattr(result, "output"), "Result should have output field"
    assert result.output is not None, "Output should not be None"

    # Validate structured output - should be CapitalCityResponse instance
    assert isinstance(result.output, CapitalCityResponse), (
        f"Output should be CapitalCityResponse, got {type(result.output).__name__}: {result.output}"
    )

    # Validate the structured fields
    assert result.output.city.lower() == "paris", f"Expected city='Paris', got '{result.output.city}'"
    assert result.output.country.lower() == "france", f"Expected country='France', got '{result.output.country}'"
    assert len(result.output.explanation) > 0, "Explanation should not be empty"

    logger.info(f"✅ LLM Structured Response valid: city={result.output.city}, country={result.output.country}")

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
    # Use model-specific processor_stage to isolate parallel test runs
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
            AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
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
        agent_info = json.loads(agent_info)

    assert agent_info.get("component_name") == "test_llmcore", "Should have correct component_name"
    assert agent_info.get("execution_type") == "llm_processing", "Should be llm_processing type"
    assert agent_info.get("processor_stage") == processor_stage, "Should have correct processor_stage"

    # Validate metadata contains LLM info
    metadata = trace.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    assert "model" in metadata, "Metadata should contain model"

    # Verify model name is actual API model, not our alias
    _validate_actual_model_name(metadata["model"], real_model_name_expensive)

    assert "duration_ms" in metadata, "Metadata should contain duration_ms"
    assert metadata["duration_ms"] > 0, "Duration should be positive"

    # Validate inputs contain our test data
    inputs = trace.inputs
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    # Validate template_vars contains the prompt and record data
    template_vars = inputs.get("template_vars", {})
    if isinstance(template_vars, str):
        template_vars = json.loads(template_vars)

    assert "prompt" in template_vars, f"template_vars should contain prompt field. template_vars keys: {template_vars.keys()}"

    # Record data is now stored in template_vars (flattened from record)
    # Essential fields: record_id, dataset_name, split_type
    assert "record_id" in template_vars, f"template_vars should have record_id field. Keys: {template_vars.keys()}"
    assert template_vars["record_id"] is not None, "template_vars.record_id should not be None"
    assert len(template_vars["record_id"]) > 0, "template_vars.record_id should not be empty"

    assert "dataset_name" in template_vars, f"template_vars should have dataset_name field. Keys: {template_vars.keys()}"
    assert "split_type" in template_vars, f"template_vars should have split_type field. Keys: {template_vars.keys()}"
    assert template_vars["dataset_name"] == "test_llmcore", "template_vars should preserve dataset_name value"
    assert template_vars["split_type"] == "test", "template_vars should preserve split_type value"

    logger.info(
        f"✅ Record structure validated: record_id={template_vars['record_id']}, "
        f"dataset_name={template_vars['dataset_name']}, "
        f"split_type={template_vars['split_type']}"
    )

    # Validate outputs contain structured response with Paris
    outputs = trace.outputs
    if isinstance(outputs, str):
        # Could be JSON string, try to parse
        try:
            outputs = json.loads(outputs)
        except (json.JSONDecodeError, TypeError):
            pass  # outputs is just a string

    # Outputs should be structured CapitalCityResponse data
    if isinstance(outputs, dict):
        assert "city" in outputs, f"Structured output should have 'city' field, got: {outputs}"
        assert outputs["city"].lower() == "paris", f"Structured output city should be 'Paris', got: {outputs['city']}"
    else:
        # Fallback to string check if not dict
        output_str = str(outputs).lower()
        assert "paris" in output_str, f"Outputs should contain Paris, got: {outputs}"

    # Validate messages field contains the exact API input/output
    # Query for messages field from BigQuery
    query_with_messages = f"""
        SELECT
            call_id,
            messages
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE call_id = '{trace.call_id}'
        LIMIT 1
    """
    df_messages = real_bm.run_query(query_with_messages)

    assert df_messages.shape[0] == 1, "Should retrieve the trace with messages"
    messages = df_messages.iloc[0].messages

    # FAIL-FAST: Messages must already be a list, not a string or other type
    assert messages is not None, "Messages field should not be None"
    assert isinstance(messages, list), f"Messages should be a list, got {type(messages).__name__}. Data should be stored in correct format."
    assert len(messages) > 0, "Messages list should not be empty"

    # Parse JSON strings into dicts (BigQuery stores messages as JSON strings)
    parsed_messages = []
    for i, msg in enumerate(messages):
        assert isinstance(msg, str), (
            f"Message {i} should be a JSON string, got {type(msg).__name__}. Messages are stored as JSON strings in BigQuery."
        )
        try:
            parsed_msg = json.loads(msg)
            parsed_messages.append(parsed_msg)
        except json.JSONDecodeError as e:
            pytest.fail(f"Message {i} is not valid JSON: {e}. Message content: {msg[:100]}...")

    # FAIL-FAST: Each parsed message must be a dict with expected structure
    for i, msg in enumerate(parsed_messages):
        assert isinstance(msg, dict), f"Parsed message {i} should be a dict, got {type(msg).__name__}."
        assert "role" in msg or "type" in msg, f"Message {i} missing role/type field. Expected message structure with role or type."
        assert "content" in msg, f"Message {i} missing content field. Expected message structure with content."

    # Validate messages contain both user/system input and assistant response
    roles = [msg.get("role", msg.get("type", "")) for msg in parsed_messages]

    # Check for user or system message (input)
    has_input = any("user" in str(role).lower() or "system" in str(role).lower() for role in roles)
    assert has_input, f"Messages should contain user or system message (input), got roles: {roles}"

    # Check for assistant message (response)
    has_assistant = any("assistant" in str(role).lower() for role in roles)
    assert has_assistant, f"Messages should contain assistant message (response), got roles: {roles}"

    # Validate message content includes expected terms
    all_content = " ".join(msg.get("content", "") for msg in parsed_messages)
    assert "capital" in all_content.lower() or "france" in all_content.lower(), "Messages should contain the original prompt about France's capital"
    assert "paris" in all_content.lower(), "Messages should contain the LLM's response mentioning Paris"

    # VERIFY: Template variables are filled - first message should have substantial content
    # The "ra" template system prompt is typically 200+ characters when filled
    first_content = parsed_messages[0].get("content", "")
    assert len(first_content) > 100, (
        f"First message (system prompt) content too short ({len(first_content)} chars). "
        f"Template variables may not be filled properly. "
        f"Content preview: {first_content[:200]}..."
    )

    # VERIFY: No unfilled Jinja2 template variables in any message
    # Unfilled variables look like: {{variable_name}} or {{ variable_name }}
    unfilled_pattern = re.compile(r"\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\}")

    for i, msg in enumerate(parsed_messages):
        msg_content = msg.get("content", "")
        unfilled_vars = unfilled_pattern.findall(msg_content)
        assert not unfilled_vars, f"Unfilled template variables found in message[{i}]: {unfilled_vars}. Content preview: {msg_content[:300]}..."

    logger.info(f"✅ Template variables validated: no unfilled variables, system prompt {len(first_content)} chars")

    # Validate message ordering: template system prompt in messages[0], user content in messages[1]
    # The "ra" template has system prompt: "You are a careful research assistant..."
    assert len(parsed_messages) >= 3, f"Expected at least 3 messages (system + user + answer), got {len(parsed_messages)}"

    # messages[0] should contain the template system prompt
    first_msg = parsed_messages[0]
    first_role = first_msg.get("role", first_msg.get("type", "")).lower()
    first_content = first_msg.get("content", "")
    assert "system" in first_role, f"First message should be system role from template, got role: {first_role}"
    assert "research assistant" in first_content.lower(), (
        f"First message should contain template system prompt 'research assistant', got: {first_content[:200]}..."
    )

    # messages[1] should contain the user's prompt about France's capital
    second_msg = parsed_messages[1]
    second_role = second_msg.get("role", second_msg.get("type", "")).lower()
    second_content = second_msg.get("content", "")
    assert "user" in second_role, f"Second message should be user role with prompt, got role: {second_role}"
    assert "capital" in second_content.lower() or "france" in second_content.lower(), (
        f"Second message should contain user prompt about France's capital, got: {second_content[:200]}..."
    )

    logger.info(f"✅ Messages field validated: {len(messages)} messages with roles {roles}")

    # Validate parameters field contains model hyperparameters
    # Query for parameters field from BigQuery
    query_with_parameters = f"""
        SELECT
            call_id,
            parameters
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE call_id = '{trace.call_id}'
        LIMIT 1
    """
    df_parameters = real_bm.run_query(query_with_parameters)

    assert df_parameters.shape[0] == 1, "Should retrieve the trace with parameters"
    parameters = df_parameters.iloc[0].parameters

    # FAIL-FAST: Parameters must exist and be parseable
    assert parameters is not None, "Parameters field should not be None"

    # Parse if JSON string
    if isinstance(parameters, str):
        parameters = json.loads(parameters)

    assert isinstance(parameters, dict), f"Parameters should be a dict, got {type(parameters).__name__}"

    # Validate core LLMCore parameters are present
    assert "model" in parameters, f"Parameters should contain 'model'. Got keys: {parameters.keys()}"
    assert "template" in parameters, f"Parameters should contain 'template'. Got keys: {parameters.keys()}"

    # Validate model configs are captured (temperature, api_version, etc. from models.json)
    # The configs dict is stored in LLMConfig.configs, NOT in ModelParameters
    if real_model_name_expensive in real_bm.llms.connections:
        llm_config = real_bm.llms.connections[real_model_name_expensive]
        model_configs = llm_config.configs if llm_config.configs else {}

        if "temperature" in model_configs:
            assert "temperature" in parameters, (
                f"Parameters should contain 'temperature' from model configs. Model configs: {model_configs}, got parameters: {parameters}"
            )
            assert parameters["temperature"] == model_configs["temperature"], (
                f"Temperature should match model config: expected {model_configs['temperature']}, got {parameters.get('temperature')}"
            )
            logger.info(f"✅ Hyperparameter 'temperature' logged correctly: {parameters['temperature']}")

        if "api_version" in model_configs:
            assert "api_version" in parameters, (
                f"Parameters should contain 'api_version' from model configs. Model configs: {model_configs}, got parameters: {parameters}"
            )
            logger.info(f"✅ Config 'api_version' logged correctly: {parameters['api_version']}")

    logger.info(f"✅ Parameters field validated: {list(parameters.keys())}")

    # ==========================================================================
    # RECORD FIELD VALIDATION: Verify trace.record is filled from BigQuery
    # ==========================================================================

    # Query for record field from BigQuery
    query_with_record = f"""
        SELECT
            call_id,
            record
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE call_id = '{trace.call_id}'
        LIMIT 1
    """
    df_record = real_bm.run_query(query_with_record)

    assert df_record.shape[0] == 1, "Should retrieve the trace with record field"
    trace_record = df_record.iloc[0].record

    # FAIL-FAST: trace.record MUST be populated with record context
    # Record data should be in the dedicated record field, not just buried in template_vars
    assert trace_record is not None, (
        "trace.record field should not be None - record context must be preserved. "
        f"LLMCore must populate ExecutionTrace.record when processing a record. "
        f"Record was passed (record_id={sample_record.record_id}) but trace.record is None."
    )

    # Parse if JSON string
    if isinstance(trace_record, str):
        trace_record = json.loads(trace_record)

    assert isinstance(trace_record, dict), f"trace.record should be a dict, got {type(trace_record).__name__}"

    # Validate essential record fields are present
    assert "record_id" in trace_record, f"trace.record should contain 'record_id'. Got keys: {trace_record.keys()}"
    assert trace_record["record_id"] is not None, "trace.record.record_id should not be None"
    assert len(trace_record["record_id"]) > 0, "trace.record.record_id should not be empty"

    # Validate record matches the input sample_record
    assert trace_record["record_id"] == sample_record.record_id, (
        f"trace.record.record_id should match input record.\nTrace: {trace_record['record_id']}\nExpected: {sample_record.record_id}"
    )

    # Validate dataset context is preserved
    if "dataset_name" in trace_record:
        assert trace_record["dataset_name"] == sample_record.dataset_name, "trace.record.dataset_name should match input record"

    logger.info(f"✅ trace.record validated: record_id={trace_record['record_id']}, keys={list(trace_record.keys())}")

    # ==========================================================================
    # HASH VALIDATION: Verify hashes exist and match recomputed values
    # ==========================================================================

    # 1. Validate template_hash exists in metadata and matches recomputed hash
    assert "template" in metadata, f"Metadata should contain 'template' with template_hash. Got keys: {metadata.keys()}"
    template_metadata = metadata["template"]
    if isinstance(template_metadata, str):
        template_metadata = json.loads(template_metadata)

    assert "template_hash" in template_metadata, f"Template metadata should contain 'template_hash'. Got: {template_metadata.keys()}"
    logged_template_hash = template_metadata["template_hash"]
    assert logged_template_hash is not None, "template_hash should not be None"
    assert len(logged_template_hash) == 64, f"template_hash should be 64-char SHA256, got {len(logged_template_hash)} chars: {logged_template_hash}"

    # Recompute template hash and verify it matches
    # The test uses template="ra"
    _, _, expected_template_hash = load_template(
        template="ra",
        parameters={},
        untrusted_inputs={},
    )
    assert logged_template_hash == expected_template_hash, (
        f"Logged template_hash should match recomputed hash.\nLogged:   {logged_template_hash}\nExpected: {expected_template_hash}"
    )
    logger.info(f"✅ template_hash validated: {logged_template_hash[:16]}...")

    # 2. Validate record exists in template_vars and has record_id for hash verification
    # Record data is flattened into template_vars
    template_vars_for_hash = inputs.get("template_vars", {})
    if isinstance(template_vars_for_hash, str):
        template_vars_for_hash = json.loads(template_vars_for_hash)

    assert "record_id" in template_vars_for_hash, f"template_vars in trace should have record_id. Got keys: {template_vars_for_hash.keys()}"
    # Verify the record_id matches our sample_record
    assert template_vars_for_hash["record_id"] == sample_record.record_id, (
        f"Record ID in trace should match input record.\nTrace:    {template_vars_for_hash['record_id']}\nExpected: {sample_record.record_id}"
    )
    logger.info(f"✅ record_id validated: {template_vars_for_hash['record_id']}")

    # Validate record_hash exists in metadata and matches recomputed hash
    assert "record" in metadata, f"Metadata should contain 'record' with record_hash. Got keys: {metadata.keys()}"
    record_metadata = metadata["record"]
    if isinstance(record_metadata, str):
        record_metadata = json.loads(record_metadata)

    assert "record_hash" in record_metadata, f"Record metadata should contain 'record_hash'. Got: {record_metadata.keys()}"
    logged_record_hash = record_metadata["record_hash"]
    assert logged_record_hash is not None, "record_hash should not be None"
    assert len(logged_record_hash) == 64, f"record_hash should be 64-char SHA256, got {len(logged_record_hash)} chars: {logged_record_hash}"

    # Recompute record hash and verify it matches
    expected_record_hash = compute_record_hash(sample_record.as_markdown())
    assert logged_record_hash == expected_record_hash, (
        f"Logged record_hash should match recomputed hash.\nLogged:   {logged_record_hash}\nExpected: {expected_record_hash}"
    )
    logger.info(f"✅ record_hash validated: {logged_record_hash[:16]}...")

    # 3. Validate config hash can be computed from parameters
    # The parameters field contains the LLMCore config that should be hashable
    config_hash = hash_dict(parameters)
    assert len(config_hash) == 64, f"config_hash should be 64-char SHA256, got {len(config_hash)} chars"
    logger.info(f"✅ config_hash computed from parameters: {config_hash[:16]}...")

    logger.info("✅ All hash validations passed")
    logger.info(f"Trace metadata: {metadata}")

    # Success!
    logger.info("=" * 80)
    logger.info("✅ END-TO-END TEST PASSED")
    logger.info("✅ LLM responded correctly")
    logger.info("✅ ExecutionTrace uploaded to BigQuery")
    logger.info("✅ Trace structure validated")
    logger.info(f"Trace ID: {trace.call_id}")
    logger.info("=" * 80)


# =============================================================================
# CRITERIA VARIANTS TEST - Verify template filling with multiple criteria
# =============================================================================

# Two distinct criteria for testing - each has unique identifying content
# NOTE: identifying_text must be in the RENDERED output (not Jinja2 comments)
CRITERIA_VARIANTS = [
    ("cte", "US Transgender Survey"),  # criteria name, identifying text in body (not header comment)
    ("tja", "Trans Journalists Association"),  # appears in body, not just header comment
]


@pytest.mark.anyio
@pytest.mark.endtoend
async def test_template_filling_with_criteria_variants(real_bm, sample_record: BaseRecord):
    """Test that template variables are correctly filled for each criteria variant.

    This test validates:
    1. Two different criteria (cte, tja) are correctly included in templates
    2. Each criteria variant produces messages with the correct criteria content
    3. No unfilled {{criteria}} template variables remain
    4. Both traces are uploaded to BigQuery with correct template content

    Bug being tested: Templates not filled properly when using variants.
    """
    # Use a cheap model for this test
    model = "gemini-flash"
    test_start_time = datetime.datetime.now(datetime.timezone.utc)
    traces_by_criteria = {}

    for criteria_name, identifying_text in CRITERIA_VARIANTS:
        logger.info(f"Testing criteria variant: {criteria_name}")

        # Create LLMCore with judge template that includes criteria
        llm_core = LLMCore(
            model=model,
            template="judge",  # Template that uses {{ render_or_include(criteria) }}
            fail_on_unfilled_parameters=True,  # FAIL if criteria not filled
        )

        # Process with the specific criteria
        processor_stage = f"test_criteria_variant_{criteria_name}"

        results = []
        async for result in llm_core.process(
            record=sample_record,
            processor_stage=processor_stage,
            component_name="test_criteria_variants",
            criteria=criteria_name,  # Pass criteria as template variable
        ):
            results.append(result)

        assert len(results) == 1, f"Expected one result for criteria {criteria_name}"
        traces_by_criteria[criteria_name] = (processor_stage, identifying_text)

    # Flush traces to BigQuery
    logger.info("Flushing traces to BigQuery...")
    await asyncio.sleep(1)  # Allow buffer to fill
    trace_writer = get_trace_writer()
    await trace_writer.flush()

    # Wait for BigQuery to process
    await asyncio.sleep(2)

    # Query BigQuery for both traces and validate
    for criteria_name, (processor_stage, identifying_text) in traces_by_criteria.items():
        query = f"""
            SELECT
                call_id,
                messages,
                inputs
            FROM `{real_bm.bq.project}.testing.traces`
            WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
                AND JSON_VALUE(agent_info, '$.component_name') = 'test_criteria_variants'
                AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
            ORDER BY timestamp DESC
            LIMIT 1
        """

        logger.debug(f"Querying BigQuery for {criteria_name} trace...")
        df = real_bm.run_query(query)

        assert df.shape[0] == 1, f"Expected exactly one trace for criteria {criteria_name}. Got {df.shape[0]} rows."

        trace_row = df.iloc[0]

        # Parse messages
        messages_raw = trace_row.messages
        if isinstance(messages_raw, str):
            messages = json.loads(messages_raw)
        else:
            messages = messages_raw

        assert messages is not None, f"Messages should not be None for {criteria_name}"
        assert len(messages) > 0, f"Should have at least one message for {criteria_name}"

        # Combine all message content
        all_content = ""
        for msg in messages:
            if isinstance(msg, str):
                msg = json.loads(msg)
            content = msg.get("content", "")
            all_content += content + "\n"

        # CRITICAL: Verify the correct criteria was included in the template
        assert identifying_text in all_content, (
            f"Criteria '{criteria_name}' not properly filled in template!\n"
            f"Expected to find: '{identifying_text}'\n"
            f"Content preview (first 1000 chars):\n{all_content[:1000]}..."
        )

        # CRITICAL: Verify no unfilled template variables
        unfilled_pattern = re.compile(r"\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\}")
        unfilled_vars = unfilled_pattern.findall(all_content)
        assert not unfilled_vars, (
            f"Unfilled template variables found for criteria {criteria_name}: {unfilled_vars}\nContent preview: {all_content[:500]}..."
        )

        # Verify criteria is in template_vars in inputs
        inputs = trace_row.inputs
        if isinstance(inputs, str):
            inputs = json.loads(inputs)

        template_vars = inputs.get("template_vars", {})
        if isinstance(template_vars, str):
            template_vars = json.loads(template_vars)

        assert "criteria" in template_vars, f"template_vars should contain 'criteria' for {criteria_name}. Got keys: {template_vars.keys()}"
        assert template_vars["criteria"] == criteria_name, (
            f"template_vars.criteria should be '{criteria_name}', got '{template_vars.get('criteria')}'"
        )

        logger.info(f"✅ Criteria '{criteria_name}' correctly filled in template")
        logger.info(f"   - Identifying text found: '{identifying_text[:50]}...'")
        logger.info("   - No unfilled variables")
        logger.info(f"   - template_vars.criteria = '{criteria_name}'")

    logger.info("=" * 80)
    logger.info("✅ CRITERIA VARIANTS TEST PASSED")
    logger.info(f"✅ Tested {len(CRITERIA_VARIANTS)} criteria variants")
    logger.info("✅ All templates correctly filled with criteria content")
    logger.info("=" * 80)


@pytest.mark.anyio
async def test_trace_writer_initialization(real_bm, llm_wrapper_type):
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
async def test_trace_writer_save(real_bm, llm_wrapper_type):
    """Test that TraceWriter can save a dummy trace."""
    trace_writer = get_trace_writer()
    trace_writer._ensure_initialized()

    # Create a dummy trace with unique test identifier
    test_start_time = datetime.datetime.now(datetime.timezone.utc)
    test_call_id = f"test-trace-writer-{test_start_time.timestamp()}"

    dummy_trace = ExecutionTrace(
        call_id=test_call_id,
        session_id=f"test-session-{test_start_time.timestamp()}",
        agent_info={
            "component_name": "test_trace_writer",
            "processor_stage": "save_test",
            "execution_type": "test",
        },
        inputs={"prompt": "Hello, world!"},
        outputs={"response": "Hi there!"},
        metadata={"test_key": "test_value", "test_type": "trace_writer_save"},
        parameters={"param1": "value1"},
    )

    # Emit trace if trace writer is available
    logger.info(f"Adding trace with call_id: {test_call_id}")
    await trace_writer.add(dummy_trace)

    # Give a moment for trace to be queued
    await asyncio.sleep(2)

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
        agent_info = json.loads(agent_info)

    assert agent_info.get("component_name") == "test_trace_writer", "Should have correct component_name"
    assert agent_info.get("processor_stage") == "save_test", "Should have correct processor_stage"

    # Validate metadata
    metadata = trace.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    assert metadata.get("test_key") == "test_value", "Metadata should contain test_key"
    assert metadata.get("test_type") == "trace_writer_save", "Metadata should contain test_type"

    # Validate inputs and outputs
    inputs = trace.inputs
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    assert inputs.get("prompt") == "Hello, world!", "Inputs should contain prompt"

    outputs = trace.outputs
    if isinstance(outputs, str):
        outputs = json.loads(outputs)

    assert outputs.get("response") == "Hi there!", "Outputs should contain response"

    logger.info("✅ All trace validations passed")
    logger.info(f"Trace metadata: {metadata}")


@pytest.mark.anyio
async def test_record_hash_stored_in_single_location(real_bm, sample_record: BaseRecord, real_model_name_expensive: str, llm_wrapper_type):
    """Test that record_hash appears in exactly ONE location in the serialized trace.

    This test validates data integrity by ensuring record_hash is stored in a
    single, predictable location rather than duplicated across multiple fields.

    Currently, record_hash appears in multiple locations:
    - metadata.record.record_hash
    - inputs.record.record_hash (if record serialized in inputs)
    - metadata.resolved_inputs.record.record_hash
    - metadata.resolved_inputs.template_vars.record[*].record_hash

    This test should FAIL until record_hash storage is fixed to use a single location.
    """
    # Skip structured output test for models that don't support it
    if real_model_name_expensive in MODELS_WITHOUT_STRUCTURED_OUTPUT:
        pytest.skip(f"{real_model_name_expensive} doesn't reliably support structured JSON output")

    # Step 1: Create LLMCore and process a request
    llm_core = LLMCore(
        model=real_model_name_expensive,
        template="ra",  # Simple research assistant template
        output_model=CapitalCityResponse,
        fail_on_unfilled_parameters=False,
    )

    # Process the request
    processor_stage = f"test_hash_location_{real_model_name_expensive}"
    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing LLMCore request for hash location test...")
    results = []
    async for result in llm_core.process(
        record=sample_record,
        processor_stage=processor_stage,
        component_name="test_hash_location",
        prompt="What is the capital of France?",
    ):
        results.append(result)

    assert len(results) == 1, "Expected exactly one result from process()"
    result = results[0]

    # Step 2: Force trace flush to BigQuery
    trace_writer = get_trace_writer()
    assert trace_writer is not None, "TraceWriter should be initialized"

    await asyncio.sleep(1)
    logger.info("Flushing traces to BigQuery...")
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Step 3: Query BigQuery for the trace
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
            AND JSON_VALUE(agent_info, '$.component_name') = 'test_hash_location'
            AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
        ORDER BY timestamp DESC
        LIMIT 1
    """

    logger.debug(f"Querying BigQuery for uploaded trace:\n{query}")
    df = real_bm.run_query(query)

    assert df.shape[0] > 0, f"Expected at least one trace in BigQuery. Query returned {df.shape[0]} rows."

    trace = df.iloc[0]
    logger.info(f"Retrieved trace call_id: {trace.call_id}")

    # Step 4: Serialize the trace to a dict (simulating what BigQuery stores)
    # Parse JSON fields from BigQuery
    trace_dict = {
        "call_id": trace.call_id,
        "agent_info": json.loads(trace.agent_info) if isinstance(trace.agent_info, str) else trace.agent_info,
        "inputs": json.loads(trace.inputs) if isinstance(trace.inputs, str) else trace.inputs,
        "outputs": json.loads(trace.outputs) if isinstance(trace.outputs, str) else trace.outputs,
        "metadata": json.loads(trace.metadata) if isinstance(trace.metadata, str) else trace.metadata,
        "parameters": json.loads(trace.parameters) if isinstance(trace.parameters, str) else trace.parameters,
    }

    # Step 5: Walk the entire structure and find ALL occurrences of record_hash
    def find_record_hash_paths(obj: Any, path: str = "root") -> list[str]:
        """Recursively find all paths where record_hash appears.

        Args:
            obj: The object to search (dict, list, or primitive)
            path: Current path in dot notation

        Returns:
            List of paths where record_hash was found
        """
        paths = []

        if isinstance(obj, dict):
            # Check if this dict has a record_hash key
            if "record_hash" in obj:
                paths.append(f"{path}.record_hash")

            # Recursively search all values
            for key, value in obj.items():
                child_paths = find_record_hash_paths(value, f"{path}.{key}")
                paths.extend(child_paths)

        elif isinstance(obj, list):
            # Recursively search all list items
            for idx, item in enumerate(obj):
                child_paths = find_record_hash_paths(item, f"{path}[{idx}]")
                paths.extend(child_paths)

        # Primitives (str, int, bool, None) don't contain nested record_hash
        return paths

    all_record_hash_paths = find_record_hash_paths(trace_dict)

    # Step 6: Assert record_hash appears exactly ONCE
    logger.info(f"Found record_hash at {len(all_record_hash_paths)} locations:")
    for path in all_record_hash_paths:
        logger.info(f"  - {path}")

    assert len(all_record_hash_paths) == 1, (
        f"record_hash should appear in EXACTLY ONE location, but found {len(all_record_hash_paths)} locations:\n"
        + "\n".join(f"  - {path}" for path in all_record_hash_paths)
        + "\n\nThis indicates record_hash is being duplicated across multiple fields, "
        + "which creates data integrity issues and confusion about the source of truth."
    )

    # Validate the single location is the expected one (metadata.record.record_hash)
    expected_path = "root.metadata.record.record_hash"
    assert all_record_hash_paths[0] == expected_path, (
        f"record_hash should be stored at '{expected_path}', but found it at '{all_record_hash_paths[0]}'"
    )

    logger.info(f"✅ record_hash appears in exactly ONE location: {all_record_hash_paths[0]}")


@pytest.mark.anyio
async def test_template_hash_stored_in_single_location(real_bm, sample_record: BaseRecord, real_model_name_expensive: str, llm_wrapper_type):
    """Test that template_hash appears in exactly ONE location in the serialized trace.

    This test validates data integrity by ensuring template_hash is stored in a
    single, predictable location rather than duplicated across multiple fields.

    Currently, template_hash appears in multiple locations:
    - metadata.template.template_hash (expected location)
    - agent_info.template_hash (duplicate, should be removed)

    This test should FAIL until template_hash storage is fixed to use a single location.
    """
    # Skip structured output test for models that don't support it
    if real_model_name_expensive in MODELS_WITHOUT_STRUCTURED_OUTPUT:
        pytest.skip(f"{real_model_name_expensive} doesn't reliably support structured JSON output")

    # Step 1: Create LLMCore and process a request
    llm_core = LLMCore(
        model=real_model_name_expensive,
        template="ra",  # Simple research assistant template
        output_model=CapitalCityResponse,
        fail_on_unfilled_parameters=False,
    )

    # Process the request
    processor_stage = f"test_template_hash_location_{real_model_name_expensive}"
    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing LLMCore request for template_hash location test...")
    results = []
    async for result in llm_core.process(
        record=sample_record,
        processor_stage=processor_stage,
        component_name="test_template_hash_location",
        prompt="What is the capital of France?",
    ):
        results.append(result)

    assert len(results) == 1, "Expected exactly one result from process()"
    result = results[0]

    # Step 2: Force trace flush to BigQuery
    trace_writer = get_trace_writer()
    assert trace_writer is not None, "TraceWriter should be initialized"

    await asyncio.sleep(1)
    logger.info("Flushing traces to BigQuery...")
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Step 3: Query BigQuery for the trace
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
            AND JSON_VALUE(agent_info, '$.component_name') = 'test_template_hash_location'
            AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
        ORDER BY timestamp DESC
        LIMIT 1
    """

    logger.debug(f"Querying BigQuery for uploaded trace:\n{query}")
    df = real_bm.run_query(query)

    assert df.shape[0] > 0, f"Expected at least one trace in BigQuery. Query returned {df.shape[0]} rows."

    trace = df.iloc[0]
    logger.info(f"Retrieved trace call_id: {trace.call_id}")

    # Step 4: Serialize the trace to a dict (simulating what BigQuery stores)
    # Parse JSON fields from BigQuery
    trace_dict = {
        "call_id": trace.call_id,
        "agent_info": json.loads(trace.agent_info) if isinstance(trace.agent_info, str) else trace.agent_info,
        "inputs": json.loads(trace.inputs) if isinstance(trace.inputs, str) else trace.inputs,
        "outputs": json.loads(trace.outputs) if isinstance(trace.outputs, str) else trace.outputs,
        "metadata": json.loads(trace.metadata) if isinstance(trace.metadata, str) else trace.metadata,
        "parameters": json.loads(trace.parameters) if isinstance(trace.parameters, str) else trace.parameters,
    }

    # Step 5: Walk the entire structure and find ALL occurrences of template_hash
    def find_template_hash_paths(obj: Any, path: str = "root") -> list[str]:
        """Recursively find all paths where template_hash appears.

        Args:
            obj: The object to search (dict, list, or primitive)
            path: Current path in dot notation

        Returns:
            List of paths where template_hash was found
        """
        paths = []

        if isinstance(obj, dict):
            # Check if this dict has a template_hash key
            if "template_hash" in obj:
                paths.append(f"{path}.template_hash")

            # Recursively search all values
            for key, value in obj.items():
                child_paths = find_template_hash_paths(value, f"{path}.{key}")
                paths.extend(child_paths)

        elif isinstance(obj, list):
            # Recursively search all list items
            for idx, item in enumerate(obj):
                child_paths = find_template_hash_paths(item, f"{path}[{idx}]")
                paths.extend(child_paths)

        # Primitives (str, int, bool, None) don't contain nested template_hash
        return paths

    all_template_hash_paths = find_template_hash_paths(trace_dict)

    # Step 6: Assert template_hash appears exactly ONCE
    logger.info(f"Found template_hash at {len(all_template_hash_paths)} locations:")
    for path in all_template_hash_paths:
        logger.info(f"  - {path}")

    assert len(all_template_hash_paths) == 1, (
        f"template_hash should appear in EXACTLY ONE location, but found {len(all_template_hash_paths)} locations:\n"
        + "\n".join(f"  - {path}" for path in all_template_hash_paths)
        + "\n\nThis indicates template_hash is being duplicated across multiple fields, "
        + "which creates data integrity issues and confusion about the source of truth."
    )

    # Validate the single location is the expected one (metadata.template.template_hash)
    expected_path = "root.metadata.template.template_hash"
    assert all_template_hash_paths[0] == expected_path, (
        f"template_hash should be stored at '{expected_path}', but found it at '{all_template_hash_paths[0]}'"
    )

    logger.info(f"✅ template_hash appears in exactly ONE location: {all_template_hash_paths[0]}")


@pytest.mark.anyio
async def test_warning_raised_on_record_mismatch(real_bm, caplog):
    """Test that a warning is raised when record data mismatches template_vars.

    This validates issue #305 acceptance criterion:
    If inputs.record is filled AND inputs.template_vars.record != inputs.record,
    raise a warning about data inconsistency.
    """
    import logging

    # Create record with one text
    record = BaseRecord(text="Content A", dataset_name="test", record_id="test-1")

    # Create template_vars with DIFFERENT text
    template_vars = {"text": "Content B"}

    llm_core = LLMCore(model="gemini-flash-lite", template="ra", fail_on_unfilled_parameters=False)

    with caplog.at_level(logging.WARNING):
        results = []
        async for result in llm_core.process(
            record=record,
            template_vars=template_vars,  # Different from record!
            processor_stage="test_mismatch_warning",
            component_name="test_mismatch",
        ):
            results.append(result)

    # Should have logged a warning about mismatch
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("mismatch" in msg.lower() for msg in warning_messages), f"Expected warning about record mismatch, got: {warning_messages}"


@pytest.mark.endtoend
@pytest.mark.anyio
async def test_no_duplicate_record_in_resolved_inputs(real_bm, sample_record: BaseRecord, real_model_name_expensive: str, llm_wrapper_type):
    """Test that record data is NOT duplicated in both inputs.record and inputs.template_vars.

    When template_vars=None is passed to LLMCore, it derives template_vars from record.
    This test verifies that the resolved_inputs does NOT duplicate record data between:
    - inputs.template_vars (containing record fields)
    - inputs.record (also containing full record data)

    Expected failure: Currently both contain the full record data (duplication exists).

    Acceptance criterion: If template_vars contains record fields (text, dataset_name, etc.),
    then inputs.record should NOT contain the same full content.
    """
    # Skip structured output test for models that don't support it
    if real_model_name_expensive in MODELS_WITHOUT_STRUCTURED_OUTPUT:
        pytest.skip(f"{real_model_name_expensive} doesn't reliably support structured JSON output")

    # Step 1: Create LLMCore with template_vars=None (will be derived from record)
    llm_core = LLMCore(
        model=real_model_name_expensive,
        template="ra",  # Simple research assistant template
        output_model=CapitalCityResponse,
        fail_on_unfilled_parameters=False,
    )

    # Process the request WITHOUT passing template_vars explicitly
    # This forces LLMCore to derive template_vars from record
    processor_stage = f"test_no_duplicate_record_{real_model_name_expensive}"
    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Processing LLMCore request with template_vars=None (derived from record)...")
    results = []
    async for result in llm_core.process(
        record=sample_record,
        processor_stage=processor_stage,
        component_name="test_no_duplicate_record",
        prompt="What is the capital of France?",
        # Note: NOT passing template_vars - it will be derived from record
    ):
        results.append(result)

    assert len(results) == 1, "Expected exactly one result from process()"
    result = results[0]

    # Step 2: Force trace flush to BigQuery
    trace_writer = get_trace_writer()
    assert trace_writer is not None, "TraceWriter should be initialized"

    await asyncio.sleep(1)
    logger.info("Flushing traces to BigQuery...")
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Step 3: Query BigQuery for the trace
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
            AND JSON_VALUE(agent_info, '$.component_name') = 'test_no_duplicate_record'
            AND JSON_VALUE(agent_info, '$.processor_stage') = '{processor_stage}'
        ORDER BY timestamp DESC
        LIMIT 1
    """

    logger.debug(f"Querying BigQuery for uploaded trace:\n{query}")
    df = real_bm.run_query(query)

    assert df.shape[0] > 0, f"Expected at least one trace in BigQuery. Query returned {df.shape[0]} rows."

    trace = df.iloc[0]
    logger.info(f"Retrieved trace call_id: {trace.call_id}")

    # Step 4: Parse the trace inputs
    inputs = trace.inputs
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    # Step 5: Check for duplication
    template_vars = inputs.get("template_vars", {})
    record_in_inputs = inputs.get("record")

    if isinstance(template_vars, str):
        template_vars = json.loads(template_vars)
    if isinstance(record_in_inputs, str):
        record_in_inputs = json.loads(record_in_inputs)

    logger.info(f"template_vars keys: {template_vars.keys() if template_vars else 'None'}")
    logger.info(f"record keys: {record_in_inputs.keys() if record_in_inputs else 'None'}")

    # Check if template_vars contains record-like fields
    record_fields = {"text", "dataset_name", "split_type", "metadata", "record_id"}
    tv_has_record_data = bool(record_fields & set(template_vars.keys() if template_vars else []))

    if tv_has_record_data:
        logger.info("✓ template_vars contains record fields (expected when derived from record)")

        # CRITICAL: If template_vars has record data, inputs.record should NOT duplicate it
        if record_in_inputs and "text" in record_in_inputs:
            # Check if the text content is duplicated
            tv_text = template_vars.get("text", "")
            record_text = record_in_inputs.get("text", "")

            # This should FAIL - proving duplication exists
            assert tv_text != record_text or record_text == "", (
                f"Record content duplicated: 'text' appears in both template_vars and inputs.record.\n"
                f"template_vars.text: {tv_text[:100]}...\n"
                f"inputs.record.text: {record_text[:100]}...\n"
                f"When template_vars is derived from record, inputs.record should NOT contain duplicate data."
            )

            logger.info("✗ DUPLICATION DETECTED: text exists in both locations")
            logger.info(f"  template_vars.text length: {len(tv_text)}")
            logger.info(f"  inputs.record.text length: {len(record_text)}")
        else:
            logger.info("✓ No duplication - inputs.record does not contain text field")
    else:
        logger.info("✓ template_vars does not contain record fields (no duplication possible)")
