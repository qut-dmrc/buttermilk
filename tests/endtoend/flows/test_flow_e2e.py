"""Generic E2E test for complete flow execution through FlowRunner.

This test validates flows from testing.yaml run successfully from start to finish,
exercising the same code path as batch runs. It verifies all configured agents
execute and the flow completes without errors.

TRUE E2E test:
- Loads REAL flow configurations from testing.yaml
- Uses REAL FlowRunner (batch entry point)
- Calls REAL LLM APIs (Vertex AI, etc.)
- Uses REAL agent implementations
- NO mocks of internal Buttermilk code

Add flows to testing.yaml flows: section to test them here.

NOTE: The 'trans' flow and its dependencies (criteria templates, TJA storage config)
were removed in commit 44c233d9 ("remove 'tasks' in favour of variants").
These tests are skipped until a replacement flow is configured in testing.yaml.
"""

import re

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.skip(reason="No flows configured: 'trans' flow removed in commit 44c233d9")]

from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture
def trans_test_record_id() -> str:
    """Real record_id from TJA dataset for trans flow testing.

    Uses an actual record from the BigQuery TJA dataset that the
    FETCH agent will retrieve. This tests the complete flow execution
    including data fetching, judging, synthesis, scoring, and diff.
    """
    return "onion_trans_prom"


# Parametrize with (flow_name, record_id_fixture_name, expected_agent_roles)
# Add more flows here as they're added to testing.yaml
FLOW_TEST_CASES = [
    pytest.param(
        "trans",
        "trans_test_record_id",
        {"FETCH", "JUDGE", "SYNTHESISER", "DIFF", "SCORERS"},
        id="trans_flow",
        marks=pytest.mark.slow,  # LLM calls are slow
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize("flow_name,record_fixture_name,expected_roles", FLOW_TEST_CASES)
async def test_flow_execution_e2e(
    flow_name: str,
    record_fixture_name: str,
    expected_roles: set[str],
    real_bm,
    real_conf,
    request,
):
    """Test complete flow execution through FlowRunner.

    This is a TRUE E2E test that:
    - Loads the actual flow configuration from testing.yaml
    - Creates a real FlowRunner (the batch run entry point)
    - Executes the complete flow with real LLM calls
    - Verifies all expected agents execute
    - Checks the flow completes successfully

    Args:
        flow_name: Name of the flow to test (must be in testing.yaml flows:)
        record_fixture_name: Name of the pytest fixture providing test record_id
        expected_roles: Set of agent roles that should execute in this flow
        real_bm: Fully configured BM instance from testing.yaml
        real_conf: Raw configuration dict from testing.yaml
        request: Pytest request object for accessing fixtures by name
    """
    # Get the record_id from the named fixture
    record_id = request.getfixturevalue(record_fixture_name)

    # Verify flow is configured in testing.yaml
    # Flows are loaded into real_conf.run.flows by Hydra defaults
    assert hasattr(real_conf, "run") and hasattr(real_conf.run, "flows"), (
        "No flows configured in testing.yaml. Add flows: section with flow configurations in defaults."
    )
    flows_config = real_conf.run.flows
    assert flow_name in flows_config, f"Flow '{flow_name}' not found in testing.yaml flows. Available flows: {list(flows_config.keys())}"

    # Create FlowRunner with flows from config
    # This is the same entry point used by batch runs
    flow_runner = FlowRunner(
        source="e2e_test",
        flows=flows_config,
        bm=real_bm,  # Use session-scoped BM
    )

    # Collect messages for verification
    messages = []

    async def collect_messages(message):
        """Callback to collect flow messages."""
        messages.append(message)

    # Create RunRequest - same structure as batch runs
    run_request = RunRequest(
        flow=flow_name,
        parameters={"record_id": record_id},
        session_info=real_bm.session_info,
        session_id="test_flow_e2e",
        callback_to_ui=collect_messages,
    )

    # Run the flow and wait for completion
    # This exercises the full flow execution path
    await flow_runner.run_flow(
        run_request=run_request,
        wait_for_completion=True,  # Wait for flow to finish
    )

    # VERIFY: Flow completed successfully
    assert len(messages) > 0, "Flow should produce messages"

    # VERIFY: All expected agent roles executed
    executed_roles = set()
    for msg in messages:
        if isinstance(msg, ExecutionTrace):
            if msg.agent_info and "role" in msg.agent_info:
                executed_roles.add(msg.agent_info["role"])

    missing_roles = expected_roles - executed_roles
    assert not missing_roles, (
        f"Expected agents did not execute: {missing_roles}. "
        f"Executed roles: {executed_roles}. "
        f"This may indicate agent initialization failures or flow configuration issues."
    )

    # VERIFY: No error messages in flow execution
    error_messages = [msg for msg in messages if hasattr(msg, "error") and msg.error]
    assert not error_messages, (
        f"Flow completed but produced {len(error_messages)} error messages. First error: {error_messages[0] if error_messages else 'N/A'}"
    )

    # VERIFY: Template variables are filled in messages
    # Get traces that have LLM messages (from agents that call LLMs)
    traces_with_messages = [msg for msg in messages if isinstance(msg, ExecutionTrace) and msg.messages and len(msg.messages) > 0]

    for trace in traces_with_messages:
        # Check messages[0] (typically system prompt with template) has substantial content
        first_message = trace.messages[0]
        content = first_message.content if hasattr(first_message, "content") else str(first_message)

        # Template-filled prompts should have substantial content (> 100 chars at minimum)
        # Real prompts with filled variables are typically 500+ chars
        assert len(content) > 100, (
            f"First message content too short ({len(content)} chars). "
            f"Template variables may not be filled. Agent: {trace.agent_info.get('role', 'unknown')}. "
            f"Content preview: {content[:200]}..."
        )

        # VERIFY: No unfilled Jinja2 template variables in any message
        # Unfilled variables look like: {{variable_name}} or {{ variable_name }}
        unfilled_pattern = re.compile(r"\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\}")

        for i, msg in enumerate(trace.messages):
            msg_content = msg.content if hasattr(msg, "content") else str(msg)
            unfilled_vars = unfilled_pattern.findall(msg_content)
            assert not unfilled_vars, (
                f"Unfilled template variables found in message[{i}]: {unfilled_vars}. "
                f"Agent: {trace.agent_info.get('role', 'unknown')}. "
                f"Content preview: {msg_content[:300]}..."
            )

    # VERIFY: trace.record is filled for traces that process records
    traces_with_record = [msg for msg in messages if isinstance(msg, ExecutionTrace) and msg.record is not None]

    # At least some traces should have record context (agents processing the input record)
    # Note: Not all agents may have record (e.g., orchestrator traces)
    if traces_with_messages:
        # For flows that process records, we expect at least one trace to have record filled
        assert len(traces_with_record) > 0 or not any("record" in str(trace.inputs).lower() for trace in traces_with_messages), (
            f"Expected at least one trace with record context filled. "
            f"Found {len(traces_with_record)} traces with record out of {len(traces_with_messages)} "
            f"traces with messages."
        )

    for trace in traces_with_record:
        record = trace.record
        record_dict = record if isinstance(record, dict) else record.model_dump() if hasattr(record, "model_dump") else {}

        # Record should have essential fields
        assert "record_id" in record_dict or hasattr(record, "record_id"), (
            f"trace.record missing record_id. Agent: {trace.agent_info.get('role', 'unknown')}. Record type: {type(record).__name__}"
        )

    # SUCCESS: Flow executed all agents and completed without errors
    print(f"✓ Flow '{flow_name}' completed successfully")
    print(f"✓ Agents executed: {sorted(executed_roles)}")
    print(f"✓ Messages produced: {len(messages)}")
    print(f"✓ Traces with filled messages: {len(traces_with_messages)}")
    print(f"✓ Traces with record context: {len(traces_with_record)}")
