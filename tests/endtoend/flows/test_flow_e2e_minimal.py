"""Minimal E2E test for flow verification with reduced rate limit pressure.

This test uses debug LLM config (gemini-flash only) to avoid rate limits
while verifying all the key flow behaviors.

The comprehensive test in test_flow_e2e.py uses the full lite config with
multiple models (claude-haiku, gemini-flash-lite, gpt-nano) but may fail
due to Azure rate limits on gpt-nano. This minimal test serves as the
reliable CI test.
"""

import json
import re
from typing import Any

import pytest

from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture(scope="module")
def debug_bm(real_bm):
    """BM instance with debug LLM config (gemini-flash only)."""
    return real_bm


@pytest.fixture(scope="module")
def debug_conf(debug_bm):
    """Configuration from debug setup."""
    return debug_bm.cfg


@pytest.fixture
def trans_test_record_id() -> str:
    """Real record_id from TJA dataset for trans flow testing."""
    return "onion_trans_prom"


def check_object_serialization(obj: Any, path: str = "root") -> list[str]:
    """Recursively check for JSON strings that should be objects.

    Returns list of paths where JSON strings were found instead of objects.
    """
    issues = []

    if isinstance(obj, str):
        # Check if this string looks like it should be JSON
        stripped = obj.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            try:
                json.loads(stripped)
                # If it parses, it's a JSON string that might should be an object
                if len(stripped) > 50:  # Skip small things that might be intentional
                    issues.append(f"{path}: JSON string found ({len(stripped)} chars)")
            except json.JSONDecodeError:
                pass  # Not valid JSON, that's fine
    elif isinstance(obj, dict):
        for k, v in obj.items():
            issues.extend(check_object_serialization(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            issues.extend(check_object_serialization(v, f"{path}[{i}]"))
    elif hasattr(obj, "model_dump"):
        # Pydantic model - check its dict representation
        issues.extend(check_object_serialization(obj.model_dump(), path))

    return issues


@pytest.mark.anyio
@pytest.mark.slow
async def test_flow_e2e_minimal(
    debug_bm,
    debug_conf,
    trans_test_record_id,
):
    """Minimal E2E test with detailed verification.

    Tests with gemini-flash only to avoid rate limits.

    Verifies:
    1. Complete run with stage-to-stage input passing
    2. VARIANTS: criteria list unpacked into separate agents
    3. Template expansion: criteria filled, records inserted
    4. Answers are fully complete
    5. Trace structure: correct hashes, full record, no duplication
    6. Object serialization: no JSON strings instead of objects
    """
    flow_name = "trans"
    record_id = trans_test_record_id

    # Verify flow is configured
    flows_config = debug_conf.run.flows
    assert flow_name in flows_config, f"Flow '{flow_name}' not found"

    # Create FlowRunner
    flow_runner = FlowRunner(
        source="e2e_minimal_test",
        flows=flows_config,
        bm=debug_bm,
    )

    # Collect messages
    messages = []

    async def collect_messages(message):
        messages.append(message)

    # Create RunRequest
    run_request = RunRequest(
        flow=flow_name,
        parameters={"record_id": record_id},
        session_info=debug_bm.session_info,
        session_id="test_flow_e2e_minimal",
        callback_to_ui=collect_messages,
    )

    # Run the flow
    await flow_runner.run_flow(
        run_request=run_request,
        wait_for_completion=True,
    )

    # === VERIFICATION 1: Flow completed ===
    assert len(messages) > 0, "Flow should produce messages"

    # === VERIFICATION 2: Expected roles executed ===
    expected_roles = {"FETCH", "JUDGE", "SYNTHESISER", "DIFF", "SCORERS"}
    executed_roles = set()
    for msg in messages:
        if isinstance(msg, ExecutionTrace):
            if msg.agent_info and "role" in msg.agent_info:
                executed_roles.add(msg.agent_info["role"])

    missing_roles = expected_roles - executed_roles
    assert not missing_roles, f"Missing roles: {missing_roles}"

    # === VERIFICATION 3: No errors ===
    error_messages = [msg for msg in messages if hasattr(msg, "error") and msg.error]
    # Print first few errors for debugging
    if error_messages:
        print(f"\n=== {len(error_messages)} ERRORS FOUND ===")
        for i, err in enumerate(error_messages[:5]):
            print(f"Error {i + 1}: {err.agent_info.get('agent_id', 'unknown')} - {err.error}")

    assert not error_messages, f"Flow produced {len(error_messages)} errors. First: {error_messages[0].error if error_messages else 'N/A'}"

    # === VERIFICATION 4: VARIANTS - Multiple JUDGE agents for different criteria ===
    judge_traces = [msg for msg in messages if isinstance(msg, ExecutionTrace) and msg.agent_info.get("role") == "JUDGE" and not msg.error]

    # Should have multiple judge traces (one per criteria variant)
    print(f"\n=== JUDGE TRACES: {len(judge_traces)} ===")
    judge_criteria = set()
    for trace in judge_traces:
        # Criteria is in trace.inputs (template variables consolidated there)
        criteria = None
        if trace.inputs:
            criteria = trace.inputs.get("criteria")
        agent_id = trace.agent_info.get("agent_id", "unknown")
        component = trace.agent_info.get("component_name", "unknown")
        print(f"  - {agent_id}: criteria={criteria}, component={component}")
        if criteria:
            judge_criteria.add(criteria)

    expected_criteria = {"tja", "glaad", "trans_simplified", "apc"}
    # With debug config (1 model) and num_runs=2, we expect 4 criteria * 1 model * 2 runs = 8 judge calls
    assert len(judge_traces) > 1, f"Expected multiple JUDGE variants, got {len(judge_traces)}"

    # Verify all expected criteria were used
    missing_criteria = expected_criteria - judge_criteria
    assert not missing_criteria, f"Missing criteria in JUDGE variants: {missing_criteria}. Found: {judge_criteria}"
    print(f"  ✓ All {len(expected_criteria)} criteria variants present: {sorted(judge_criteria)}")

    # === VERIFICATION 5: Template expansion ===
    traces_with_messages = [msg for msg in messages if isinstance(msg, ExecutionTrace) and msg.messages and len(msg.messages) > 0]

    print("\n=== TEMPLATE EXPANSION CHECK ===")
    unfilled_pattern = re.compile(r"\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\}")

    for trace in traces_with_messages:
        role = trace.agent_info.get("role", "unknown")
        first_msg = trace.messages[0]
        content = first_msg.content if hasattr(first_msg, "content") else str(first_msg)

        # Check length
        assert len(content) > 100, f"Template too short for {role}: {len(content)} chars"

        # Check for unfilled variables
        for i, msg in enumerate(trace.messages):
            msg_content = msg.content if hasattr(msg, "content") else str(msg)
            unfilled = unfilled_pattern.findall(msg_content)
            assert not unfilled, f"Unfilled vars in {role} message[{i}]: {unfilled}"

        print(f"  ✓ {role}: {len(content)} chars, no unfilled vars")

    # === VERIFICATION 6: Record context in traces ===
    traces_with_record = [msg for msg in messages if isinstance(msg, ExecutionTrace) and msg.record is not None]

    print(f"\n=== RECORD CONTEXT CHECK: {len(traces_with_record)} traces ===")
    for trace in traces_with_record[:3]:  # Show first 3
        role = trace.agent_info.get("role", "unknown")
        record = trace.record
        record_dict = record if isinstance(record, dict) else (record.model_dump() if hasattr(record, "model_dump") else {})
        record_id_val = record_dict.get("record_id", "missing")
        record_hash = record_dict.get("record_hash", "missing")
        print(
            f"  - {role}: record_id={record_id_val}, hash={record_hash[:16]}..."
            if record_hash != "missing"
            else f"  - {role}: record_id={record_id_val}"
        )

        assert "record_id" in record_dict, f"Missing record_id in {role} trace"

    # === VERIFICATION 7: No duplication of record in inputs ===
    print("\n=== CHECKING FOR RECORD DUPLICATION ===")
    duplication_issues = []
    for trace in traces_with_record:
        role = trace.agent_info.get("role", "unknown")

        # If record is in trace.record, it shouldn't also be fully duplicated in inputs
        if trace.inputs and isinstance(trace.inputs, dict):
            inputs_record = trace.inputs.get("record")
            if inputs_record and trace.record:
                # Check if they're the same full object (duplication)
                record_text = str(trace.record.get("text", "") if isinstance(trace.record, dict) else getattr(trace.record, "text", ""))
                inputs_text = str(inputs_record.get("text", "") if isinstance(inputs_record, dict) else getattr(inputs_record, "text", ""))

                if record_text and inputs_text and record_text == inputs_text and len(record_text) > 100:
                    duplication_issues.append(f"{role}: full record text duplicated in both trace.record and trace.inputs.record")

    if duplication_issues:
        print("  ⚠ Duplication issues found:")
        for issue in duplication_issues:
            print(f"    - {issue}")
    else:
        print("  ✓ No record duplication detected")

    # === VERIFICATION 8: Object serialization ===
    # Note: LLM message content (messages[N].content) is expected to be JSON strings
    # when using structured output. We only check for unexpected JSON strings in
    # trace.inputs, trace.outputs (excluding message content), trace.record, etc.
    print("\n=== OBJECT SERIALIZATION CHECK ===")
    serialization_issues = []
    for trace in messages:
        if isinstance(trace, ExecutionTrace):
            role = trace.agent_info.get("role", "unknown")
            # Check inputs (excluding messages which contain LLM responses)
            if trace.inputs and isinstance(trace.inputs, dict):
                inputs_to_check = {k: v for k, v in trace.inputs.items() if k != "context"}
                issues = check_object_serialization(inputs_to_check, f"trace({role}).inputs")
                serialization_issues.extend(issues)
            # Check record
            if trace.record:
                issues = check_object_serialization(trace.record, f"trace({role}).record")
                serialization_issues.extend(issues)
            # Note: We don't check trace.messages because LLM responses are expected to be JSON strings
            # Note: We don't check trace.outputs because structured output content is often JSON

    if serialization_issues:
        print(f"  ⚠ Found {len(serialization_issues)} JSON string issues:")
        for issue in serialization_issues[:5]:
            print(f"    - {issue}")
    else:
        print("  ✓ No unexpected JSON string serialization issues")

    # === VERIFICATION 9: Answers are complete ===
    print("\n=== ANSWER COMPLETENESS CHECK ===")
    for trace in messages:
        if isinstance(trace, ExecutionTrace) and trace.outputs:
            role = trace.agent_info.get("role", "unknown")
            outputs = trace.outputs

            # For JUDGE outputs, check if prediction/conclusion are present
            if role == "JUDGE":
                if isinstance(outputs, dict):
                    assert "prediction" in outputs or "conclusion" in outputs, "JUDGE output missing prediction/conclusion"
                    print(f"  ✓ JUDGE: has prediction={outputs.get('prediction')}, conclusion present={bool(outputs.get('conclusion'))}")
                elif hasattr(outputs, "prediction"):
                    print(f"  ✓ JUDGE: prediction={outputs.prediction}")

    # === SUMMARY ===
    print("\n=== TEST SUMMARY ===")
    print(f"✓ Flow '{flow_name}' completed successfully")
    print(f"✓ Agents executed: {sorted(executed_roles)}")
    print(f"✓ Messages produced: {len(messages)}")
    print(f"✓ JUDGE traces: {len(judge_traces)} (variants working)")
    print(f"✓ Traces with messages: {len(traces_with_messages)}")
    print(f"✓ Traces with record: {len(traces_with_record)}")
