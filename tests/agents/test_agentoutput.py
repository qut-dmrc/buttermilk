import pytest
from pydantic import ValidationError

# Buttermilk core imports
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput

# --- Test Data ---

SAMPLE_REASONS_DATA = {
    "conclusion": "Test conclusion.",
    "prediction": False,
    "reasons": ["Reason 1", "Reason 2"],
    "confidence": "medium",
    "uncertainty": "low",
}

# --- Pytest Tests ---

try:
    from buttermilk._core.contract import AgentTrace as ActualAgentTrace

    # Correct the import name from Reasons to JudgeReasons
    from buttermilk.agents.judge import JudgeReasons as ActualJudgeReasons

    ACTUAL_MODELS_AVAILABLE = True
except ImportError:
    ACTUAL_MODELS_AVAILABLE = False
    ActualAgentTrace = None  # Define as None if import fails
    ActualJudgeReasons = None  # Define as None if import fails


# Use pytest.mark.skipif to skip these tests if the actual models can't be imported
@pytest.mark.skipif(not ACTUAL_MODELS_AVAILABLE, reason="Actual implementation models not found")
def test_actual_judge_reasons_direct_dump():
    """Verify ACTUAL JudgeReasons dumps correctly on its own."""
    try:
        reasons_obj = ActualJudgeReasons(**SAMPLE_REASONS_DATA)
        dumped_reasons = reasons_obj.model_dump()
        # Check that key fields are present
        assert dumped_reasons["conclusion"] == "Test conclusion."
        assert dumped_reasons["prediction"] == False
        assert dumped_reasons["reasons"] == ["Reason 1", "Reason 2"]
        assert dumped_reasons["uncertainty"] == "low"
        # Check that a preview field is generated
        assert "preview" in dumped_reasons
    except ValidationError as e:
        pytest.fail(f"ActualJudgeReasons validation failed: {e}")


@pytest.mark.skipif(not ACTUAL_MODELS_AVAILABLE, reason="Actual implementation models not found")
def test_actual_agent_trace_full_dump_includes_nested_outputs():
    """Test the ACTUAL AgentTrace full dump to see if it includes nested outputs.
    This directly tests the problematic scenario with your real code.
    """
    try:
        reasons_obj = ActualJudgeReasons(**SAMPLE_REASONS_DATA)
    except ValidationError as e:
        pytest.fail(f"ActualJudgeReasons validation failed during instantiation: {e}")

    # Create a minimal AgentConfig for testing
    minimal_agent_config = AgentConfig(role="TEST")

    # Instantiate ActualAgentTrace - provide minimal required fields
    output_obj = ActualAgentTrace(
        agent_info=minimal_agent_config,
        session_id="test_session",
        call_id="actual_test_id",
        inputs=AgentInput(),  # Add the required inputs field
    )
    output_obj.outputs = reasons_obj  # Assign the nested actual model

    # Perform the full dump using the actual implementation
    full_dump = output_obj.model_dump()

    # --- Assertions ---
    assert "outputs" in full_dump, "'outputs' key missing in actual model_dump result"
    assert full_dump["outputs"] != {}, "'outputs' field is an empty dict in actual model_dump result"
    assert full_dump["outputs"] == SAMPLE_REASONS_DATA, (
        f"Content of 'outputs' field ({full_dump.get('outputs')}) in actual dump does not match expected data ({SAMPLE_REASONS_DATA})"
    )
    assert full_dump["call_id"] == "actual_test_id"  # Verify other fields


@pytest.mark.skipif(not ACTUAL_MODELS_AVAILABLE, reason="Actual implementation models not found")
def test_actual_agent_trace_full_dump_with_default_outputs():
    """Test dumping the actual AgentTrace when 'outputs' is the default."""
    # Create a minimal AgentConfig for testing
    minimal_agent_config = AgentConfig(role="TEST")

    output_obj = ActualAgentTrace(
        agent_info=minimal_agent_config,
        session_id="test_session",
        agent_id="test",  # agent_id is part of AgentOutput base class
        inputs=AgentInput(),  # Add the required inputs field
    )  # Instantiate with defaults

    full_dump = output_obj.model_dump()

    assert "outputs" in full_dump
    # Check against the actual default value defined in AgentTrace
    # Assuming it defaults to {} based on the definition shown earlier
    assert full_dump["outputs"] == {}
