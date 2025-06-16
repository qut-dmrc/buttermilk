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
        agent_id="test",  # Add required agent_id field
        inputs=AgentInput(),  # Add the required inputs field
    )
    output_obj.outputs = reasons_obj  # Assign the nested actual model

    # Perform the full dump using the actual implementation
    full_dump = output_obj.model_dump()

    # --- Assertions ---
    assert "outputs" in full_dump, "'outputs' key missing in actual model_dump result"
    assert full_dump["outputs"] != {}, "'outputs' field is an empty dict in actual model_dump result"
    
    # Verify core fields from SAMPLE_REASONS_DATA are preserved
    outputs = full_dump["outputs"]
    assert outputs["conclusion"] == SAMPLE_REASONS_DATA["conclusion"]
    assert outputs["prediction"] == SAMPLE_REASONS_DATA["prediction"]
    assert outputs["reasons"] == SAMPLE_REASONS_DATA["reasons"]
    assert outputs["uncertainty"] == SAMPLE_REASONS_DATA["uncertainty"]
    
    # Verify computed field is present (but don't check exact value since it has randomness)
    assert "preview" in outputs, "Computed 'preview' field should be present in output"
    assert isinstance(outputs["preview"], str), "Preview field should be a string"
    
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

    # When outputs is default/None, it may be excluded from model_dump due to exclude_none/exclude_unset config
    # Check that we can access the outputs field directly even if it's not in the dump
    assert hasattr(output_obj, 'outputs'), "AgentTrace should have outputs attribute"
    
    # If outputs is excluded from dump, it should be because it's None or default
    if "outputs" in full_dump:
        assert full_dump["outputs"] in [None, {}], "Default outputs should be None or empty dict"
    else:
        # Verify that the outputs field exists but is None/default so it gets excluded
        assert output_obj.outputs is None or output_obj.outputs == {}, "Excluded outputs should be None or empty"
