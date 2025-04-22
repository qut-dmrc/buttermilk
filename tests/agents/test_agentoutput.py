import pytest
from pydantic import BaseModel, Field, ValidationError, computed_field
from typing import Any, List, Dict, Literal, Optional, Union


# --- Test Data ---

SAMPLE_REASONS_DATA = {
    "conclusion": "Test conclusion.",
    "prediction": False,
    "reasons": ["Reason 1", "Reason 2"],
    "confidence": "medium",
}

# --- Pytest Tests ---

try:
    from buttermilk._core.contract import AgentOutput as ActualAgentOutput
    from buttermilk.agents.judge import AgentReasons as ActualAgentReasons

    ACTUAL_MODELS_AVAILABLE = True
except ImportError:
    ACTUAL_MODELS_AVAILABLE = False
    ActualAgentOutput = None  # Define as None if import fails
    ActualAgentReasons = None  # Define as None if import fails


# Use pytest.mark.skipif to skip these tests if the actual models can't be imported
@pytest.mark.skipif(not ACTUAL_MODELS_AVAILABLE, reason="Actual implementation models not found")
def test_actual_agent_reasons_direct_dump():
    """Verify ACTUAL AgentReasons dumps correctly on its own."""
    try:
        reasons_obj = ActualAgentReasons(**SAMPLE_REASONS_DATA)
        dumped_reasons = reasons_obj.model_dump()
        assert dumped_reasons == SAMPLE_REASONS_DATA
    except ValidationError as e:
        pytest.fail(f"ActualAgentReasons validation failed: {e}")


@pytest.mark.skipif(not ACTUAL_MODELS_AVAILABLE, reason="Actual implementation models not found")
def test_actual_agent_output_full_dump_includes_nested_outputs():
    """
    Test the ACTUAL AgentOutput full dump to see if it includes nested outputs.
    This directly tests the problematic scenario with your real code.
    """
    try:
        reasons_obj = ActualAgentReasons(**SAMPLE_REASONS_DATA)
    except ValidationError as e:
        pytest.fail(f"ActualAgentReasons validation failed during instantiation: {e}")

    # Instantiate ActualAgentOutput - provide minimal required fields if any
    # Adjust instantiation based on ActualAgentOutput's actual fields/defaults
    output_obj = ActualAgentOutput(
        call_id="actual_test_id",
        # Add other necessary fields for ActualAgentOutput if they lack defaults
    )
    output_obj.outputs = reasons_obj  # Assign the nested actual model

    # Perform the full dump using the actual implementation
    full_dump = output_obj.model_dump()

    # --- Assertions ---
    assert "outputs" in full_dump, "'outputs' key missing in actual model_dump result"
    assert full_dump["outputs"] != {}, "'outputs' field is an empty dict in actual model_dump result"
    assert (
        full_dump["outputs"] == SAMPLE_REASONS_DATA
    ), f"Content of 'outputs' field ({full_dump.get('outputs')}) in actual dump does not match expected data ({SAMPLE_REASONS_DATA})"
    assert full_dump["call_id"] == "actual_test_id"  # Verify other fields


@pytest.mark.skipif(not ACTUAL_MODELS_AVAILABLE, reason="Actual implementation models not found")
def test_actual_agent_output_full_dump_with_default_outputs():
    """Test dumping the actual AgentOutput when 'outputs' is the default."""
    output_obj = ActualAgentOutput()  # Instantiate with defaults

    full_dump = output_obj.model_dump()

    assert "outputs" in full_dump
    # Check against the actual default value defined in AgentOutput
    # Assuming it defaults to {} based on the definition shown earlier
    assert full_dump["outputs"] == {}
