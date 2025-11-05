"""TRUE end-to-end test for agent initialization failure detection.

This test uses REAL components to verify that flows FAIL LOUDLY when agent
initialization fails. This reproduces the October 31st "trans" flow silent
failure where scorer agents couldn't initialize due to missing output_model class.

NO mocks - this tests the real orchestrator, real agent registration, real config
validation to ensure fail-fast behavior.
"""

import pytest

from buttermilk._core.config import AgentVariants
from buttermilk._core.types import RunRequest
from buttermilk.orchestrators.groupchat import AutogenOrchestrator


@pytest.mark.anyio
async def test_agent_init_failure_fails_loud(real_bm):
    """Test that flows FAIL LOUDLY when agent initialization fails.

    Reproduces the October 31st silent failure scenario:
    - Agent config references non-existent agent_obj class
    - Agent registration should fail immediately
    - Flow should raise exception, NOT complete silently
    - Error message should identify WHICH agent failed and WHY

    This is a TRUE E2E test using:
    - Real AutogenOrchestrator
    - Real AgentVariants configuration
    - Real agent registration (_register_agents)

    Expected behavior (fail-fast):
    - Exception raised during agent registration
    - Error message identifies the bad agent config
    - Flow does NOT reach "completed" status
    """
    # ARRANGE: Create agent config with invalid agent_obj (non-existent class in registry)
    # This simulates the October 31st bug: config referenced a class that doesn't exist
    # We use a valid config structure but reference a non-existent agent class
    invalid_agent_config = AgentVariants(
        agent_id="test-scorer-with-missing-class",
        role="SCORER",
        description="Test scorer with invalid agent_obj",
        agent_obj="NonExistentAgentClass",  # This class is not in AgentRegistry
        parameters={
            "model": "gemini-2.0-flash-001",
            "template": "simple",
        },
    )

    # Create minimal orchestrator config with the bad agent
    orchestrator = AutogenOrchestrator(
        name="test-agent-init-failure",
        agents={"scorer": invalid_agent_config},
        observers={},
        parameters={},
    )

    # Create a minimal RunRequest
    run_request = RunRequest(
        flow="test-agent-init-failure",
        session_id="test-session-silent-failure",
        inputs={"test": "data"},
    )

    # ACT + ASSERT: Attempting to run should raise Exception LOUDLY
    # The flow should NEVER complete successfully with a broken agent config
    with pytest.raises(Exception) as exc_info:
        # This should fail during _setup() -> _register_agents() -> get_configs()
        # The bug was that exceptions were caught at line 470-471 in groupchat.py
        # and logged but not re-raised, allowing the flow to "complete" silently
        await orchestrator._run(request=run_request, flow_name="test-agent-init-failure")

    # VERIFY: The exception should be FatalError or TypeError
    # and should contain information about the missing agent class
    error = exc_info.value
    error_msg = str(error)

    # The error should mention the nonexistent agent class or registry
    assert (
        "NonExistentAgentClass" in error_msg or "not found" in error_msg.lower() or "AgentRegistry" in error_msg
    ), f"Error message should identify the missing agent class. Got: {error_msg}"

    # The error should NOT be swallowed - we should get a clear exception
    assert error is not None, "Flow must raise exception when agent initialization fails"

    # SUCCESS: If we reach here, the flow failed LOUDLY as expected
    # The October 31st bug would have let this complete silently with status="completed"


@pytest.mark.anyio
async def test_agent_init_with_invalid_validator_fails_loud(real_bm):
    """Test that invalid output_model validator failures are caught and reported.

    Tests the specific validation failure path:
    - AgentConfig._validate_output_model calls make_class_import_validator
    - import_class_from_path fails to import the class
    - Exception should propagate with clear error message during config creation

    This verifies the fail-fast principle at the config validation layer.
    This test verifies that Pydantic validation catches errors BEFORE orchestrator run.
    """
    # ACT + ASSERT: Creating the config should fail validation immediately
    with pytest.raises(ImportError) as exc_info:
        # This should fail during AgentVariants.__init__ validation
        AgentVariants(
            agent_id="test-validator-failure",
            role="TESTER",
            description="Test config validation failure",
            agent_obj="LLMAgent",
            output_model="this.is.not.a.valid.module.path.ClassName",
            parameters={
                "model": "gemini-2.0-flash-001",
                "template": "simple",
            },
        )

    error_msg = str(exc_info.value)

    # Verify error message is actionable
    assert (
        "this.is.not.a.valid.module.path" in error_msg or "import" in error_msg.lower()
    ), f"Error should mention the invalid module path. Got: {error_msg}"


@pytest.mark.anyio
async def test_valid_agent_init_succeeds(real_bm):
    """Positive test: Valid agent config should initialize successfully.

    This test ensures our failure detection doesn't break valid configurations.
    Uses a real, valid agent config to verify normal operation.
    """
    # ARRANGE: Create valid agent config with correct agent class name
    # AgentRegistry uses short names like "LLMAgent", not full paths
    valid_config = AgentVariants(
        agent_id="test-valid-agent",
        role="TESTER",
        description="Test valid agent initialization",
        agent_obj="LLMAgent",
        parameters={
            "model": "gemini-2.0-flash-001",
            "template": "simple",
        },
    )

    orchestrator = AutogenOrchestrator(
        name="test-valid-init",
        agents={"tester": valid_config},
        observers={},
        parameters={},
    )

    run_request = RunRequest(
        flow="test-valid-init",
        session_id="test-session-valid",
        inputs={"test": "data"},
    )

    # ACT: Run setup - this should succeed
    try:
        termination_handler, interrupt_handler = await orchestrator._setup(request=run_request)

        # ASSERT: Setup completed successfully
        assert termination_handler is not None
        assert interrupt_handler is not None
        assert orchestrator._runtime is not None
        assert "TESTER" in orchestrator._agent_types

    finally:
        # Cleanup: Stop the runtime
        if hasattr(orchestrator, "_runtime") and orchestrator._runtime is not None:
            await orchestrator._runtime.stop()
