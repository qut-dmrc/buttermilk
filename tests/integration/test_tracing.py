import importlib

import pytest

from buttermilk._core.bm_init import BM

weave = importlib.import_module("weave")

EXPECTED_PROJECT_NAME = "buttermilk"
EXPECTED_JOB = "testing"


@pytest.mark.skip(reason="Weave dependency removed from buttermilk")
@pytest.mark.anyio
async def test_weave_tracing_initialised_and_creates_calls(real_bm: BM):
    """Simple integration check for Weave tracing.

    - If Weave is not configured in this environment, skip the test.
    - Otherwise, create a tiny @weave.op, start a call, assert IDs exist, and finish it.
    """
    if weave is None:
        pytest.skip("Weave library is not installed in this environment.")

    try:
        client = await real_bm.get_weave_client()
    except Exception:
        pytest.skip("Weave client is not available or not configured in this environment.")

    if client is None:
        pytest.skip("Weave client is not configured (get_weave_client returned None) in this environment.")

    # Define a minimal operation to be traced
    def _inc(x: int) -> int:
        return x + 1

    op = weave.op(_inc, call_display_name="test-inc-op")

    # Start a trace call via the client
    call = client.create_call(
        op,
        inputs={"x": 41},
        display_name="test-weave-trace",
        attributes={"component": "tests", "purpose": "smoke"},
    )

    # Basic assertions that a call was created with identifiers
    assert call is not None, "Weave create_call should return a call object"
    assert getattr(call, "id", None), "Weave call should have an id"
    assert getattr(call, "trace_id", None), "Weave call should have a trace_id"

    # Finish the call (simulate successful output)
    client.finish_call(call, output={"result": 42}, op=op)

    fetched = client.get_call(call.id)
    assert fetched is not None, "Weave get_call should return the created call"


@pytest.mark.skip(reason="Weave dependency removed from buttermilk")
@pytest.mark.anyio
async def test_unified_tracing_config_integration(real_bm: BM):
    """Verify that unified tracing configuration works with existing BM infrastructure.

    This test validates that the recent changes to unified tracing configuration
    under infrastructure.tracing work correctly with the existing BM.get_weave_client()
    delegation pattern.
    """
    if weave is None:
        pytest.skip("Weave library is not installed in this environment.")

    try:
        client = await real_bm.get_weave_client()
    except Exception as e:
        pytest.skip(f"Weave client is not available or not configured: {e}")

    if client is None:
        pytest.skip("Weave client is not configured (get_weave_client returned None).")

    # Verify that the client has the expected methods (fixes NoneType error)
    assert hasattr(client, "create_call"), "Weave client should have create_call method"
    assert hasattr(client, "finish_call"), "Weave client should have finish_call method"
    assert hasattr(client, "get_call"), "Weave client should have get_call method"

    # Test that we can use the client for tracing operations
    def test_unified_config_op(value: str) -> str:
        return f"processed: {value}"

    op = weave.op(test_unified_config_op, call_display_name="test-unified-config")

    # Create a call to test the unified configuration
    call = client.create_call(
        op,
        inputs={"value": "unified-config-test"},
        display_name="test-unified-tracing-config",
        attributes={"config_type": "unified", "test": "integration"},
    )

    # Verify call was created successfully
    assert call is not None, "Call should be created successfully with unified config"
    assert getattr(call, "id", None), "Call should have an ID"
    assert getattr(call, "trace_id", None), "Call should have a trace_id"

    # Finish the call
    client.finish_call(call, output={"result": "processed: unified-config-test"}, op=op)

    # Verify we can retrieve the call
    retrieved_call = client.get_call(call.id)
    assert retrieved_call is not None, "Should be able to retrieve the completed call"

    # This test confirms that:
    # 1. The unified tracing config works with existing BM infrastructure
    # 2. The fix for "'NoneType' object has no attribute 'create_call'" is working
    # 3. Config-based credential initialization is functional
    # 4. The delegation from BM.get_weave_client() to ExecutionContext works properly


@pytest.mark.skip(reason="Weave dependency removed from buttermilk")
@pytest.mark.anyio
async def test_weave_collection_uses_project_name(real_bm):
    """Test that weave initialization uses project name, not execution context ID.

    Weave should ALWAYS use project_name for collection, never fall back to
    'execution-context-{id}'.
    """

    try:
        weave_client = await real_bm.get_weave_client()
        if weave_client is None:
            pytest.skip("Weave client not initialized (may be missing credentials or disabled)")
    except Exception as e:
        # If weave initialization fails (missing credentials, etc.),
        # that's okay for this test - we're just checking the config
        pytest.skip(f"Weave initialization failed: {e}")

    # Verify the weave project name matches our project_name
    # The weave project should be {entity}/{project_name}
    assert EXPECTED_PROJECT_NAME in weave_client.project, f"Weave project should contain '{EXPECTED_PROJECT_NAME}', got: {weave_client.project}"

    # Verify it does NOT use execution-context prefix
    assert "execution-context" not in weave_client.project, f"Weave should not use 'execution-context' prefix, got: {weave_client.project}"
