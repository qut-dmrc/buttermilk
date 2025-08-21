import importlib

import pytest

from buttermilk._core.bm_init import BM

weave = importlib.import_module("weave")


@pytest.mark.anyio
async def test_weave_tracing_initialised_and_creates_calls(bm: BM):
    """Simple integration check for Weave tracing.

    - If Weave is not configured in this environment, skip the test.
    - Otherwise, create a tiny @weave.op, start a call, assert IDs exist, and finish it.
    """
    if weave is None:
        pytest.skip("Weave library is not installed in this environment.")

    client = bm.weave

    if client is None:
        pytest.skip("Weave client is not configured (bm.weave is None) in this environment.")

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
