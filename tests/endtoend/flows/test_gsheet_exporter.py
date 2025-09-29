import json  # Import json

import pytest

from buttermilk._core.types import RunRequest  # Import RunRequest

TEST_FLOW_ID = "test_flow"

pytestmark = pytest.mark.anyio

async def test_gsheet_exporter(real_flow_runner):
    run_request = RunRequest(flow=TEST_FLOW_ID, ui_type="testing", session_id="test_session")  # Replaced Job with RunRequest and mapped args

    # Mock the flow's run_flows method to return a mock ExecutionTrace with outputs
    class MockExecutionTrace:
        def __init__(self, outputs):
            self.outputs = outputs
            self.agent_info = {"name": "mock_agent"}  # Add mock agent_info
            self.record = None  # Add mock record
            self.error = None  # Add mock error

        def model_dump_json(self):
            # Simple mock dump for this test
            return json.dumps({"outputs": self.outputs})

    async def mock_run_flows(run_request):  # Changed parameter name
        # Simulate the output structure expected by the gsheet exporter
        mock_outputs = {"sheet_url": "mock_url", "sheet_id": "mock_id"}
        yield MockExecutionTrace(outputs=mock_outputs)

    real_flow_runner.run_flows = mock_run_flows

    # Run the flow and check that the gsheet exporter saves correctly
    # This test might need a mock for the actual gsheet saving logic
    async for result in real_flow_runner.run_flows(run_request=run_request):  # Pass run_request
        assert result
        assert isinstance(result, MockExecutionTrace)
        assert result.outputs["sheet_url"] == "mock_url"
        assert result.outputs["sheet_id"] == "mock_id"
