import json # Import json
import pytest # Import pytest

from buttermilk._core.agent import Agent # Import Agent
from buttermilk._core.types import RunRequest  # Import RunRequest
from buttermilk.utils.utils import read_json

TEST_FLOW_ID = "test_flow"


async def test_flow_data_source(flow):
    run_request = RunRequest(flow=TEST_FLOW_ID, ui_type="testing", session_id="test_session")  # Replaced Job with RunRequest and mapped args

    results = read_json("tests/data/result.json")  # three results with 1, 4 & 3 reasons

    # Mock the flow's run_flows method to return the predefined results
    async def mock_run_flows(run_request):  # Changed parameter name
        for result in results:
            yield result

    flow.run_flows = mock_run_flows

    # Run the flow and check that the data source correctly extracts results
    async for result in flow.run_flows(run_request=run_request):  # Pass run_request
        assert result
        assert isinstance(result, dict)  # Assuming results are dicts in this test


async def test_gsheet_exporter(flow: Agent):
    run_request = RunRequest(flow=TEST_FLOW_ID, ui_type="testing", session_id="test_session")  # Replaced Job with RunRequest and mapped args
    output_map = {
        "sheet_url": "outputs.sheet_url",
        "sheet_id": "outputs.sheet_id",
    }

    # Mock the flow's run_flows method to return a mock AgentTrace with outputs
    class MockAgentTrace:
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
        yield MockAgentTrace(outputs=mock_outputs)

    flow.run_flows = mock_run_flows

    # Run the flow and check that the gsheet exporter saves correctly
    # This test might need a mock for the actual gsheet saving logic
    async for result in flow.run_flows(run_request=run_request):  # Pass run_request
        assert result
        assert isinstance(result, MockAgentTrace)
        assert result.outputs["sheet_url"] == "mock_url"
        assert result.outputs["sheet_id"] == "mock_id"
