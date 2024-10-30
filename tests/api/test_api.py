import pytest
from fastapi.testclient import TestClient
from buttermilk.api.flow import app, FlowRequest
from buttermilk.runner._runner_types import Job

client = TestClient(app)

@pytest.fixture
def flow_request_data():
    return {
        "model": "gpt4o",
        "template": "judge",
        "template_vars": {},
        "text": "Sample text",
        "uri": None,
        "media_b64": None
    }

def test_run_flow(flow_request_data):
    response = client.post("/flow/test_flow", json=flow_request_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response

def test_run_flow_html(flow_request_data):
    response = client.post("/html/flow/test_flow", json=flow_request_data)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Sample text" in response.text  # Check if the response contains the expected text

def test_get_runs():
    response = client.get("/runs")
    assert response.status_code == 200
    assert "text/html" not in response.headers["content-type"]
    assert all([isinstance(x, Job) for x in response])

