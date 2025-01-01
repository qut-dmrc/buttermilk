from typing import Any

import pytest
from fastapi.testclient import TestClient

from buttermilk._core.runner_types import Job
from buttermilk.api.flow import app
from buttermilk.api.stream import FlowRequest

client = TestClient(app)


@pytest.fixture
def flow_request_data():
    req_cfg = {
        "model": "haiku",
        "template": "judge",
        "template_vars": {"formatting": "json_rules", "criteria": "criteria_ordinary"},
        "text": "Sample text",
        "uri": None,
        "media_b64": None,
    }
    req = FlowRequest(**req_cfg)
    return req.model_dump()


def test_run_flow(bm: Any, flow_request_data: dict[str, Any]):
    response = client.post("/flow/test", json=flow_request_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response


def test_run_flow_html(bm: Any, flow_request_data: dict[str, Any]):
    response = client.post("/html/flow/test_flow", json=flow_request_data)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert (
        "Sample text" in response.text
    )  # Check if the response contains the expected text


def test_get_runs(bm: Any):
    response = client.get("/runs")
    assert response.status_code == 200
    assert "text/html" not in response.headers["content-type"]
    assert all([isinstance(x, Job) for x in response])
