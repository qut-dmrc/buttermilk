from typing import Any

import pytest
from fastapi.testclient import TestClient

from buttermilk.api.flow import FlowRequest, app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


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


def test_api_request_simple(
    flow_request_data: dict,
    client,
):
    resolved_req_cfg = {
        k: v() if callable(v) else v for k, v in flow_request_data.items()
    }
    flow_request = FlowRequest(**resolved_req_cfg)
    response = client.post("/flow/simple", json=flow_request.model_dump(mode="json"))
    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response


def test_run_flow(bm: Any, client, flow_request_data: dict[str, Any]):
    response = client.post("/flow/test", json=flow_request_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response


def test_run_flow_html(client, bm: Any, flow_request_data: dict[str, Any]):
    response = client.post("/html/flow/test_flow", json=flow_request_data)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert (
        "Sample text" in response.text
    )  # Check if the response contains the expected text


def test_get_runs(client, bm: Any):
    response = client.get("/runs")
    assert response.status_code == 200
    assert "text/html" not in response.headers["content-type"]
