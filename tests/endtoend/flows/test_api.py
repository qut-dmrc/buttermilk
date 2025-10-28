from typing import Any

import pytest
from fastapi.testclient import TestClient

from buttermilk import BM
from buttermilk.api.flow import create_app


@pytest.fixture(scope="session")
def client(real_flow_runner, bm: BM) -> TestClient:
    # Initialize with minimal configuration for testing

    app = create_app(flows=real_flow_runner, bm=bm)
    return TestClient(app)


@pytest.fixture
def flow_request_data():
    # Return raw dict for flow request data
    return {
        "flow": "test_minimal",
        "model": "claude45haiku",
        "template": "judge",
        "template_vars": {"formatting": "json_rules", "criteria": "criteria_ordinary"},
        "text": "Sample text",
        "uri": None,
        "media_b64": None,
    }


def test_api_request_simple(
    flow_request_data: dict,
    client,
):
    # Send data directly as dict to API
    response = client.post("/flow/simple", json=flow_request_data)

    # Debug the response if it fails
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")

    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response


def test_run_flow(client, flow_request_data: dict[str, Any]):
    response = client.post("/flow/test", json=flow_request_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response


def test_run_flow_html(client, flow_request_data: dict[str, Any]):
    response = client.post("/html/flow/test_flow", json=flow_request_data)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Sample text" in response.text  # Check if the response contains the expected text


def test_get_runs(client):
    response = client.get("/runs")
    assert response.status_code == 200
    assert "text/html" not in response.headers["content-type"]
