from typing import Any

import pytest
from fastapi.testclient import TestClient

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.api.flow import FlowRequest, app
from buttermilk.runner.flow import Flow


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


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(
            {"q": "democrats are arseholes"},
            id="q only",
        ),
        pytest.param({}, id="no record no text"),
        pytest.param(
            {"record": lambda: example_coal, "q": "i really love coal"},
            id="q with record",
        ),  # Lambda for lazy eval
        pytest.param({"record": lambda: blm}, id="record only"),
        pytest.param({"record": lambda: video_bytes}, id="video bytes"),
        pytest.param({"record": lambda: image_bytes}, id="image bytes"),
        pytest.param({"record": lambda: video_url}, id="video url"),
    ],
)
def test_api_request_simple(
    options: dict,
    client,
):  # Inject the client
    # Resolve fixtures using lambda functions
    # resolved_req_cfg = {k: v() if callable(v) else v for k, v in req_cfg.items()}
    flow_request = FlowRequest(**options)
    response = client.post("/flow/simple", json=flow_request.model_dump(mode="json"))
    assert response.status_code == 200
    json_response = response.json()
    assert "outputs" in json_response
    assert "agent_info" in json_response


def test_run_flow(bm: Any, flow_request_data: dict[str, Any]):
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
    assert all([isinstance(x, Job) for x in response])


@pytest.fixture
def flow(bm):
    return Flow(source="test", steps=[TestAgent()])


async def test_run_flow(flow, flow_request_data):
    async for response in flow.run(record=RecordInfo(text=flow_request_data["text"])):
        print(response)
