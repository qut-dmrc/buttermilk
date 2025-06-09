from typing import Any

import pytest
from fastapi.testclient import TestClient
from hydra import compose, initialize
from omegaconf import OmegaConf

from buttermilk import BM
from buttermilk.api.flow import RunRequest, create_app
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture(scope="session")
def client():
    # Initialize with minimal configuration for testing
    with initialize(config_path="../../conf", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["run=api_clean"])
        
        # Create BM instance
        resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        bm = BM(**resolved_cfg_dict["bm"])
        
        # Create FlowRunner instance  
        flows = FlowRunner.model_validate(cfg.run)
        
        # Set BM singleton
        from buttermilk._core.dmrc import set_bm
        set_bm(bm)
        
        app = create_app(bm=bm, flows=flows)
        return TestClient(app)


@pytest.fixture
def flow_request_data():
    # Return raw dict without RunRequest serialization since ui_type is excluded
    return {
        "flow": "test_minimal",
        "model": "haiku",
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
    assert "Sample text" in response.text  # Check if the response contains the expected text


def test_get_runs(client, bm: Any):
    response = client.get("/runs")
    assert response.status_code == 200
    assert "text/html" not in response.headers["content-type"]
