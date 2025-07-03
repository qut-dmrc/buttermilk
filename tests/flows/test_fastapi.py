import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from buttermilk._core.contract import AgentInput, AgentTrace


class TestAgent:
    async def process(self, *, agent_input: AgentInput) -> AgentTrace:
        return AgentTrace(
            agent_id="test",
            source="test_agent",
            role="TestAgent",
            content="Processed",
            outputs={"result": 2 * agent_input.inputs["value"]},
        )


# TEST
# curl -X 'POST' 'http://127.0.0.1:8000/flow' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"inputs":{"value":4}}'
app = FastAPI()


@app.post("/flow")
async def run_flow(agent_input: AgentInput) -> AgentTrace:
    result = await agent.process(agent_input=agent_input)
    return result


agent = TestAgent()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.anyio
async def test_run_flow(client: TestClient):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/flow",
            json={
                "agent_id": "test_agent",
                "role": "TestAgent",
                "inputs": {"value": 4},
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            "agent_id": "test_agent",
            "role": "TestAgent",
            "content": "Processed",
            "outputs": {"result": 8},
        }


@pytest.mark.skip(reason="callback function no longer exists in buttermilk.api.flow")
@pytest.mark.anyio
async def test_pubsub_callback(client: TestClient, monkeypatch):
    async def mock_post(url: str, json: dict):
        class MockResponse:
            def __init__(self, json_data: dict, status_code: int):
                self.json_data = json_data
                self.status_code = status_code

            async def json(self) -> dict:
                return self.json_data

        return MockResponse(
            {
                "agent_id": "test_agent",
                "role": "TestAgent",
                "content": "Processed",
                "outputs": {"result": 8},
            },
            200,
        )

    monkeypatch.setattr("requests.post", mock_post)

    class MockMessage:
        def __init__(self, data: bytes):
            self.data = data

        def ack(self) -> None:
            pass

    message = MockMessage(
        data=json.dumps(
            {
                "task": "flow",
                "agent_id": "test_agent",
                "role": "TestAgent",
                "inputs": {"value": 4},
            },
        ).encode("utf-8"),
    )
    from buttermilk.api.flow import callback

    callback(message)

    agent_input = AgentInput(
        role="test_agent",
        source="TestAgent",
        inputs={"value": 4},
    )
    response = await agent.process(agent_input=agent_input)
    assert response.outputs["result"] == 8
