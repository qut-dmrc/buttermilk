import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from buttermilk.api.flow import app, TestJob, agent

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_process_job(client):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/flow", json={"input": 4})
        assert response.status_code == 200
        assert response.json() == {"input": 4, "output": 8}

@pytest.mark.asyncio
async def test_pubsub_callback(client, monkeypatch):
    # Mock the requests.post call to the FastAPI endpoint
    async def mock_post(url, json):
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code

            async def json(self):
                return self.json_data

        return MockResponse({"input": 4, "output": 8}, 200)

    monkeypatch.setattr("requests.post", mock_post)

    # Simulate a Pub/Sub message
    class MockMessage:
        def __init__(self, data):
            self.data = data

        def ack(self):
            pass

    message = MockMessage(data=json.dumps({"task": "flow", "input": 4}).encode("utf-8"))

    from buttermilk.api.flow import callback
    callback(message)

    # Ensure the job was processed correctly
    response = await agent.process(job=TestJob(input=4))
    assert response.output == 8