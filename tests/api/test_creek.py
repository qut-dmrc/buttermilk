import pytest
from fastapi.testclient import TestClient
from buttermilk.agents.testagent import TestAgent
from buttermilk.api.flow import app, FlowRequest
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.runner.creek import Creek


client = TestClient(app)

@pytest.fixture
def flow_request_data():
    return {
        "flow": "judge_full",
        "text": """An image depicting a caricature of a Jewish man with an exaggerated hooked nose and a Star of David marked with "Jude" (resembling Holocaust-era badges), holding a music box labeled "media." A monkey labeled "BLM" sits on the man's shoulder.""",
    }
    # "video": "gs://dmrc-platforms/data/tonepolice/v2IF1Kw4.mp4",

@pytest.fixture
def creek(bm):
    return Creek(source="test", steps=[TestAgent()])

text =  """An image depicting a caricature of a Jewish man with an exaggerated hooked nose and a Star of David marked with "Jude" (resembling Holocaust-era badges), holding a music box labeled "media." A monkey labeled "BLM" sits on the man's shoulder."""

async def test_run_flow(creek, flow_request_data):
    async for response in creek.run(record=RecordInfo(text=text)):
        print(response)
        pass
