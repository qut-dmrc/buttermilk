from autogen import AssistantAgent as AutoGenClient, OpenAIWrapper
from autogen_agentchat.agents import AssistantAgent
import pytest

import pytest

from autogen_core import CancellationToken
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.judger import ChatAgent
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS

from autogen_agentchat.messages import TextMessage

@pytest.fixture(params=CHATMODELS, scope="function")
def judger(request):
    agent = ChatAgent(
        model_client=request.param,
        name="testjudger",
        parameters={
            "template": "judge",
            "model": request.param,
            "criteria": "criteria_ordinary",
            "formatting": "json_rules",
        }
    )
    return agent

@pytest.mark.anyio
@pytest.mark.parametrize("name", CHATMODELS)
async def test_autogen_clients(bm, name):
    client = bm.llms.get_autogen_client(name)
    agent = AssistantAgent("assistant1", model_client=client)

    response = await agent.on_messages(
        [TextMessage(content="What is the capital of France?", source="user")], CancellationToken()
    )
    print(response)
    assert response

@pytest.mark.anyio
async def test_run_agent_judge(judger, fight_no_more_forever, bm: BM):
    result = await judger.on_messages(
        [TextMessage(content=fight_no_more_forever.fulltext, source="user")], CancellationToken()
    )
    assert result

        # assert isinstance(result, Job)
        # assert not result.error
        # assert isinstance(result.record, RecordInfo)
        # assert result.outputs and isinstance(result.outputs, dict)
        # assert result.outputs["prediction"] is False
        # assert len(result.outputs["reasons"]) > 0
        # assert "joseph" in " ".join(result.outputs["reasons"]).lower()
