import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken, SingleThreadedAgentRuntime, TypeSubscription

from buttermilk.agents.judger import Judger, Owl
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS
from buttermilk.runner.moa import RequestToSpeak


@pytest.fixture
def runtime():
    return SingleThreadedAgentRuntime()


@pytest.fixture(params=CHATMODELS, scope="function")
def judger(request):
    agent = Judger(
        model_client=request.param,
        name="testjudger",
        parameters={
            "template": "judge",
            "model": request.param,
            "criteria": "criteria_ordinary",
            "formatting": "json_rules",
        },
    )
    return agent


@pytest.fixture(params=CHATMODELS, scope="function")
async def llm_autogen(request, bm: BM):
    return bm.llms.get_autogen_client(request.param)


@pytest.mark.anyio
async def test_autogen_clients(llm_autogen):
    agent = AssistantAgent("assistant1", model_client=llm_autogen)

    response = await agent.on_messages(
        [TextMessage(content="What is the capital of France?", source="user")],
        CancellationToken(),
    )
    print(response)
    assert response


@pytest.mark.anyio
async def test_run_agent_judge(judger, fight_no_more_forever):
    result = await judger.on_messages(
        [TextMessage(content=fight_no_more_forever.fulltext, source="user")],
        CancellationToken(),
    )
    assert result

    # assert isinstance(result, Job)
    # assert not result.error
    # assert isinstance(result.record, RecordInfo)
    # assert result.outputs and isinstance(result.outputs, dict)
    # assert result.outputs["prediction"] is False
    # assert len(result.outputs["reasons"]) > 0
    # assert "joseph" in " ".join(result.outputs["reasons"]).lower()


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHATMODELS)
async def test_run_owl(runtime, model, fight_no_more_forever):
    agent_id = await Owl.register(
        runtime,
        "default",
        lambda: Owl(name="assistant1", llm=model),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type="default",
            agent_type="owl",
        ),
    )
    runtime.start()
    # result = await runtime.publish_message(
    #     RequestToSpeak(record=fight_no_more_forever.fulltext),
    #     "default",
    # )
    result = await runtime.send_message(
        RequestToSpeak(record=fight_no_more_forever.fulltext),
        await runtime.get("default"),
    )
    await runtime.stop_when_idle()
    assert result
