import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
    TypeSubscription,
)

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
        llm=request.param,
        name="testjudger",
    )
    return agent


@pytest.fixture(params=CHATMODELS, scope="function")
async def llm_autogen(request, bm: BM):
    return bm.llms.get_autogen_chat_client(request.param)


@pytest.mark.anyio
async def test_autogen_clients(llm_autogen):
    agent = AssistantAgent("assistant1", model_client=llm_autogen)

    response = await agent.on_messages(
        [TextMessage(content="What is the capital of France?", source="user")],
        CancellationToken(),
    )
    print(response)
    assert response


@pytest.fixture(params=[Judger, Owl])
def record_agent(request):
    return request.param


@pytest.mark.anyio
async def test_run_record_agent(
    runtime,
    record_agent,
    model_name,
    fight_no_more_forever,
):
    """Test agents that just take a record as input."""
    agent_id = await record_agent.register(
        runtime,
        DefaultTopicId().type,
        lambda: record_agent(name="assistant1", llm=model_name),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=DefaultTopicId().type,
            agent_type=agent_id.type,
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
