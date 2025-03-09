import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
    TypeSubscription,
)

from buttermilk.agents.llmchat import LLMAgent
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS
from buttermilk.runner.moa import RequestToSpeak


@pytest.fixture
def runtime():
    return SingleThreadedAgentRuntime()


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


@pytest.fixture(params=["Judger", "Owl"], scope="function")
def record_agent_cfg(request):
    match request.param:
        case "Judger":
            return dict(
                template="judge",
                description="apply rules",
                formatting="json_rules",
                criteria="criteria_ordinary",
            )
        case "Owl":
            return dict(
                template="owl",
                formatting="json_rules",
                description="look for things",
            )
        case _:
            raise ValueError(f"Unknown agent type: {request.param}")


@pytest.mark.anyio
async def test_run_record_agent(
    runtime,
    record_agent_cfg,
    model_name,
    fight_no_more_forever,
):
    """Test agents that just take a record as input."""
    agent_id = await LLMAgent.register(
        runtime,
        DefaultTopicId().type,
        lambda: LLMAgent(
            name="assistant1",
            model=model_name,
            step_name="test",
            **record_agent_cfg,
        ),
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
