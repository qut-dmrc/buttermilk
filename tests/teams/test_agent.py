import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
    TypeSubscription,
)
from autogen_core.models import (
    UserMessage,
)

from buttermilk._core.agent import Agent
from buttermilk.agents.llmchat import LLMAgent
from buttermilk.bm import bm
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
def record_agent_cfg(
    request,
    model_name,
) -> Agent:
    match request.param:
        case "Judger":
            return Agent(
                agent="LLMClient",
                name=request.param,
                description="apply rules",
                parameters=dict(
                    model=model_name,
                    template="judge",
                    formatting="json_rules",
                    criteria="criteria_ordinary",
                ),
            )
        case "Owl":
            return Agent(
                agent="LLMClient",
                name=request.param,
                description="look for things",
                parameters=dict(
                    model=model_name,
                    template="owl",
                    watch="ambiguity",
                ),
            )
        case _:
            raise ValueError(f"Unknown agent type: {request.param}")


@pytest.mark.anyio
async def test_run_record_agent(
    runtime,
    record_agent_cfg,
    fight_no_more_forever,
):
    """Test agents that just take a record as input."""
    agent_id = await LLMAgent.register(
        runtime,
        DefaultTopicId().type,
        lambda: LLMAgent(
            config=record_agent_cfg,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=DefaultTopicId().type,
            agent_type=agent_id.type,
        ),
    )
    runtime.start()
    record = UserMessage(content=fight_no_more_forever.fulltext, source="testing")
    result = await runtime.send_message(
        RequestToSpeak(placeholders={"record": [record]}),
        await runtime.get("default"),
    )
    await runtime.stop_when_idle()
    assert result
