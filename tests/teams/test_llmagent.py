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

from buttermilk._core.contract import AgentInput
from buttermilk._core.agent import Agent
from buttermilk.agents.llm import LLMAgent
from buttermilk._core.llms import CHATMODELS, CHEAP_CHAT_MODELS


@pytest.fixture
def runtime():
    return SingleThreadedAgentRuntime()


@pytest.fixture(params=CHEAP_CHAT_MODELS, scope="function")
async def llm_autogen(request, bm):
    return bm.llms.get_autogen_chat_client(request.param)


async def request_paris() -> AgentInput:
    return AgentInput(
        role="user",
        messages=[TextMessage(content="What is the capital of France?", source="user")],
        topic_id=DefaultTopicId,
    )


@pytest.mark.anyio
async def test_autogen_clients(llm_autogen):
    agent = LLMAgent(name="assistant1", role="test", description="test", model_client=llm_autogen)

    response = await agent.on_messages(
        [TextMessage(content="What is the capital of France?", source="user")],
        CancellationToken(),
    )
    print(response)
    assert response
