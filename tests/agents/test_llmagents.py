import pytest

from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.llms import CHATMODELS
from buttermilk._core.types import Record
from buttermilk.agents.llm import LLMAgent
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
from buttermilk.agents.judge import Judge


@pytest.fixture
def request_paris() -> AgentInput:
    return AgentInput(prompt="What is the capital of France?")


@pytest.fixture
def request_chief(fight_no_more_forever) -> AgentInput:
    return AgentInput(records=[fight_no_more_forever])


@pytest.mark.anyio
async def test_autogen_clients(model_name, request_paris):
    """Test direct invocation of LLM Agent."""
    agent = LLMAgent(name="assistant1", description="test", model=model_name)

    response = await agent(message=request_paris)
    print(response)
    assert response


@pytest.mark.anyio
async def test_autogen_judge(model_name, request_chief):
    """Test direct invocation of LLM Agent with record."""
    agent = Judge(name="judge", description="test", model=model_name, parameters={})

    result = await agent._process(inputs=request_chief)
    assert result
    assert isinstance(result, AgentOutput)
    assert not result.error
    assert isinstance(result.outputs, Record)
    assert result.outputs and isinstance(result.outputs, dict)
    assert result.outputs["prediction"] is False
    assert len(result.outputs["reasons"]) > 0
    assert "joseph" in " ".join(result.outputs["reasons"]).lower()
