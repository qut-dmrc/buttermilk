import pytest
from autogen_core import (
    DefaultTopicId,
    SingleThreadedAgentRuntime,
    TypeSubscription,
)
from autogen_core.models import (
    UserMessage,
)

from buttermilk._core import AgentInput
from buttermilk._core.agent import Agent
from buttermilk.agents.llm import LLMAgent


@pytest.fixture
def runtime():
    return SingleThreadedAgentRuntime()


@pytest.fixture(params=["Judger", "Owl"], scope="function")
def record_agent_cfg(
    request,
    model_name,
) -> Agent:
    match request.param:
        case "Judger":
            return LLMAgent(
                id=request.param,
                role="judge",
                name="judger",
                description="apply rules",
                parameters=dict(
                    model=model_name,
                    template="judge",
                    formatting="json_rules",
                    criteria="criteria_ordinary",
                ),
            )
        case "Owl":
            return LLMAgent(
                role="owl",
                name="owl",
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
    record = UserMessage(content=fight_no_more_forever.fulltext, source="testing")
    result = await runtime.send_message(
        AgentInput(records=[record]),
        await runtime.get("default"),
    )
    await runtime.stop_when_idle()
    assert result


@pytest.mark.anyio
async def test_judger(runtime, fight_no_more_forever):
    pass
