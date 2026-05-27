import pytest

from buttermilk._core.runtime_types import DefaultTopicId
from buttermilk._core.messages import (
    UserMessage,
)

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput
from buttermilk.agents.llm import LLMAgent


@pytest.fixture(params=["Judger", "Owl"], scope="function")
def record_agent_cfg(
    request,
    real_model_name,
) -> Agent:
    match request.param:
        case "Judger":
            return LLMAgent(
                id=request.param,
                role="judge",
                name="judger",
                description="apply rules",
                parameters=dict(
                    model=real_model_name,
                    template="judge",
                    formatting="json_rules",
                    criteria="criteria_ordinary",
                ),
                session_id="test_session",
            )
        case "Owl":
            return LLMAgent(
                role="owl",
                name="owl",
                description="look for things",
                parameters=dict(
                    model=real_model_name,
                    template="owl",
                    watch="ambiguity",
                ),
                session_id="test_session",
            )
        case _:
            raise ValueError(f"Unknown agent type: {request.param}")


@pytest.mark.anyio
@pytest.mark.skip(reason="Test uses outdated AgentInput API that no longer accepts 'records' parameter")
async def test_run_record_agent(
    record_agent_cfg,
    fight_no_more_forever,
):
    """Test agents that just take a record as input.

    NOTE: This test is outdated - AgentInput no longer accepts 'records' as a parameter.
    Need to update to use current AgentInput API with 'inputs' dict.
    """
    pass


@pytest.mark.anyio
async def test_judger(runtime, fight_no_more_forever):
    pass
