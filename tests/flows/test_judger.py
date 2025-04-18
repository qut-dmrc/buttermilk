import pytest

from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.llms import CHATMODELS
from buttermilk._core.types import Record
from buttermilk.agents.llm import LLMAgent


@pytest.fixture(params=CHATMODELS)
def judger(request):
    return LLMAgent(
        role="judge",
        name="judge",
        description="judger test agent",
        parameters={
            "template": "judge",
            "model": request.param,
            "criteria": "criteria_ordinary",
            "formatting": "json_rules",
        },
        inputs={},
    )


@pytest.mark.anyio
async def test_run_flow_judge(judger, fight_no_more_forever):
    agent_input = AgentInput(
        role="judge",
        records=[fight_no_more_forever],
    )
    result = await judger._run_fn(message=agent_input)
    assert result
    assert isinstance(result, AgentOutput)
    assert not result.error
    assert result.outputs and isinstance(result.outputs, dict)
    assert result.outputs["prediction"] is False
    assert len(result.outputs["reasons"]) > 0
    assert "joseph" in " ".join(result.outputs["reasons"]).lower()
