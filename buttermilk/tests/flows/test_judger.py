import pytest

from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.runner_types import Record
from buttermilk.agents.llm import LLMAgent
from buttermilk._core.llms import CHATMODELS
from buttermilk.runner.flow import Flow


@pytest.fixture(params=CHATMODELS)
def judger(request):
    return LLMAgent(
        agent_id="testjudger",
        parameters={
            "template": "judge",
            "model": request.param,
            "criteria": "criteria_ordinary",
            "formatting": "json_rules",
        },
        inputs={"record": "record"},
        save={
            "type": "bq",
            "dataset": "prosocial-443205.toxicity.flow",
            "db_schema": "flow.json",
        },
    )


@pytest.fixture
def single_step_flow(judger):
    return Flow(source="testing", steps=[judger])


@pytest.mark.anyio
async def test_run_flow_judge(single_step_flow, fight_no_more_forever, bm):
    agent_input = AgentInput(
        source="testing",
        flow_id="testflow",
        record=fight_no_more_forever,
        run_info=bm.run_info,
    )
    async for result in single_step_flow.run_flows(agent_input=agent_input):
        assert result
        assert isinstance(result, AgentOutput)
        assert not result.error
        assert isinstance(result.record, Record)
        assert result.outputs and isinstance(result.outputs, dict)
        assert result.outputs["prediction"] is False
        assert len(result.outputs["reasons"]) > 0
        assert "joseph" in " ".join(result.outputs["reasons"]).lower()
