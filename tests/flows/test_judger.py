import pytest

from buttermilk._core.runner_types import RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.bm import BM
from buttermilk.llms import CHEAP_CHAT_MODELS
from buttermilk.utils.media import download_and_convert


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def judger(request):
    agent = LC(name="testjudger", 
                      parameters={"template": "judge", "model": request.param, "criteria": "criteria_ordinary", "formatting": "json_rules"},
                      inputs={"record": "record"},
                      outputs={"record": "record"})
    return agent

@pytest.fixture
def single_step_flow(judger):
    from buttermilk.runner.flow import Flow
    return Flow(source="testing", steps=[judger])

@pytest.mark.anyio
async def test_run_flow_judge(single_step_flow,  fight_no_more_forever, bm: BM):
    async for result in single_step_flow.run_flows(flow_id="testflow", source='testing', record=fight_no_more_forever, run_info=bm.run_info):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert result.outputs.prediction is False
        assert len(result.outputs.reasons) > 0
        assert "joseph" in " ".join(result.outputs.reasons).lower()
