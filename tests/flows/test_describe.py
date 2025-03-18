import pytest

from buttermilk._core.runner_types import  Record
from buttermilk.agents.describer import Describer
from buttermilk.bm import bm
from buttermilk.llms import MULTIMODAL_MODELS
from buttermilk.runner.flow import Flow
from buttermilk.runner.helpers import parse_flow_vars
from buttermilk.utils.media import download_and_convert


@pytest.fixture(params=MULTIMODAL_MODELS)
def flow_describer(request) -> Flow:
    agent = Describer(
        agent_id="testdescriber",
        parameters={
            "template": "describe",
            "download_if_necessary": True,
            "model": request.param,
        },
        inputs={"record": "record"},
        outputs={"record": "record"},
    )
    return Flow(source="testing", steps=[agent])


@pytest.mark.anyio
async def test_run_flow_describe_only(flow_describer, image_bytes, bm: BM):
    record = await download_and_convert(image_bytes, "image/jpeg")
    job = Job(source="testing", flow_id="testflow", record=record, run_info=bm.run_info)
    async for result in flow_describer.run_flows(job=job):
        assert result
        assert not result.error
        assert isinstance(result.record, Record)
        assert "painting" in str(result.record.all_text).lower()
        assert "night watch" in str(result.record.title).lower()

