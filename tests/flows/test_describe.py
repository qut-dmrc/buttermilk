import pytest

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.describer import Describer
from buttermilk.bm import BM
from buttermilk.llms import MULTIMODAL_MODELS
from buttermilk.runner.flow import Flow
from buttermilk.runner.helpers import parse_flow_vars
from buttermilk.utils.media import download_and_convert


@pytest.fixture(params=MULTIMODAL_MODELS)
def flow_describer(request):
    agent = Describer(
        name="testdescriber",
        parameters={"template": "describe", "model": request.param},
        inputs={"record": "record"},
        outputs={"record": "record"},
    )
    return Flow(source="testing", steps=[agent])


@pytest.mark.anyio
async def test_run_flow_describe(flow_describer, image_bytes, bm: BM):
    record = RecordInfo(
        components=await download_and_convert(image_bytes, "image/jpeg")
    )
    async for result in flow_describer.run_flows(
        flow_id="testflow", record=record, run_info=bm.run_info
    ):
        assert result
        assert not result.error
        assert isinstance(result.record, RecordInfo)
        assert "painting" in str(result.record.text).lower()
        assert "night watch" in str(result.record.title).lower()


@pytest.mark.anyio
async def test_run_flow_describe_no_media(flow_describer, lady_macbeth: RecordInfo, bm: BM):
    async for result in flow_describer.run_flows(flow_id="testflow", record=lady_macbeth, run_info=bm.run_info):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert not result.outputs


def test_find_record(bm, image_bytes):
    record = RecordInfo(components=[image_bytes])
    job = Job(
        flow_id="test",
        record=record,
        source="test",
        inputs={"record": "record"},
        parameters={"template": "describe"},
        run_info=bm.run_info,
    )
    job.inputs = parse_flow_vars(job.inputs, job=job)
    assert job.inputs["record"] == record
    assert job.inputs["record"].media[0] == image_bytes
