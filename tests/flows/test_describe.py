import pytest

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.describer import Describer
from buttermilk.agents.lc import LC
from buttermilk.bm import BM
from buttermilk.llms import MULTIMODAL_MODELS
from buttermilk.runner.helpers import parse_flow_vars
from buttermilk.utils.media import download_and_convert


@pytest.fixture(params=MULTIMODAL_MODELS)
def describer(request):
    agent = Describer(name="testdescriber", 
                      parameters={"template": "describe", "model": request.param},
                      inputs={"record": "record"},
                      outputs={"record": "record"})
    return agent

@pytest.fixture
def flow(describer):
    from buttermilk.runner.flow import Flow
    return Flow(source="testing", steps=[describer])

@pytest.mark.anyio
async def test_run_flow_describe(flow,  image_bytes, bm: BM):
    record = RecordInfo(media=[await download_and_convert(image_bytes, "image/jpeg")])
    async for result in flow.run_flows(flow_id="testflow", source='testing', record=record, run_info=bm._run_metadata):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert "painting" in str(result.record.description).lower()
        assert "night watch" in str(result.record.title).lower()

@pytest.mark.anyio
async def test_run_flow_describe_no_media(flow, lady_macbeth: RecordInfo, bm: BM):
    async for result in flow.run_flows(flow_id="testflow", source='testing', record=lady_macbeth, run_info=bm._run_metadata):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert not result.outputs


@pytest.mark.anyio
async def test_painting(bm, describer, image_bytes):
    record = RecordInfo(media=[await download_and_convert(image_bytes, "image/jpg")])
    job = Job(
        flow_id="test",
        record=record,
        source="test",
        inputs={"record": "record"},
        parameters={"template": "describe"},
        run_info=bm._run_metadata,
    )
    job.inputs = parse_flow_vars(job.inputs, job=job)
    result = await describer.run(job=job)
    assert result 
    assert not result.error
    assert "night watch" in str(result.outputs.model_dump()).lower()


def test_find_record(bm, image_bytes):
    record = RecordInfo(media=[image_bytes])
    job = Job(
        flow_id="test",
        record=record,
        source="test",
        inputs={"record": "record"},
        parameters={"template": "describe"},
        run_info=bm._run_metadata,
    )
    job.inputs = parse_flow_vars(job.inputs, job=job)
    assert job.inputs["record"] == record
    assert job.inputs["record"].media[0] == image_bytes