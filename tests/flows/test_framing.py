import pytest

from buttermilk._core.runner_types import MediaObj, Record
from buttermilk.llms import CHEAP_CHAT_MODELS, MULTIMODAL_MODELS
from buttermilk.runner.flow import Flow


def param_model(request):
    return {"model": request.param}


@pytest.fixture(params=MULTIMODAL_MODELS)
def param_model_multimodal(request):
    return {"model": request.param}


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def param_model_cheap(request):
    return {"model": request.param}


@pytest.fixture
def framer():
    agent = LC(agent_id="testframer", parameters={"template": "frames"})
    return agent


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
async def test_frames_text(framer, text_record, bm: BM, model):
    framer.parameters["model"] = model
    flow = Flow(source="testing", steps=[framer])
    job = Job(
        source="testing",
        flow_id="testflow",
        record=text_record,
        run_info=bm.run_info,
    )

    async for result in flow.run_flows(job=job):
        assert result
        assert isinstance(result.record, Record)
        assert not result.error


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
async def test_frames_article(framer, news_record, bm: BM, model):
    framer.parameters["model"] = model
    flow = Flow(source="testing", steps=[framer])
    job = Job(
        source="testing",
        flow_id="testflow",
        record=news_record,
        run_info=bm.run_info,
    )
    async for result in flow.run_flows(job=job):
        assert result
        assert isinstance(result.record, Record)
        assert not result.error


@pytest.mark.anyio
@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
async def test_framing_video(framer, model, bm, link_to_video_gcp):
    framer.parameters["model"] = model
    flow = Flow(source="testing", steps=[framer])

    record = Record(data=link_to_video_gcp)
    job = Job(source="testing", flow_id="testflow", record=record, run_info=bm.run_info)
    async for result in flow.run_flows(job=job):
        assert result
        assert isinstance(result.record, Record)
        assert not result.error


@pytest.fixture(scope="session")
def link_to_video_gcp() -> MediaObj:
    obj = MediaObj(uri="gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4", mime="video/mp4")
    return obj
