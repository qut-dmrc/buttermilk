import pytest

from buttermilk.llms import CHEAP_CHAT_MODELS, MULTIMODAL_MODELS
import pytest

from buttermilk._core.runner_types import MediaObj, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.bm import BM
from buttermilk.llms import CHEAP_CHAT_MODELS, CHATMODELS
from buttermilk.utils.media import download_and_convert

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
    agent = LC(name="testframer", 
                      parameters={"template": "frames"})
    return agent


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
async def test_frames_cheap(framer,  example_coal, bm: BM, model):
    framer.parameters["model"] = model
    flow = Flow(source="testing", steps=[framer])
    async for result in flow.run_flows(flow_id="testflow", record=example_coal, run_info=bm.run_info):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert not result.error
        pass

@pytest.mark.anyio
@pytest.mark.parametrize("model", CHATMODELS)
async def test_frames_text(framer,  example_coal, bm: BM, model):
    framer.parameters["model"] = model
    flow = Flow(source="testing", steps=[framer])
    async for result in flow.run_flows(flow_id="testflow", record=example_coal, run_info=bm.run_info):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert not result.error
        pass

@pytest.mark.anyio
@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
async def test_framing_video(framer, model, bm, link_to_video_gcp):
    framer.parameters["model"] = model
    flow = Flow(source="testing", steps=[framer])

    record = RecordInfo(components=link_to_video_gcp)
    async for result in flow.run_flows(flow_id="testflow", record=record, run_info=bm.run_info):
        assert result
        assert isinstance(result.record, RecordInfo)
        assert not result.error
        pass

@pytest.fixture(scope="session")
def link_to_video_gcp() -> MediaObj:
    obj = MediaObj(uri="gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4", mime="video/mp4")
    return obj
