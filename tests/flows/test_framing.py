import pytest

from buttermilk._core import BM, logger
from buttermilk._core.llms import CHEAP_CHAT_MODELS, MULTIMODAL_MODELS
from buttermilk._core.log import logger  # noqa
from buttermilk._core.types import Record, RunRequest  # Import RunRequest
from buttermilk.agents.llm import LLMAgent as LC
from buttermilk.runner.flowrunner import FlowRunner  # Import Flow


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
    flow = FlowRunner(source="testing", steps=[framer])
    run_request = RunRequest(  # Replaced Job with RunRequest
        ui_type="testing",  # Mapped source to ui_type
        flow="testflow",  # Mapped flow_id to flow
        records=[text_record],  # Mapped record to records list
        run_info=bm.run_info,
        session_id="test_session",  # Added required session_id
    )

    async for result in flow.run_flows(run_request=run_request):  # Pass run_request
        assert result
        assert isinstance(result.record, Record)
        assert not result.error


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
async def test_frames_article(framer, news_record, bm: BM, model):
    framer.parameters["model"] = model
    flow = FlowRunner(source="testing", steps=[framer])
    run_request = RunRequest(  # Replaced Job with RunRequest
        ui_type="testing",  # Mapped source to ui_type
        flow="testflow",  # Mapped flow_id to flow
        records=[news_record],  # Mapped record to records list
        run_info=bm.run_info,
        session_id="test_session",  # Added required session_id
    )
    async for result in flow.run_flows(run_request=run_request):  # Pass run_request
        assert result
        assert isinstance(result.record, Record)
        assert not result.error


@pytest.mark.anyio
@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
async def test_framing_video(framer, model, bm, link_to_video_gcp):
    framer.parameters["model"] = model
    flow = FlowRunner(source="testing", steps=[framer])

    record = link_to_video_gcp  # Use the video record directly
    run_request = RunRequest(ui_type="testing", source="testing", flow="testflow", records=[record], run_info=bm.run_info, session_id="test_session")  # Added ui_type, Replaced Job with RunRequest and mapped args
    async for result in flow.run_flows(run_request=run_request):  # Pass run_request
        assert result
        assert isinstance(result.record, Record)
        assert not result.error


@pytest.fixture(scope="session")
def link_to_video_gcp() -> Record:
    obj = Record(uri="gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4", content="Video content", mime="video/mp4")
    return obj
