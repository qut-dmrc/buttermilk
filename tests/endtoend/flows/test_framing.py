import pytest

from buttermilk import BM, logger
from buttermilk._core.config import AgentVariants
from buttermilk._core.llms import CHEAP_CHAT_MODELS, MULTIMODAL_MODELS
from buttermilk._core.log import logger  # noqa
from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk._core.types import Record, RunRequest
from buttermilk.agents.llm import LLMAgent as LC
from buttermilk.runner.flowrunner import FlowRunner


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
    agent = LC(agent_id="testframer", parameters={"template": "frames", "model": "haiku0"})
    return agent


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
async def test_frames_text(framer, text_record, real_bm: BM, model):
    framer.parameters["model"] = model
    agent_variants = AgentVariants(
        agent_obj="buttermilk.agents.llm.LLMAgent",
        role="FRAMER",
        variants={"default": [framer.parameters]},
    )
    orchestrator = OrchestratorProtocol(
        agents={"FRAMER": agent_variants},
        orchestrator="buttermilk.orchestrators.groupchat.GroupChatOrchestrator",
    )
    flow = FlowRunner(source="testing", flows={"testframer": orchestrator})

    results = []

    async def result_callback(result):
        results.append(result)

    run_request = RunRequest(
        flow="testframer",
        parameters={"records": [text_record]},
        session_info=real_bm.session_info,
        session_id="test_session",
        callback_to_ui=result_callback,
    )

    await flow.run_flow(run_request=run_request)

    assert len(results) > 0
    # Add more assertions on the results


@pytest.mark.anyio
@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
async def test_frames_article(framer, news_record, real_bm: BM, model):
    framer.parameters["model"] = model
    agent_variants = AgentVariants(
        agent_obj="buttermilk.agents.llm.LLMAgent",
        role="FRAMER",
        variants={"default": [framer.parameters]},
    )
    orchestrator = OrchestratorProtocol(
        agents={"FRAMER": agent_variants},
        orchestrator="buttermilk.orchestrators.groupchat.GroupChatOrchestrator",
    )
    flow = FlowRunner(source="testing", flows={"testframer": orchestrator})

    results = []

    async def result_callback(result):
        results.append(result)

    run_request = RunRequest(
        flow="testframer",
        parameters={"records": [news_record]},
        session_info=real_bm.session_info,
        session_id="test_session",
        callback_to_ui=result_callback,
    )
    await flow.run_flow(run_request=run_request)

    assert len(results) > 0


@pytest.mark.anyio
@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
async def test_framing_video(framer, model, real_bm, link_to_video_gcp):
    framer.parameters["model"] = model
    agent_variants = AgentVariants(
        agent_obj="buttermilk.agents.llm.LLMAgent",
        role="FRAMER",
        variants={"default": [framer.parameters]},
    )
    orchestrator = OrchestratorProtocol(
        agents={"FRAMER": agent_variants},
        orchestrator="buttermilk.orchestrators.groupchat.GroupChatOrchestrator",
    )
    flow = FlowRunner(source="testing", flows={"testframer": orchestrator})

    record = link_to_video_gcp

    results = []

    async def result_callback(result):
        results.append(result)

    run_request = RunRequest(
        flow="testframer",
        parameters={"records": [record]},
        session_info=real_bm.session_info,
        session_id="test_session",
        callback_to_ui=result_callback,
    )
    await flow.run_flow(run_request=run_request)

    assert len(results) > 0


@pytest.fixture(scope="session")
def link_to_video_gcp() -> Record:
    obj = Record(
        metadata={"uri": "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"},
        content="Video content",
        mime="video/mp4",
    )
    return obj
