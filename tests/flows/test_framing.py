import pytest

from buttermilk.llms import CHEAP_CHAT_MODELS, MULTIMODAL_MODELS

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_frames_text(bm, example_coal, model):
    from buttermilk.flows.extract import Analyst
    flow = Analyst(template="frames.prompty", model=model)
    output = flow(content=example_coal)
    pass
    assert output

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_framing_climate(bm, example_coal, model):
    from buttermilk.flows.extract import Analyst
    flow = Analyst(template="frames.prompty", model=model)
    output = flow(content=example_coal)
    pass
    assert output

@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_framing_video(model, link_to_video):
    from buttermilk.flows.video.video import Analyst
    flow = Analyst(template="frames_system.jinja2", model='gemini15pro')
    output = flow(content='see video', media_attachment_uri=link_to_video)
    pass
    assert output

@pytest.fixture(scope="session")
def link_to_video():
    return "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"
