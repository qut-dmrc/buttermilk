import pytest

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.llms import MULTIMODAL_MODELS
from buttermilk.utils.media import download_and_convert


@pytest.fixture
def describer():
    agent = LC(name="testdescriber", template="describe", inputs={"record": "record"})
    return agent


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
async def test_painting(bm, describer, model, image_bytes):
    record = RecordInfo(media=[await download_and_convert(image_bytes, "image/jpg")])
    job = Job(
        flow_id="test",
        record=record,
        source="test",
        inputs={"record": "record"},
        parameters={"template": "describe", "model": model},
        run_info=bm._run_metadata,
    )
    result = describer.run(job=job)
    assert result
