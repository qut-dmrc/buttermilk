import pytest

from buttermilk._core.llms import MULTIMODAL_MODELS
from buttermilk._core.types import (
    Record,
    RunRequest,  # Import RunRequest
)
from buttermilk.agents.describer import Describer
from buttermilk.utils.media import download_and_convert


@pytest.fixture(params=MULTIMODAL_MODELS)
def flow_describer(request):
    return Describer(
        agent_id="testdescriber",
        parameters={
            "template": "describe",
            "download_if_necessary": True,
            "model": request.param,
        },
        inputs={"record": "record"},
        outputs={"record": "record"},
        session_id="test_session",  # Add required session_id
    )


@pytest.mark.anyio
async def test_run_flow_describe_only(flow_describer, image_bytes, real_bm):
    record = await download_and_convert(image_bytes, "image/jpeg")
    # Create a RunRequest instance
    run_request = RunRequest(
        flow="testflow",  # Assuming a flow_id like "testflow"
        inputs=dict(record=record),
    )
    async for result in flow_describer.run_flows(
        run_request=run_request
    ):  # Pass run_request
        assert result
        assert not result.error
        assert isinstance(result.record, Record)
        assert "painting" in str(result.record).lower()
        assert "night watch" in str(result.record.title).lower()
