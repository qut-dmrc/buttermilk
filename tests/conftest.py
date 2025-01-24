from unittest.mock import patch

import hydra
import pytest
from pytest import MarkDecorator

from buttermilk import BM
from buttermilk.llms import CHATMODELS, MULTIMODAL_MODELS, LLMs
from buttermilk.utils.utils import read_file


@pytest.fixture
def mock_save():
    with patch("buttermilk.bm.save.save") as mock_save:
        yield mock_save


@pytest.fixture(scope="session", autouse=True)
def objs():
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    return objs


@pytest.fixture(scope="session", autouse=True)
def bm(objs) -> BM:
    return objs.bm


@pytest.fixture(scope="session", autouse=True)
def flow(objs):
    return objs.flows["test"]


@pytest.fixture(scope="session")
def logger(bm):
    return bm.logger


@pytest.fixture(scope="session")
def llms(bm: BM) -> LLMs:
    return bm.llms


@pytest.fixture(params=MULTIMODAL_MODELS)
def multimodal_llm(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture(params=CHATMODELS)
def llm(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def image_bytes() -> bytes:
    return read_file("tests/data/Rijksmuseum_(25621972346).jpg")


@pytest.fixture(scope="session")
def video_bytes(video_url: str) -> bytes:
    return read_file(video_url)


VIDEO_URIS = [
    (
        "web",
        r"https://github.com/chthomos/video-media-samples/raw/refs/heads/master/big-buck-bunny-480p-30sec.mp4",
    ),
    ("gcs", "gs://dmrc-platforms/data/tonepolice/v2IF1Kw4.mp4"),
]


@pytest.fixture(
    scope="session",
    params=[x[1] for x in VIDEO_URIS],
    ids=[x[0] for x in VIDEO_URIS],
)
def video_url(request) -> str:
    return request.param


def skipif_no_gpu(reason: str = "No GPU available") -> MarkDecorator:
    """Convenience for pytest.mark.skipif() in case no GPU is available.

    :param reason: The reason for skipping the test.
    :return: A Pytest skipif mark decorator.
    """
    from torch import cuda

    has_gpu = cuda.is_available() and cuda.device_count() > 0
    return pytest.mark.skipif(not has_gpu, reason=reason)


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run gpu and memory intensive tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        # --gpu given in cli: do not skip gpu and memory intensive tests
        return
    skipgpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skipgpu)
