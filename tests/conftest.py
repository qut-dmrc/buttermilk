from typing import Any, Generator
import pytest

from pytest import MarkDecorator

@pytest.fixture(scope="session")
def bm() -> Generator["BM", Any, None]:
    from buttermilk.bm import BM
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="conf", ):
        cfg = compose(config_name="config")
    yield BM(cfg=cfg)

@pytest.fixture(scope="session")
def logger(BM):
    return BM.logger

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
        "--gpu", action="store_true", default=False, help="run gpu and memory intensive tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        # --gpu given in cli: do not skip gpu and memory intensive tests
        return
    skipgpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skipgpu)