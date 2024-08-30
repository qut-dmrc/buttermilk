from typing import Any, Generator
import pytest

from hydra import initialize, compose
from buttermilk.buttermilk import BM

@pytest.fixture(scope="session")
def bm() -> Generator[BM, Any, None]:
    with initialize(version_base=None, config_path="conf", ):
        cfg = compose(config_name="config")
        yield BM(cfg=cfg)


