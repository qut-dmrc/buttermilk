from typing import Any, Generator
import pytest

from buttermilk.buttermilk import BM

@pytest.fixture(scope="session")
def bm() -> Generator[BM, Any, None]:
    yield BM()