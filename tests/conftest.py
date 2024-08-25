import pytest

from buttermilk.config import Config

@pytest.fixture(scope="session")
def config():
    yield Config()