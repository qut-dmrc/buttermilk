import asyncio

import pytest
from hydra import compose, initialize

from buttermilk import get_bm, set_bm
from buttermilk._core.bm_init import BM
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.llms import CHAT_MODELS, CHEAP_CHAT_MODELS, MULTIMODAL_MODELS, LLMs
from buttermilk._core.log import logger

# Loads the full hydra config from config.yaml, instead of testing.yaml


@pytest.fixture(scope="session", autouse=True)
def conf():
    """Hydra config fixture."""

    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Keep as DictConfig for infrastructure, but resolve for other uses
    return cfg


@pytest.fixture(scope="session", autouse=True)
def bootstrapper(conf):
    """ConfigurationBootstrapper fixture."""
    return ConfigurationBootstrapper(config=conf)


# Create infrastructure manager from test configuration with proper ExecutionContext
@pytest.fixture(scope="session", autouse=True)
def infrastructure(bootstrapper, conf):
    """Provide the Infrastructure instance created from config with ExecutionContext."""

    # Create infrastructure manager first
    infrastructure = bootstrapper.get_infrastructure_manager()
    
    # Create a test session-scoped BM and set as singleton BEFORE ExecutionContext initialization
    test_bm = infrastructure.create_session_bm(name="buttermilk", job="testing", platform="local")
    set_bm(test_bm)
    
    # Now bootstrap full context (ExecutionContext + tracing initialization)
    # BM singleton is available, so tracing can initialize properly
    execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

    return infrastructure


@pytest.fixture(scope="session", autouse=True)
def real_bm(infrastructure, conf):
    """Provide the real BM instance for integration tests."""
    return get_bm()


@pytest.fixture(scope="session")
def real_logger(real_bm: BM):
    """Provide the logger from the real BM instance."""
    return logger


@pytest.fixture(scope="session")
def llms(real_bm: BM) -> LLMs:
    return real_bm.llms


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def model_name(request) -> str:
    return request.param


@pytest.fixture(params=MULTIMODAL_MODELS)
def llm_multimodal(request, real_bm: BM):
    return real_bm.llms[request.param]


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def llm(request, real_bm: BM):
    return real_bm.llms[request.param]


@pytest.fixture(params=CHAT_MODELS)
def llm_expensive(request, real_bm: BM):
    return real_bm.llms[request.param]
