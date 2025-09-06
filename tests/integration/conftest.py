import pytest
from hydra import compose, initialize

from buttermilk import create_infrastructure_from_config, get_bm, set_bm
from buttermilk._core.bm_init import BM
from buttermilk._core.llms import CHAT_MODELS, CHEAP_CHAT_MODELS, MULTIMODAL_MODELS, LLMs
from buttermilk._core.log import logger

# Loads the full hydra config from config.yaml, instead of testing.yaml


@pytest.fixture(scope="session", autouse=True)
def conf():
    """Hydra config fixture."""

    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="testing")

    # Keep as DictConfig for infrastructure, but resolve for other uses
    return cfg


# Create infrastructure manager from test configuration
@pytest.fixture(scope="session")
def infrastructure(conf):
    """Provide the Infrastructure instance created from config."""

    # Use new infrastructure configuration
    if "infrastructure" in conf:
        # Pass the DictConfig directly to preserve instantiation capability
        infrastructure = create_infrastructure_from_config(conf["infrastructure"])
    else:
        raise ValueError("Test configuration must contain 'infrastructure' configuration.")

    # Initialize infrastructure components
    infrastructure.initialize_components()

    # Create a test session-scoped BM and set as singleton for backward compatibility
    test_bm = infrastructure.create_session_bm(name="buttermilk", job="testing", platform="local")

    set_bm(test_bm)

    return infrastructure


@pytest.fixture(scope="session", autouse=True)
def bm(infrastructure):
    """Provide the real BM instance for integration tests."""
    return get_bm()


@pytest.fixture(scope="session")
def logger_fixture(bm):
    """Provide the logger from the real BM instance."""
    return logger


@pytest.fixture(scope="session")
def llms(bm: BM) -> LLMs:
    return bm.llms


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def model_name(request) -> str:
    return request.param


@pytest.fixture(params=MULTIMODAL_MODELS)
def llm_multimodal(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def llm(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture(params=CHAT_MODELS)
def llm_expensive(request, bm: BM):
    return bm.llms[request.param]
