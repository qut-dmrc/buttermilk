import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from buttermilk import create_infrastructure_from_config, set_bm
from buttermilk._core.bm_init import BM
from buttermilk._core.llms import CHAT_MODELS, CHEAP_CHAT_MODELS, MULTIMODAL_MODELS, LLMs

# Don't initialize BM here, we'll let the fixture handle it

# Loads the full hydra config from config.yaml, instead of testing.yaml


@pytest.fixture(scope="session", autouse=True)
def conf():
    """Hydra config fixture."""
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    try:
        resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        # Create infrastructure manager from test configuration

        # Use new infrastructure configuration if available, otherwise migrate from BM config
        if "infrastructure" in resolved_cfg_dict:
            infrastructure = create_infrastructure_from_config(resolved_cfg_dict["infrastructure"])
        elif "bm" in resolved_cfg_dict:
            infrastructure = create_infrastructure_from_config(resolved_cfg_dict["bm"])
        else:
            raise ValueError("Test configuration must contain either 'infrastructure' or 'bm' configuration.")

        # Initialize infrastructure components
        infrastructure.initialize_components()

        # Create a test session-scoped BM and set as singleton for backward compatibility
        test_bm = infrastructure.create_session_bm(name="buttermilk", job="testing", platform="local")

        set_bm(test_bm)

    except Exception as e:
        print(f"Error with test configuration, cannot create BM instance: {e}")
        raise

    return cfg


@pytest.fixture(scope="session")
def logger(bm):
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
