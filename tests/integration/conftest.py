import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from buttermilk._core.bm_init import BM

# Loads the full hydra config from config.yaml, instead of testing.yaml


@pytest.fixture(scope="session", autouse=True)
def conf():
    """Hydra config fixture."""
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config")

    try:
        resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        # Initialize the global Buttermilk instance (bm) with its configuration section
        if "bm" not in resolved_cfg_dict or not isinstance(resolved_cfg_dict["bm"], dict):
            raise ValueError("Hydra configuration must contain a 'bm' dictionary for Buttermilk initialization.")
        bm = BM(**resolved_cfg_dict["bm"])  # type: ignore # Assuming dict matches BM fields
        # Set the singleton BM instance
        from buttermilk._core.dmrc import set_bm

        set_bm(bm)  # Set the Buttermilk instance using the singleton pattern

    except Exception as e:
        print(f"Error with test configuration, cannot create BM instance: {e}")
        raise

    return cfg
