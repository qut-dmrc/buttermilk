# Quickly initialise a notebook

import os
from typing import Any
from hydra import initialize, initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf
from rich import print as rprint

# Configuration files are stored in the local directory, and
# options can be passed in at initialization.
def init(overrides: list[str]=[], path: str = None) -> Any:
    if not path: 
        # Must be absolute
        path = os.getcwd() + "/conf"

    with initialize_config_dir(version_base=None, config_dir=path):
        cfg = compose(config_name="config", overrides=overrides)

    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    logger = bm.logger
    logger.info(f"Starting interactive run for {bm.cfg.name} job {bm.cfg.job} in notebook")

    # print config details
    rprint("\nConfiguration:")
    rprint(OmegaConf.to_yaml(cfg))

    return objs