# Quickly initialise a notebook

import os
from typing import Any

import hydra
from hydra import compose, initialize_config_dir


# Configuration files are stored in the local directory, and
# options can be passed in at initialization.
def init(job: str, overrides: list[str] = [], path: str = None) -> Any:
    if not path:
        # Must be absolute
        path = os.getcwd() + "/conf"

    overrides.append("name=notebook")
    overrides.append(f"job={job}")

    with initialize_config_dir(version_base=None, config_dir=path):
        cfg = compose(config_name="config", overrides=overrides)

    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    logger = bm.logger
    logger.info(
        f"Starting interactive run for {bm.cfg.name} job {bm.cfg.job} in notebook",
    )

    return objs
