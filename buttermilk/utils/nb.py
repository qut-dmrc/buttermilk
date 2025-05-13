# Quickly initialise a notebook

import os
from typing import Any

# flake8: noqa

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import seaborn as sns
from cmap import Colormap
from rich.console import Console

console = Console()
print = console.print

import hydra
from hydra import compose, initialize_config_dir
import nest_asyncio

# Apply nest_asyncio to handle potential event loop issues in notebooks
nest_asyncio.apply()


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
    bm.setup_instance()
    logger = bm.logger
    logger.info(
        f"Starting interactive run for {bm.run_info.name} job {bm.run_info.job} in notebook",
    )

    return objs


def graph_defaults():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (10, 8)
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    plt.rcParams["font.size"] = 14


graph_defaults()
