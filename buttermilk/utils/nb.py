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
from IPython.display import display
from rich import print

from dotenv import load_dotenv
from buttermilk import BM, bm, get_bm, set_bm, logger  # noqa

console = Console()
print = console.print

import hydra
from hydra import compose, initialize_config_dir
import nest_asyncio

# Apply nest_asyncio to handle potential event loop issues in notebooks
nest_asyncio.apply()

def graph_defaults():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (10, 8)
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    plt.rcParams["font.size"] = 14
    print("Notebook graphing defaults applied")


def nb_init(job: str, project: str = None, overrides: list[str] = [], config_dir: str = None) -> BM:
    """Simple one-liner initialization for Buttermilk.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        overrides: List of Hydra override strings for customization
        config_dir: Path to configuration directory (defaults to packaged config)

    Returns:
        bm: the Buttermilk instance

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.

    Example:
        >>> from buttermilk import init, nb_init, logger  # Always use global logger import
        >>> # First session - project required
        >>> bm = nb_init(job="my_analysis_notebook", project="my_project")
        >>> logger.info("Analysis started")  # Session context automatically included
        >>>
        >>> # Subsequent sessions - project optional (inherits from execution context)
        >>> bm2 = nb_init(job="data_visualization")  # Uses "my_project"

        # To use your own config directory:
        >>> bm = nb_init(job="my_analysis", project="my_project", config_dir="./conf")
    """
    bm, _ = nb_init_with_config(job=job, project=project, overrides=overrides, config_dir=config_dir)

    return bm


def nb_init_with_config(job: str, project: str, overrides: list[str] = [], config_dir: str = None) -> tuple[BM, Any]:
    """Initialization for Buttermilk that also returns the config object.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        overrides: List of Hydra override strings for customization
        config_dir: Path to configuration directory (defaults to packaged config)

    Returns:
        Tuple of (bm, config): the Buttermilk instance and the Hydra config object

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.

    Example:
        >>> from buttermilk import init, nb_init_with_config, logger  # Always use global logger import
        >>> # First session - project required
        >>> bm, cfg = nb_init_with_config(job="my_analysis_notebook", project="my_project")
        >>> logger.info(f"Analysis started with config: {cfg}")  # Session context automatically included
        >>>
        >>> # Subsequent sessions - project optional (inherits from execution context)
        >>> bm2, cfg2 = nb_init_with_config(job="data_visualization")  # Uses "my_project"
    """
    # Use unified bootstrap function with config return
    from buttermilk._core.config_bootstrap import bootstrap_session_with_config

    try:
        load_dotenv(dotenv_path=os.path.expanduser("~/.env"))
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")

    bm, config = bootstrap_session_with_config(job=job, project=project, run_type="notebook", config_dir=config_dir, overrides=overrides)

    graph_defaults()

    return bm, config