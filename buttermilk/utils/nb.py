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

from buttermilk import BM, bm, logger  # noqa

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


def nb_init(job: str, project: str = None, overrides: list[str] = [], config_dir: str = None, config_name: str = "config") -> BM:
    """Simple one-liner initialization for Buttermilk.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        overrides: List of Hydra override strings for customization
        config_dir: Path to configuration directory (defaults to packaged config)
        config_name: Name of the configuration file to load (without .yaml extension)

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
    # Use unified bootstrap function with config return
    from buttermilk import init

    # Add notebook run type to overrides
    nb_overrides = overrides + ["run=notebook"]

    bm = init(job=job, project_name=project, config_dir=config_dir, config_name=config_name, overrides=nb_overrides)

    graph_defaults()

    return bm