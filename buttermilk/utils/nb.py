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

from buttermilk import BM, bm, get_bm, set_bm, logger  # noqa

console = Console()
print = console.print

import hydra
from hydra import compose, initialize_config_dir
import nest_asyncio

# Apply nest_asyncio to handle potential event loop issues in notebooks
nest_asyncio.apply()


# Configuration files are stored in the local directory, and
# options can be passed in at initialization.
def nb_init(job: str, project: str = None, overrides: list[str] = [], config_dir: str = None) -> Any:
    """Initialize Buttermilk for notebook use with simple interface.

    This function uses the new ConfigurationBootstrapper architecture internally
    but provides a simple interface for researchers.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        overrides: List of Hydra override strings for customization
        config_dir: Path to configuration directory (defaults to packaged config)

    Returns:
        Configuration object with .bm attribute containing the Buttermilk instance

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    import asyncio
    from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
    from buttermilk import set_bm

    if not config_dir:
        # Default to packaged config directory
        config_dir = Path(__file__).parent.parent.resolve() / "conf"
        config_dir = config_dir.as_posix()

    # Add notebook-specific overrides
    notebook_overrides = overrides.copy()
    notebook_overrides.append("++run=notebook")
    notebook_overrides.append(f"++run.job={job}")
    
    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=config_dir, overrides=notebook_overrides)
    
    try:
        # Bootstrap full context and session
        _, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

        # Validate and set project name using ExecutionContext
        validated_project = infrastructure.validate_and_set_project(project)

        # Create session BM instance with validated project
        bm = asyncio.run(
            bootstrapper.bootstrap_session_context(
                name=validated_project,
                job=job,
                infrastructure=infrastructure
            )
        )
        
        # Set the singleton BM instance
        set_bm(bm)
        
        logger.info(
            f"Starting interactive run for {bm.session_info.project_name} job {bm.session_info.job} in notebook"
        )
        
        # Create a backwards-compatible object structure
        class NotebookObjects:
            def __init__(self, bm, config):
                self.bm = bm
                self.config = config
                self.logger = logger
                
        config = bootstrapper.get_configuration()
        return NotebookObjects(bm, config)
        
    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise


def init(job: str, project: str = None, config_dir: str = None, **kwargs) -> Any:
    """Simple one-liner initialization for notebooks.

    This is the simplest way to initialize Buttermilk for notebook use.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        config_dir: Path to configuration directory (defaults to packaged config)
        **kwargs: Additional arguments passed to nb_init()

    Returns:
        Buttermilk instance ready to use

    Example:
        >>> from buttermilk.utils import nb
        >>> from buttermilk import logger  # Always use global logger import
        >>> # First session - project required
        >>> bm = nb.init(job="my_analysis", project="my_project")
        >>> logger.info("Analysis started")  # Session context automatically included
        >>>
        >>> # Subsequent sessions - project optional (inherits from execution context)
        >>> bm2 = nb.init(job="data_visualization")  # Uses "my_project"

        # To use your own config directory:
        >>> bm = nb.init(job="my_analysis", project="my_project", config_dir="./conf")

    Note:
        Always use `from buttermilk import logger` for logging. The logger is
        a global singleton that automatically includes session context (session_id,
        job, project_name) in all log messages once a session is initialized.

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    objs = nb_init(job=job, project=project, config_dir=config_dir, **kwargs)
    return objs.bm


def graph_defaults():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (10, 8)
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    plt.rcParams["font.size"] = 14


graph_defaults()
