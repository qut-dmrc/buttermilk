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
def nb_init(job: str, name: str = None, overrides: list[str] = [], path: str = None) -> Any:
    """Initialize Buttermilk for notebook use with simple interface.
    
    This function uses the new ConfigurationBootstrapper architecture internally
    but provides a simple interface for researchers.
    
    Args:
        job: Name for the specific job or task
        name: Name for the session or project (optional, extracted from overrides if not provided)
        overrides: List of Hydra override strings for customization
        path: Path to configuration directory (defaults to ../../conf from this file)
        
    Returns:
        Configuration object with .bm attribute containing the Buttermilk instance
    """
    import asyncio
    from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
    from buttermilk import set_bm
    
    # Handle backwards compatibility for name in overrides
    if name is None:
        # Look for name in overrides for backwards compatibility
        for override in overrides[:]:  # Copy list to avoid modification during iteration
            if override.startswith("name="):
                name = override.split("=", 1)[1]
                overrides.remove(override)
                break
            elif override.startswith("bm.session_info.name="):
                name = override.split("=", 1)[1] 
                overrides.remove(override)
                break
        
        # Default name if not found
        if name is None:
            name = "notebook_session"
    
    if not path:
        # Must be absolute. Get the abs path of ../../conf from the current file
        path = Path(__file__).parent.parent.resolve() / "conf"
        path = path.as_posix()

    # Add notebook-specific overrides
    notebook_overrides = overrides.copy()
    notebook_overrides.append(f"++run=notebook")
    
    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=path, overrides=notebook_overrides)
    
    try:
        # Bootstrap full context and session
        _, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())
        
        # Create session BM instance
        bm = asyncio.run(
            bootstrapper.bootstrap_session_context(
                name=name,
                job=job,
                infrastructure=infrastructure
            )
        )
        
        # Set the singleton BM instance
        set_bm(bm)
        
        logger.info(
            f"Starting interactive run for {bm.session_info.name} job {bm.session_info.job} in notebook"
        )
        
        # Create a backwards-compatible object structure
        class NotebookObjects:
            def __init__(self, bm, config):
                self.bm = bm
                self.config = config
                
        config = bootstrapper.get_configuration()
        return NotebookObjects(bm, config)
        
    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise


def init(job: str, name: str = "notebook_session", **kwargs) -> Any:
    """Simple one-liner initialization for notebooks.
    
    This is the simplest way to initialize Buttermilk for notebook use.
    
    Args:
        job: Name for the specific job or task  
        name: Name for the session or project
        **kwargs: Additional arguments passed to nb_init()
        
    Returns:
        Buttermilk instance ready to use
        
    Example:
        >>> from buttermilk.utils import nb
        >>> bm = nb.init(job="my_analysis", name="my_project")
        >>> logger = bm.logger
    """
    objs = nb_init(job=job, name=name, **kwargs)
    return objs.bm


def graph_defaults():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (10, 8)
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    plt.rcParams["font.size"] = 14


graph_defaults()
