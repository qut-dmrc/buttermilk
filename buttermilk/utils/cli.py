# Simple initialization for CLI scripts

import os
import asyncio
from typing import Any
from pathlib import Path

from buttermilk import BM, logger, set_bm


def init(job: str, name: str = None, overrides: list[str] = [], path: str = None) -> BM:
    """Initialize Buttermilk for CLI script use with simple interface.
    
    This function uses the new ConfigurationBootstrapper architecture internally
    but provides a simple interface for CLI scripts.
    
    Args:
        job: Name for the specific job or task
        name: Name for the session or project (optional, defaults to script name)
        overrides: List of Hydra override strings for customization
        path: Path to configuration directory (defaults to ./conf from current working directory)
        
    Returns:
        Buttermilk instance ready to use
        
    Example:
        >>> from buttermilk.utils import cli
        >>> bm = cli.init(job="data_processing", name="my_project")
        >>> logger = bm.logger
    """
    from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
    
    # Handle default name from script filename if not provided
    if name is None:
        # Try to get script name from __main__ module or current process
        import sys
        if hasattr(sys.modules.get('__main__'), '__file__'):
            script_file = sys.modules['__main__'].__file__
            if script_file:
                name = Path(script_file).stem
            else:
                name = "cli_script"
        else:
            name = "cli_script"
    
    if not path:
        # Default to ./conf from current working directory for CLI scripts
        path = Path.cwd() / "conf"
        path = path.as_posix()

    # Add CLI-specific overrides
    cli_overrides = overrides.copy()
    cli_overrides.append(f"++run=cli")
    
    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=path, overrides=cli_overrides)
    
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
            f"Starting CLI run for {bm.session_info.name} job {bm.session_info.job}"
        )
        
        return bm
        
    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise


def init_with_config(job: str, name: str = None, overrides: list[str] = [], path: str = None) -> tuple[BM, Any]:
    """Initialize Buttermilk for CLI script use and return both BM instance and configuration.
    
    This variant returns both the BM instance and the full configuration object
    for scripts that need access to additional configuration.
    
    Args:
        job: Name for the specific job or task
        name: Name for the session or project (optional, defaults to script name)  
        overrides: List of Hydra override strings for customization
        path: Path to configuration directory (defaults to ./conf from current working directory)
        
    Returns:
        Tuple of (Buttermilk instance, configuration object)
        
    Example:
        >>> from buttermilk.utils import cli
        >>> bm, config = cli.init_with_config(job="data_processing")
        >>> # Access additional config: config.my_custom_settings
    """
    from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
    
    # Handle default name from script filename if not provided
    if name is None:
        # Try to get script name from __main__ module or current process
        import sys
        if hasattr(sys.modules.get('__main__'), '__file__'):
            script_file = sys.modules['__main__'].__file__
            if script_file:
                name = Path(script_file).stem
            else:
                name = "cli_script"
        else:
            name = "cli_script"
    
    if not path:
        # Default to ./conf from current working directory for CLI scripts
        path = Path.cwd() / "conf"
        path = path.as_posix()

    # Add CLI-specific overrides
    cli_overrides = overrides.copy()
    cli_overrides.append(f"++run=cli")
    
    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=path, overrides=cli_overrides)
    
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
            f"Starting CLI run for {bm.session_info.name} job {bm.session_info.job}"
        )
        
        # Get the configuration
        config = bootstrapper.get_configuration()
        
        return bm, config
        
    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise