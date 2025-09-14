# Simple initialization for CLI scripts

import asyncio
import sys
from pathlib import Path
from typing import Any

from buttermilk import BM, logger, set_bm
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper


def init(job: str, project: str = None, overrides: list[str] = [], config_dir: str = None) -> BM:
    """Initialize Buttermilk for CLI script use with simple interface.

    This function uses the new ConfigurationBootstrapper architecture internally
    but provides a simple interface for CLI scripts.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        overrides: List of Hydra override strings for customization
        config_dir: Path to configuration directory (defaults to packaged config)

    Returns:
        Buttermilk instance ready to use

    Example:
        >>> from buttermilk.utils import cli
        >>> from buttermilk import logger  # Always use global logger import
        >>> # First session - project required
        >>> bm1 = cli.init(job="data_ingestion", project="my_project")
        >>> logger.info("Processing started")  # Session context automatically included
        >>>
        >>> # Subsequent sessions - project optional (inherits from execution context)
        >>> bm2 = cli.init(job="data_analysis")  # Uses "my_project"

        # To use your own config directory:
        >>> bm = cli.init(job="data_processing", project="my_project", config_dir="./conf")

    Note:
        Always use `from buttermilk import logger` for logging. The logger is
        a global singleton that automatically includes session context (session_id,
        job, project_name) in all log messages once a session is initialized.

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    if not config_dir:
        # Default to packaged config directory
        config_dir = Path(__file__).parent.parent.resolve() / "conf"
        config_dir = config_dir.as_posix()

    # Add CLI-specific overrides
    cli_overrides = overrides.copy()
    cli_overrides.append("++run=cli")
    cli_overrides.append(f"++run.job={job}")

    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=config_dir, overrides=cli_overrides)

    try:
        # Bootstrap full context and session
        _, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

        # Validate and set project name using ExecutionContext
        validated_project = infrastructure.validate_and_set_project(project)

        # Create session BM instance with validated project
        bm = asyncio.run(bootstrapper.bootstrap_session_context(name=validated_project, job=job, infrastructure=infrastructure))

        # Set the singleton BM instance
        set_bm(bm)

        logger.info(f"Starting CLI run for {bm.session_info.project_name} job {bm.session_info.job}")

        return bm

    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise


def init_with_config(job: str, project: str = None, overrides: list[str] = [], config_dir: str = None) -> tuple[BM, Any]:
    """Initialize Buttermilk for CLI script use and return both BM instance and configuration.

    This variant returns both the BM instance and the full configuration object
    for scripts that need access to additional configuration.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        overrides: List of Hydra override strings for customization
        config_dir: Path to configuration directory (defaults to packaged config)

    Returns:
        Tuple of (Buttermilk instance, configuration object)

    Example:
        >>> from buttermilk.utils import cli
        >>> from buttermilk import logger  # Always use global logger import
        >>> # First session - project required
        >>> bm, config = cli.init_with_config(job="data_processing", project="my_project")
        >>> logger.info("Processing started")  # Session context automatically included
        >>> # Access additional config: config.my_custom_settings
        >>>
        >>> # Subsequent sessions - project optional (inherits from execution context)
        >>> bm2, config2 = cli.init_with_config(job="data_analysis")  # Uses "my_project"

        # To use your own config directory:
        >>> bm, config = cli.init_with_config(job="data_processing", project="my_project", config_dir="./conf")

    Note:
        Always use `from buttermilk import logger` for logging. The logger is
        a global singleton that automatically includes session context (session_id,
        job, project_name) in all log messages once a session is initialized.

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    # Handle default project name from script filename if not provided
    # This preserves the old behavior for compatibility but project will be validated
    # by ExecutionContext to ensure consistency across sessions
    default_project_name = None
    if project is None:
        # Try to get script name from __main__ module or current process
        if hasattr(sys.modules.get("__main__"), "__file__"):
            script_file = sys.modules["__main__"].__file__
            if script_file:
                default_project_name = Path(script_file).stem
            else:
                default_project_name = "cli_script"
        else:
            default_project_name = "cli_script"

    if not config_dir:
        # Default to packaged config directory
        config_dir = Path(__file__).parent.parent.resolve() / "conf"
        config_dir = config_dir.as_posix()

    # Add CLI-specific overrides
    cli_overrides = overrides.copy()
    cli_overrides.append("++run=cli")
    cli_overrides.append(f"++run.job={job}")

    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=config_dir, overrides=cli_overrides)

    try:
        # Bootstrap full context and session
        _, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

        # Validate and set project name using ExecutionContext
        # Use provided project or default, let ExecutionContext handle validation
        effective_project = project or default_project_name
        validated_project = infrastructure.validate_and_set_project(effective_project)

        # Create session BM instance with validated project
        bm = asyncio.run(bootstrapper.bootstrap_session_context(name=validated_project, job=job, infrastructure=infrastructure))

        # Set the singleton BM instance
        set_bm(bm)

        logger.info(f"Starting CLI run for {bm.session_info.project_name} job {bm.session_info.job}")

        # Get the configuration
        config = bootstrapper.get_configuration()

        return bm, config

    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise
