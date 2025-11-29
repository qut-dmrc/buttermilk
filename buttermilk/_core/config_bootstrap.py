"""Configuration bootstrapper for centralized setup of all Buttermilk infrastructure.

This module provides the ConfigurationBootstrapper class that serves as the single
entry point for all configuration management, eliminating scattered environment
variable access and configuration initialization throughout the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from buttermilk.utils.utils import load_dotenv


def register_library_configs_in_store(library_config_dir: Path) -> None:
    """Register library configs in ConfigStore for fallback during Hydra composition.

    This function scans the library config directory and registers all YAML configs
    in Hydra's ConfigStore. When Hydra composes configs, it will:
    1. First look for config files in the project directory
    2. Fall back to ConfigStore (library configs) if not found

    This provides automatic fallback from project → library configs.

    Args:
        library_config_dir: Path to Buttermilk's library config directory
    """
    cs = ConfigStore.instance()

    # Scan library config directory for all YAML files
    for config_file in library_config_dir.rglob("*.yaml"):
        # Skip config.yaml in root (it's the main config, not a group)
        if (
            config_file.name == "config.yaml"
            and config_file.parent == library_config_dir
        ):
            continue

        # Determine group from directory structure
        # e.g., conf/flows/trans.yaml -> group="flows", name="trans"
        relative_path = config_file.relative_to(library_config_dir)
        parts = relative_path.parts

        if len(parts) == 1:
            # Top-level config (no group)
            group = None
            name = config_file.stem
        else:
            # Config in a group directory
            group = "/".join(parts[:-1])  # Join all parent dirs as group
            name = config_file.stem

        try:
            # Load YAML config
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)

            # Register in ConfigStore
            # Use library provider name for clarity
            if group:
                cs.store(
                    group=group,
                    name=name,
                    node=config_dict,
                    provider="buttermilk-library",
                )
            else:
                cs.store(name=name, node=config_dict, provider="buttermilk-library")

        except Exception as e:
            # Don't fail initialization if a single config fails to register
            # Can't use logger at this stage before we are initialised properly.
            print(f"Failed to register library config {config_file}: {e}")


def resolve_config_dir(config_dir: str | None = None) -> str:
    """Resolve configuration directory path with intelligent defaults.

    Resolution logic:
    1. If config_dir is None:
       - Try cwd/buttermilk/conf (if exists)
       - Else use packaged config: <package>/buttermilk/conf
    2. If config_dir is relative:
       - Resolve relative to current working directory (CWD)
       - NOT relative to the calling script's directory
    3. If config_dir is absolute:
       - Expand ~ and $VAR, return as-is

    Args:
        config_dir: Optional path to configuration directory.
            Relative paths are resolved against the current working directory,
            not the script's directory. Use Path(__file__).parent / "conf"
            if you need paths relative to your script.

    Returns:
        Absolute path to configuration directory as string

    Example:
        >>> resolve_config_dir()  # Returns cwd/buttermilk/conf or package conf
        '/home/user/myproject/buttermilk/conf'
        >>> resolve_config_dir("conf")  # Returns cwd/conf (NOT script_dir/conf)
        '/home/user/myproject/conf'
        >>> resolve_config_dir("~/myconf")  # Returns expanded home path
        '/home/user/myconf'
        >>> # For script-relative paths:
        >>> from pathlib import Path
        >>> script_dir = Path(__file__).parent
        >>> resolve_config_dir(str(script_dir / "conf"))
        '/home/user/myproject/scripts/conf'
    """
    if config_dir is None:
        # Try CWD/buttermilk/conf first
        cwd_conf = Path.cwd() / "buttermilk" / "conf"
        if cwd_conf.exists():
            return str(cwd_conf.resolve())

        # Fall back to packaged config
        package_conf = Path(__file__).parent.parent / "conf"
        return str(package_conf.resolve())

    # Expand ~ and environment variables
    expanded = os.path.expandvars(os.path.expanduser(config_dir))
    config_path = Path(expanded)

    # Resolve to absolute path (relative paths are resolved against CWD)
    resolved_path = config_path.resolve()

    # Provide helpful error message if path doesn't exist
    if not resolved_path.exists():
        # Try to give helpful context about CWD vs script directory
        print(
            f"Config directory does not exist: {resolved_path}\n"
            f"  Original path: {config_dir}\n"
            f"  Current working directory: {Path.cwd()}\n"
            f"  Note: Relative paths are resolved relative to CWD, not the script's directory.\n"
            f"  If you need script-relative paths, use: Path(__file__).parent / 'conf'"
        )

    return str(resolved_path)


class ConfigurationBootstrapper:
    """Simplified configuration loader with library fallback.

    This class provides centralized config loading with project → library fallback.

    Key responsibilities:
    1. Hydra initialization with library config fallback
    2. Single instantiation pathway via Pydantic
    """

    def __init__(
        self,
        config_path: str = "conf",
        config_name: str = "config",
        overrides: list[str] | None = None,
        config: DictConfig | None = None,
    ):
        """Initialize the configuration bootstrapper.

        Args:
            config_path: Path to Hydra configuration directory
            config_name: Name of the configuration file to load (without .yaml extension)
            overrides: List of configuration overrides
            config: Pre-loaded configuration (if already available from Hydra context)
        """
        self.config_path = config_path
        self.config_name = config_name
        self.overrides = overrides or []
        self.config = self._load_configuration(
            config
        )  # ButtermilkConfig, not DictConfig

        load_dotenv()

    def _load_configuration(self, config: DictConfig | None = None):
        """Load configuration via Hydra with ConfigStore fallback.

        This method implements project → library config fallback by:
        1. Registering library configs in ConfigStore (fallback)
        2. Initializing Hydra with project config directory (priority)
        3. Hydra naturally falls back to ConfigStore if files not found

        This allows project-specific configs to override library defaults.

        Returns Pydantic ButtermilkConfig, not DictConfig - single instantiation pathway.

        Returns:
            ButtermilkConfig: Typed Pydantic configuration
        """
        if config is not None:
            dict_config = config
        else:
            try:
                # Check if we're already in a Hydra context (like CLI)
                from hydra.core.global_hydra import GlobalHydra

                if GlobalHydra.instance().is_initialized():
                    # We're already in a Hydra context, get the existing config
                    dict_config = compose(
                        config_name=self.config_name, overrides=self.overrides
                    )

                else:
                    # Determine library and project config directories
                    library_config_dir = Path(__file__).parent.parent / "conf"
                    library_config_dir = library_config_dir.resolve()
                    project_config_dir = Path(self.config_path).resolve()
                    is_custom_config = project_config_dir != library_config_dir

                    # Load configuration using Hydra compose API
                    with initialize_config_dir(
                        config_dir=str(project_config_dir), version_base="1.3"
                    ):
                        # Register library configs INSIDE the Hydra context for fallback
                        # This must happen AFTER initialize but BEFORE compose
                        if is_custom_config:
                            register_library_configs_in_store(library_config_dir)

                        dict_config = compose(
                            config_name=self.config_name, overrides=self.overrides
                        )

            except Exception as e:
                print(f"Failed to load configuration: {e}")
                raise

        # Single instantiation via Pydantic (not Hydra)
        from buttermilk._core.main_config import create_config_from_hydra

        return create_config_from_hydra(dict_config)

    def setup_environment_variables(self) -> None:
        """Set all required environment variables in one place.

        This centralizes all environment variable setup that was previously
        scattered throughout the codebase.
        """
        config = self._load_configuration()

        # Set up environment variables for various services
        env_vars = {}

        # OpenTelemetry configuration (previously in otel.py)
        if hasattr(config, "observability") and config.observability:
            otel_config = config.observability.get("opentelemetry", {})
            if otel_config.get("enabled", False):
                env_vars.update(
                    {
                        "OTEL_SERVICE_NAME": otel_config.get(
                            "service_name", "buttermilk"
                        ),
                        "OTEL_RESOURCE_ATTRIBUTES": f"service.name={otel_config.get('service_name', 'buttermilk')}",
                    }
                )

                if otel_config.get("endpoint"):
                    env_vars["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_config["endpoint"]

        # Cloud provider environment setup (previously scattered in cloud.py)
        if hasattr(config, "infrastructure") and config.infrastructure.get("clouds"):
            for cloud_config in config.infrastructure.clouds:
                if cloud_config.get("type") == "gcp":
                    # Set GCP-specific environment variables
                    if cloud_config.get("project_id"):
                        env_vars["GOOGLE_CLOUD_PROJECT"] = cloud_config["project_id"]
                    if cloud_config.get("credentials_path"):
                        env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = cloud_config[
                            "credentials_path"
                        ]

        # Apply all environment variables
        for key, value in env_vars.items():
            os.environ[key] = str(value)


def create_configuration_bootstrapper(
    config_path: str = "conf",
    config_name: str = "config",
    overrides: list[str] | None = None,
    config: DictConfig | None = None,
) -> ConfigurationBootstrapper:
    """Factory function to create a ConfigurationBootstrapper instance.

    Args:
        config_path: Path to Hydra configuration directory
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of configuration overrides
        config: Pre-loaded configuration (if already available from Hydra context)

    Returns:
        ConfigurationBootstrapper instance
    """
    return ConfigurationBootstrapper(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
        config=config,
    )


async def init_async(
    job: str | None = None,
    project_name: str | None = None,
    *,
    config_dir: str | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    config: DictConfig | None = None,
    base_dir: str | None = None,
):
    """PRIMARY async initialization function for Buttermilk.

    This is the recommended way to initialize Buttermilk in async contexts.
    Simple one-liner with async/await pattern for modern Python code.

    Args:
        job: Name for the specific job or task (defaults to "default" or from config)
        project_name: Project name (auto-detected from directory or config if not provided)
        config_dir: Path to configuration directory. Can be:
            - None: Auto-discover (tries cwd/buttermilk/conf, then package conf)
            - Relative path: Resolved relative to current working directory (CWD)
            - Absolute path: Used as-is (with ~ and $VAR expansion)
            Note: Relative paths are NOT resolved relative to the calling script's
            directory. For script-relative paths, use:
                from pathlib import Path
                config_dir = str(Path(__file__).parent / "conf")
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of Hydra override strings for customization (e.g., ["run=cli", "debug=true"])
        config: Pre-loaded configuration (if already available from Hydra context)
        base_dir: DEPRECATED - no longer used, kept for backward compatibility

    Returns:
        Buttermilk instance ready to use with config accessible via bm.cfg

    Example:
        >>> from buttermilk import init_async, bm
        >>> _ = await init_async()  # Primary async pathway
        >>> cfg = bm.cfg  # Access config
        >>> logger = bm.logger  # Contextualized logger

        >>> # With explicit config directory (relative to CWD)
        >>> _ = await init_async(config_dir="conf")

        >>> # With script-relative config directory
        >>> from pathlib import Path
        >>> script_dir = Path(__file__).parent
        >>> _ = await init_async(config_dir=str(script_dir / "conf"))
    """
    bm, config = await bootstrap_session_with_config_async(
        job=job,
        project_name=project_name,
        config_dir=config_dir,
        config_name=config_name,
        overrides=overrides,
        config=config,
        base_dir=base_dir,
    )
    return bm


def _run_coro_sync(coro):
    """Run a coroutine from sync code.

    - If no event loop is running, use asyncio.run.
    - If an event loop is already running in this thread, run the coroutine
      on a dedicated loop in a background thread and block for the result.
    """
    import asyncio
    import concurrent.futures as cf
    import threading

    try:
        # Raises RuntimeError if no running loop in this thread
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: safe to use asyncio.run
        return asyncio.run(coro)

    # A loop is running in this thread: use a separate thread + loop
    fut: cf.Future = cf.Future()

    def _thread_runner():
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(coro)
                fut.set_result(result)
            finally:
                loop.close()
        except BaseException as e:
            fut.set_exception(e)

    t = threading.Thread(
        target=_thread_runner, name="buttermilk-init-loop", daemon=True
    )
    t.start()
    return fut.result()


def init(
    job: str | None = None,
    project_name: str | None = None,
    *,
    config_dir: str | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    config: DictConfig | None = None,
    base_dir: str | None = None,
):
    """Lightweight sync wrapper for init_async() - DEPRECATED.

    This is a simple wrapper that exists for backward compatibility only.
    New code should use init_async() directly for better async/await patterns.

    Args:
        job: Name for the specific job or task (defaults to "default" or from config)
        project_name: Project name (auto-detected from directory or config if not provided)
        config_dir: Path to configuration directory. Can be:
            - None: Auto-discover (tries cwd/buttermilk/conf, then package conf)
            - Relative path: Resolved relative to current working directory (CWD)
            - Absolute path: Used as-is (with ~ and $VAR expansion)
            Note: Relative paths are NOT resolved relative to the calling script's
            directory. For script-relative paths, use:
                from pathlib import Path
                config_dir = str(Path(__file__).parent / "conf")
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of Hydra override strings for customization (e.g., ["run=cli", "debug=true"])
        config: Pre-loaded configuration (if already available from Hydra context)
        base_dir: DEPRECATED - no longer used, kept for backward compatibility

    Returns:
        Buttermilk instance ready to use with config accessible via bm.cfg

    Example:
        >>> from buttermilk import init, bm
        >>> _ = init()  # Sync wrapper (not recommended for new code)
        >>> cfg = bm.cfg  # Access config

        >>> # With script-relative config directory
        >>> from pathlib import Path
        >>> script_dir = Path(__file__).parent
        >>> _ = init(config_dir=str(script_dir / "../../conf"))
    """

    # Run directly if no loop; otherwise run in a background thread loop
    return _run_coro_sync(
        init_async(
            job=job,
            project_name=project_name,
            config_dir=config_dir,
            config_name=config_name,
            overrides=overrides,
            config=config,
            base_dir=base_dir,
        )
    )


async def bootstrap_session_with_config_async(
    job: str | None = None,
    project_name: str | None = None,
    config_dir: str | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    config: DictConfig | None = None,
    base_dir: str | None = None,
):
    """Simplified unified async session bootstrap.

    This is the primary async bootstrap pathway using the new simplified architecture.
    Returns both the BM instance and the full typed configuration.

    Args:
        job: Name for the specific job or task
        project_name: Project name (required for first session, optional for subsequent sessions)
        config_dir: Path to configuration directory
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of Hydra override strings
        config: Pre-loaded configuration (if already available from Hydra context)
        base_dir: DEPRECATED - no longer used, kept for backward compatibility

    Returns:
        Tuple of (Buttermilk instance, typed ButtermilkConfig)
    """
    from pathlib import Path

    from buttermilk._core.dmrc import set_bm
    from buttermilk._core.execution_context import (
        create_session_from_context_async,
        from_config_async,
    )

    # Phase 1: Load typed config (single instantiation via Pydantic)
    if config is None:
        config_dir = resolve_config_dir(config_dir)

    # Prepare overrides
    bootstrap_overrides = (overrides or []).copy()
    if project_name is not None:
        bootstrap_overrides.append(f"++project_name={project_name}")
    if job is not None:
        bootstrap_overrides.append(f"++job={job}")

    # Load config - single instantiation pathway via Pydantic
    bootstrapper = ConfigurationBootstrapper(
        config_path=config_dir,
        config_name=config_name,
        overrides=bootstrap_overrides,
        config=config,
    )
    typed_config = (
        bootstrapper.config
    )  # Already ButtermilkConfig from _load_configuration()

    # Resolve template paths
    template_paths = list(typed_config.session.template_paths)
    resolved_template_paths = []
    for path in template_paths:
        if not Path(path).is_absolute():
            if config_dir:
                resolved_path = Path(config_dir) / path
                resolved_template_paths.append(str(resolved_path.resolve()))
            else:
                resolved_template_paths.append(str(Path(path).resolve()))
        else:
            resolved_template_paths.append(path)

    # Update session with resolved paths
    typed_config.session.template_paths = resolved_template_paths

    # Phase 2: Create or get ExecutionContext (singleton)
    # Pass root-level llms config if present (for model_parameters)
    llms_config = typed_config.llms if hasattr(typed_config, 'llms') else None
    execution_context = await from_config_async(
        typed_config.infrastructure,
        project_name=typed_config.session.project_name,
        default_llm_wrapper=typed_config.session.llm_wrapper,
        llms_config=llms_config,
    )

    # Phase 3: Create session BM instance
    bm = await create_session_from_context_async(
        execution_context=execution_context,
        session=typed_config.session,
        storage_configs=typed_config.storage,
        full_config=typed_config,
    )

    # Set singleton
    set_bm(bm)

    # Log startup
    run_type_str = (
        typed_config.run.mode if hasattr(typed_config.run, "mode") else "session"
    )
    bm.logger.info(
        f"Starting {run_type_str} for {bm.session_info.project_name} job {bm.session_info.job}"
    )

    return bm, typed_config
