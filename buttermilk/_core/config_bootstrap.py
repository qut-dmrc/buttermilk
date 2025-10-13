"""Configuration bootstrapper for centralized setup of all Buttermilk infrastructure.

This module provides the ConfigurationBootstrapper class that serves as the single
entry point for all configuration management, eliminating scattered environment
variable access and configuration initialization throughout the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from buttermilk._core.execution_context import ExecutionContext
from buttermilk._core.log import logger
from buttermilk.utils.utils import load_dotenv


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
        logger.warning(
            f"Config directory does not exist: {resolved_path}\n"
            f"  Original path: {config_dir}\n"
            f"  Current working directory: {Path.cwd()}\n"
            f"  Note: Relative paths are resolved relative to CWD, not the script's directory.\n"
            f"  If you need script-relative paths, use: Path(__file__).parent / 'conf'"
        )

    return str(resolved_path)


class ConfigurationBootstrapper:
    """Single point of entry for all configuration management.

    This class eliminates scattered environment variable access and configuration
    initialization by providing centralized bootstrapping of all infrastructure
    components and execution contexts.

    Key responsibilities:
    1. Single Hydra initialization
    2. Environment variable management
    3. Credential resolution
    4. Infrastructure configuration
    5. Session context setup
    """

    def __init__(self, config_path: str = "conf", config_name: str = "config", overrides: list[str] | None = None, config: DictConfig | None = None):
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
        self.config: DictConfig = self._load_configuration(config)
        self._execution_context: ExecutionContext | None = None

        load_dotenv()

    def _load_configuration(self, config: DictConfig | None = None) -> DictConfig:
        """Load configuration via Hydra with search path support for fallback.

        If a config_path is provided, Hydra will search:
        1. Project config directory (config_path)
        2. Library config directory (fallback)

        This allows project-specific configs to override library defaults.

        Returns:
            Loaded and resolved configuration
        """
        if config is not None:
            config_to_instantiate = config
        else:
            try:
                # Check if we're already in a Hydra context (like CLI)
                from hydra.core.global_hydra import GlobalHydra

                if GlobalHydra.instance().is_initialized():
                    # We're already in a Hydra context, get the existing config
                    config_to_instantiate = compose(config_name=self.config_name, overrides=self.overrides)

                else:
                    # Load configuration using Hydra compose API
                    from pathlib import Path

                    # Get absolute path to config directory
                    config_dir = Path(self.config_path).resolve()

                    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
                        config_to_instantiate = compose(config_name=self.config_name, overrides=self.overrides)

                    # logger.debug("Configuration loaded via new Hydra initialization")  # Removed: logging not configured yet

            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise

        # Instantiate the config
        instantiated_config = hydra.utils.instantiate(config_to_instantiate)
        return instantiated_config

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
                        "OTEL_SERVICE_NAME": otel_config.get("service_name", "buttermilk"),
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
                        env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = cloud_config["credentials_path"]

        # Apply all environment variables
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            logger.debug(f"Set environment variable: {key}")

        if env_vars:
            logger.info(f"Configured {len(env_vars)} environment variables")

    def get_infrastructure_config(self) -> DictConfig:
        """Get configuration for all infrastructure components.

        Returns:
            DictConfig containing infrastructure configuration with _target_ keys intact
        """
        config = self._load_configuration()

        # Extract infrastructure configuration as DictConfig to preserve _target_ keys
        if hasattr(config, "infrastructure"):
            return config.infrastructure
        elif hasattr(config, "bm"):
            # Fallback: migrate old BM configuration to infrastructure format
            logger.warning("Using legacy 'bm' configuration - consider migrating to 'infrastructure'")
            return config.bm
        else:
            raise RuntimeError("No infrastructure configuration found in config")

    async def bootstrap_full_context(self, project_name: str | None = None) -> ExecutionContext:
        """Bootstrap complete execution context with all infrastructure (async).

        This method creates a baseline execution context for the application
        (e.g., API server) with all required infrastructure.

        Args:
            project_name: Optional project name to include in logging setup

        Returns:
            ExecutionContext: The configured execution context
        """
        # logger.debug("Bootstrapping full application context...")  # Removed: logging not configured yet

        # Create baseline execution context FIRST to ensure structured logging
        if self._execution_context is None:
            # Get infrastructure configuration to create ExecutionContext with full infrastructure
            # Note: config is already instantiated
            infrastructure_config = self.config.get("infrastructure", {})

            # Clouds are already instantiated
            hydrated_clouds = infrastructure_config.get("clouds", [])

            # Use async factory method with project_name
            from buttermilk._core.execution_context import get_or_create_execution_context_async

            self._execution_context = await get_or_create_execution_context_async(
                project_name=project_name,
                clouds=hydrated_clouds,
                logging=infrastructure_config.get("logging"),
                tracing=infrastructure_config.get("tracing", {}),
                datasets=infrastructure_config.get("datasets", {}),
            )
            await self._execution_context.ensure_initialized()
            logger.info(
                "ExecutionContext created with full infrastructure configuration", execution_context_id=self._execution_context.execution_context_id
            )

        # Initialize tracing now that infrastructure is ready
        try:
            await self._execution_context._initialize_all_tracing_providers()
            logger.info("Tracing providers initialized successfully")
        except Exception as e:
            # Log and fail the bootstrap - tracing is critical
            logger.exception("Failed to initialize tracing providers", error=str(e))
            raise RuntimeError(f"Tracing initialization failed: {e}") from e

        logger.info("Full application context bootstrap complete", execution_context_id=self._execution_context.execution_context_id)
        return self._execution_context

    async def bootstrap_session_context(self, name: str, job: str, template_paths: list[str] | None = None, config=None, **kwargs) -> Any:
        """Bootstrap session-specific BM instance (async).

        Args:
            name: User-defined name for the current session or project
            job: User-defined name for the specific job or task
            template_paths: Optional list of paths to search for templates.
            **kwargs: Additional arguments for session creation

        Returns:
            Session-scoped BM instance
        """
        logger.info("Bootstrapping session context", name=name, job=job)

        if self._execution_context is None:
            raise RuntimeError("ExecutionContext not initialized. Call bootstrap_full_context() first.")

        # Extract infrastructure components from ExecutionContext
        from buttermilk._core.bm_init import create_session_bm_async

        # Create session-scoped BM instance using ExecutionContext's infrastructure (async)
        session_bm = await create_session_bm_async(
            project_name=name,
            job=job,
            template_paths=template_paths,
            cloud_manager=self._execution_context.cloud_manager if self._execution_context.clouds else None,
            secret_manager=self._execution_context.secret_manager if self._execution_context._find_cloud_with_service("secrets") else None,
            llms_instance=self._execution_context.llms,
            query_runner=self._execution_context.query_runner if self._execution_context.clouds else None,
            logger_cfg=self._execution_context.logging,
            config=config,
            **kwargs,
        )

        # Ensure session BM is fully initialized
        await session_bm.ensure_initialized()
        logger.info(f"Session context bootstrap complete: {session_bm.session_info.session_id}")

        return session_bm


def create_configuration_bootstrapper(
    config_path: str = "conf", config_name: str = "config", overrides: list[str] | None = None, config: DictConfig | None = None
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
    return ConfigurationBootstrapper(config_path=config_path, config_name=config_name, overrides=overrides, config=config)


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

    t = threading.Thread(target=_thread_runner, name="buttermilk-init-loop", daemon=True)
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
    """Unified async session bootstrap function that also returns configuration.

    This is the primary async bootstrap pathway. Returns both the BM instance
    and the full configuration object.

    Args:
        job: Name for the specific job or task
        project_name: Project name (required for first session, optional for subsequent sessions)
        config_dir: Path to configuration directory. Can be:
            - None: Try cwd/buttermilk/conf, else use packaged conf
            - Relative path: Resolved against CWD
            - Absolute path: Used as-is (with ~ and $VAR expansion)
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of Hydra override strings for customization (e.g., ["run=cli", "debug=true"])
        config: Pre-loaded configuration (if already available from Hydra context)
        base_dir: DEPRECATED - no longer used, kept for backward compatibility

    Returns:
        Tuple of (Buttermilk instance, configuration object)

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    from pathlib import Path

    from buttermilk._core.dmrc import set_bm

    if config is None:  # we need to load it.
        # Resolve config directory using centralized resolution logic
        config_dir = resolve_config_dir(config_dir)

    # Prepare overrides
    bootstrap_overrides = (overrides or []).copy()

    # Override project_name if provided (needed for interpolation in config)
    if project_name is not None:
        bootstrap_overrides.append(f"++project_name={project_name}")

    # Only override job if explicitly provided (otherwise use config default)
    if job is not None:
        bootstrap_overrides.append(f"++bm.session_info.job={job}")
        bootstrap_overrides.append(f"++job={job}")

    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=config_dir, config_name=config_name, overrides=bootstrap_overrides, config=config)

    # Get the final resolved configuration to extract project/job BEFORE bootstrapping
    final_config = bootstrapper.config

    # Extract job and project from config if not provided as parameters
    # This must happen BEFORE creating ExecutionContext so we can pass project_name
    resolved_job = job if job is not None else final_config.bm.session_info.job
    resolved_project = project_name if project_name is not None else final_config.bm.session_info.project_name

    # Bootstrap async with project_name available
    execution_context = await bootstrapper.bootstrap_full_context(project_name=resolved_project)

    # Extract template_paths from config and resolve relative paths
    template_paths = getattr(final_config.bm.session_info, "template_paths", [])
    resolved_template_paths = []
    for path in template_paths:
        if not Path(path).is_absolute():
            # Resolve relative to config directory
            if config_dir:
                resolved_path = Path(config_dir) / path
                resolved_template_paths.append(str(resolved_path.resolve()))
            else:
                resolved_template_paths.append(str(Path(path).resolve()))
        else:
            resolved_template_paths.append(path)
    template_paths = resolved_template_paths

    # Validate and set project name using ExecutionContext
    validated_project = execution_context.validate_and_set_project(resolved_project)

    # Create session BM instance with validated project
    bm = await bootstrapper.bootstrap_session_context(name=validated_project, job=resolved_job, template_paths=template_paths, config=final_config)

    # Set the singleton BM instance
    set_bm(bm)

    # Extract run type from config if present, otherwise use generic message
    run_type_str = final_config.get("run", {}).get("_target_", "").split(".")[-1] if "run" in final_config else "session"
    logger.info(f"Starting {run_type_str} for {bm.session_info.project_name} job {bm.session_info.job}")

    return bm, final_config
