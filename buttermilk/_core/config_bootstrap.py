"""Configuration bootstrapper for centralized setup of all Buttermilk infrastructure.

This module provides the ConfigurationBootstrapper class that serves as the single
entry point for all configuration management, eliminating scattered environment
variable access and configuration initialization throughout the codebase.
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

from buttermilk._core.execution_context import ExecutionContext, get_or_create_execution_context
from buttermilk._core.log import logger


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
        self._config: DictConfig | None = config
        self._execution_context: ExecutionContext | None = None

    def _load_configuration(self) -> DictConfig:
        """Load configuration via Hydra (single initialization).

        Returns:
            Loaded and resolved configuration
        """
        if self._config is None:
            try:
                # Check if we're already in a Hydra context (like CLI)
                from hydra.core.global_hydra import GlobalHydra

                if GlobalHydra.instance().is_initialized():
                    # We're already in a Hydra context, get the existing config
                    from hydra import compose

                    self._config = compose(config_name=self.config_name, overrides=self.overrides)
                    OmegaConf.resolve(self._config)
                    logger.info("Configuration loaded from existing Hydra context")
                else:
                    # Load configuration using Hydra compose API
                    from pathlib import Path

                    from hydra import compose, initialize_config_dir

                    # Get absolute path to config directory
                    config_dir = Path(__file__).parent.parent / self.config_path
                    config_dir = config_dir.resolve()

                    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
                        self._config = compose(config_name=self.config_name, overrides=self.overrides)
                        OmegaConf.resolve(self._config)

                    logger.info("Configuration loaded via new Hydra initialization")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise

        return self._config

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

    async def bootstrap_full_context(self) -> ExecutionContext:
        """Bootstrap complete execution context with all infrastructure.

        This method creates a baseline execution context for the application
        (e.g., API server) with all required infrastructure.

        Returns:
            ExecutionContext: The configured execution context
        """
        logger.info("Bootstrapping full application context...")

        # Create baseline execution context FIRST to ensure structured logging
        if self._execution_context is None:
            # Get infrastructure configuration to create ExecutionContext with full infrastructure
            config = self._load_configuration()
            infrastructure_config = config.get("infrastructure", {})

            # Debug: Log top-level configuration keys for troubleshooting
            config_keys = list(config.keys()) if config else []
            logger.debug("Configuration loaded", config_keys=config_keys)

            # Pass the FULL infrastructure configuration to ExecutionContext
            # This ensures ExecutionContext has its own CloudManager, SecretManager, etc.

            # Debug: Log infrastructure configuration for troubleshooting
            logger.debug(
                "Infrastructure configuration loaded",
                clouds_count=len(infrastructure_config.get("clouds", [])),
                has_secret_provider=bool(infrastructure_config.get("secret_provider")),
                has_logging=bool(infrastructure_config.get("logging")),
                has_tracing=bool(infrastructure_config.get("tracing")),
                has_datasets=bool(infrastructure_config.get("datasets")),
            )

            # Instantiate cloud configurations using Hydra
            hydrated_clouds = []
            for cloud_config in infrastructure_config.get("clouds", []):
                try:
                    if isinstance(cloud_config, DictConfig):
                        import hydra

                        hydrated_cloud = hydra.utils.instantiate(cloud_config)
                        hydrated_clouds.append(hydrated_cloud)
                    else:
                        hydrated_clouds.append(cloud_config)
                except Exception as e:
                    logger.error(f"Failed to instantiate cloud config {cloud_config}: {e}")
                    raise RuntimeError(f"Cannot instantiate cloud provider: {e}") from e

            self._execution_context = get_or_create_execution_context(
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
            # Log but don't fail the bootstrap - tracing is important but not critical
            logger.warning("Failed to initialize tracing providers", error=str(e))

        logger.info("Full application context bootstrap complete", execution_context_id=self._execution_context.execution_context_id)
        return self._execution_context

    async def bootstrap_session_context(self, name: str, job: str, template_paths: list[str] | None = None, **kwargs) -> Any:
        """Bootstrap session-specific BM instance.

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
        from buttermilk._core.bm_init import create_session_bm

        # Create session-scoped BM instance using ExecutionContext's infrastructure
        session_bm = create_session_bm(
            name=name,
            job=job,
            template_paths=template_paths,
            cloud_manager=self._execution_context.cloud_manager if self._execution_context.clouds else None,
            secret_manager=self._execution_context.secret_manager if self._execution_context._find_cloud_with_service("secrets") else None,
            llms_instance=self._execution_context.llms,
            query_runner=self._execution_context.query_runner if self._execution_context.clouds else None,
            logger_cfg=self._execution_context.logging,
            **kwargs,
        )

        # Ensure session BM is fully initialized
        await session_bm.ensure_initialized()
        logger.info(f"Session context bootstrap complete: {session_bm.session_info.session_id}")

        return session_bm

    def get_configuration(self) -> DictConfig:
        """Get the loaded configuration.

        Returns:
            Loaded Hydra configuration
        """
        return self._load_configuration()


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


# Configuration files are stored in the local directory, and
# options can be passed in at initialization.
def init(
    job: str,
    project: str | None = None,
    *,
    run_type: str = "cli",
    config_dir: str | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    config: DictConfig | None = None,
):
    """Unified session bootstrap function for all entry points.

    Simple one-liner initialization for Buttermilk.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        run_type: Type of run ("cli", "notebook", etc.) for override management
        config_dir: Path to configuration directory (defaults to packaged config)
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of Hydra override strings for customization
        config: Pre-loaded configuration (if already available from Hydra context)

    Returns:
        Buttermilk instance ready to use

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    bm, config = bootstrap_session_with_config(
        job=job, project=project, run_type=run_type, config_dir=config_dir, config_name=config_name, overrides=overrides, config=config
    )
    return bm


def bootstrap_session_with_config(
    job: str | None = None,
    project: str | None = None,
    run_type: str = "cli",
    config_dir: str | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    config: DictConfig | None = None,
):
    """Unified session bootstrap function that also returns configuration.

    This variant returns both the BM instance and the full configuration object
    for scripts that need access to additional configuration.

    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        run_type: Type of run ("cli", "notebook", etc.) for override management
        config_dir: Path to configuration directory (defaults to packaged config)
        config_name: Name of the configuration file to load (without .yaml extension)
        overrides: List of Hydra override strings for customization
        config: Pre-loaded configuration (if already available from Hydra context)

    Returns:
        Tuple of (Buttermilk instance, configuration object)

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    import asyncio
    from pathlib import Path

    from buttermilk._core.dmrc import set_bm

    # Resolve config directory - default to packaged config if not provided
    if not config_dir:
        config_dir = Path(__file__).parent.parent.resolve() / "conf"
        config_dir = config_dir.as_posix()
    else:
        # If config_dir is provided, resolve it relative to the calling app's CWD
        # Also expand user (~) and environment variables for convenience
        expanded = os.path.expandvars(os.path.expanduser(config_dir))
        cfg_path = Path(expanded)
        if not cfg_path.is_absolute():
            cfg_path = Path(os.getcwd()) / cfg_path
        config_dir = cfg_path.resolve().as_posix()

    # Prepare overrides with run-specific settings
    bootstrap_overrides = (overrides or []).copy()
    bootstrap_overrides.append(f"run={run_type}")

    # Only override job if explicitly provided (otherwise use config default)
    if job is not None:
        bootstrap_overrides.append(f"++run.job={job}")

    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=config_dir, config_name=config_name, overrides=bootstrap_overrides, config=config)

    try:
        # Bootstrap full context and session
        execution_context = asyncio.run(bootstrapper.bootstrap_full_context())

        # Get the final resolved configuration
        final_config = bootstrapper.get_configuration()

        # Extract job and project from config if not provided as parameters
        resolved_job = job if job is not None else final_config.run.job
        resolved_project = project if project is not None else final_config.run.name

        # Extract template_paths from config
        template_paths = final_config.bm.session_info.get("template_paths", [])

        # Validate and set project name using ExecutionContext
        validated_project = execution_context.validate_and_set_project(resolved_project)

        # Create session BM instance with validated project
        bm = asyncio.run(bootstrapper.bootstrap_session_context(name=validated_project, job=resolved_job, template_paths=template_paths))

        # Set the singleton BM instance
        set_bm(bm)

        logger.info(f"Starting {run_type} run for {bm.session_info.project_name} job {bm.session_info.job}")

        return bm, final_config

    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise
