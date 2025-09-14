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
from buttermilk._core.infrastructure import InfrastructureManager
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

    def __init__(self, config_path: str = "conf", overrides: list[str] | None = None, config: DictConfig | None = None):
        """Initialize the configuration bootstrapper.

        Args:
            config_path: Path to Hydra configuration directory
            overrides: List of configuration overrides
            config: Pre-loaded configuration (if already available from Hydra context)
        """
        self.config_path = config_path
        self.overrides = overrides or []
        self._config: DictConfig | None = config
        self._infrastructure: InfrastructureManager | None = None
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

                    self._config = compose(config_name="config", overrides=self.overrides)
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
                        self._config = compose(config_name="config", overrides=self.overrides)
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
    
    def _create_infrastructure_manager(self) -> InfrastructureManager:
        """Create and initialize the infrastructure manager.
        
        Returns:
            Configured InfrastructureManager instance
        """
        if self._infrastructure is None:
            # Set up environment variables first
            self.setup_environment_variables()
            
            # Get infrastructure configuration
            infrastructure_config = self.get_infrastructure_config()
            
            # Create infrastructure manager from configuration
            from buttermilk import create_infrastructure_from_config
            self._infrastructure = create_infrastructure_from_config(infrastructure_config)
            
            # Initialize infrastructure components
            self._infrastructure.initialize_components()
            logger.info("Infrastructure manager created and initialized")
            
        return self._infrastructure
    
    async def bootstrap_full_context(self) -> tuple[ExecutionContext, InfrastructureManager]:
        """Bootstrap complete execution context with all infrastructure.
        
        This method creates a baseline execution context for the application
        (e.g., API server) along with the infrastructure manager.
        
        Returns:
            Tuple of (ExecutionContext, InfrastructureManager)
        """
        logger.info("Bootstrapping full application context...")
        
        # Create baseline execution context FIRST to ensure structured logging
        if self._execution_context is None:
            # Get infrastructure configuration to create ExecutionContext with full infrastructure
            config = self._load_configuration()
            infrastructure_config = config.get("infrastructure", {})
            
            # Pass the FULL infrastructure configuration to ExecutionContext
            # This ensures ExecutionContext has its own CloudManager, SecretManager, etc.
            self._execution_context = get_or_create_execution_context(
                clouds=infrastructure_config.get("clouds", []),
                secret_provider=infrastructure_config.get("secret_provider"),
                logging=infrastructure_config.get("logging"),
                pubsub=infrastructure_config.get("pubsub"),
                tracing=infrastructure_config.get("tracing", {}),
                datasets=infrastructure_config.get("datasets", {}),
            )
            await self._execution_context.ensure_initialized()
            logger.info("ExecutionContext created with full infrastructure configuration")
        
        # Get infrastructure manager from ExecutionContext
        # This ensures InfrastructureManager uses the same infrastructure as ExecutionContext
        infrastructure = self._execution_context.get_infrastructure_manager()
        
        # Initialize tracing now that infrastructure is ready and BM singleton should be available
        try:
            await self._execution_context._initialize_all_tracing_providers()
            logger.info("Tracing providers initialized successfully")
        except Exception as e:
            # Log but don't fail the bootstrap - tracing is important but not critical
            logger.warning(f"Failed to initialize tracing providers: {e}")
        
        logger.info("Full application context bootstrap complete")
        return self._execution_context, infrastructure
    
    async def bootstrap_session_context(self, name: str, job: str, infrastructure=None, **kwargs) -> Any:
        """Bootstrap session-specific BM instance.
        
        Args:
            name: User-defined name for the current session or project
            job: User-defined name for the specific job or task
            infrastructure: Optional existing InfrastructureManager to use (preferred)
            **kwargs: Additional arguments for session creation
            
        Returns:
            Session-scoped BM instance
        """
        logger.info(f"Bootstrapping session context: {name}/{job}")
        
        # Use existing infrastructure if provided, otherwise try to get from ExecutionContext
        if infrastructure is not None:
            logger.debug("Using existing infrastructure from ExecutionContext")
        elif self._execution_context is not None:
            logger.debug("Getting infrastructure from ExecutionContext")
            infrastructure = self._execution_context.get_infrastructure_manager()
        else:
            logger.debug("Creating new infrastructure for session (legacy mode)")
            infrastructure = self._create_infrastructure_manager()
        
        # Create session-scoped BM instance
        session_bm = infrastructure.create_session_bm(
            name=name,
            job=job,
            **kwargs
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
    
    def get_infrastructure_manager(self) -> InfrastructureManager:
        """Get the infrastructure manager instance.
        
        Returns:
            InfrastructureManager instance
        """
        return self._create_infrastructure_manager()


def create_configuration_bootstrapper(
    config_path: str = "conf", overrides: list[str] | None = None, config: DictConfig | None = None
) -> ConfigurationBootstrapper:
    """Factory function to create a ConfigurationBootstrapper instance.

    Args:
        config_path: Path to Hydra configuration directory
        overrides: List of configuration overrides
        config: Pre-loaded configuration (if already available from Hydra context)

    Returns:
        ConfigurationBootstrapper instance
    """
    return ConfigurationBootstrapper(config_path=config_path, overrides=overrides, config=config)


# Configuration files are stored in the local directory, and
# options can be passed in at initialization.
def init(
    job: str,
    project: str | None = None,
    *,
    run_type: str = "cli",
    config_dir: str | None = None,
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
        overrides: List of Hydra override strings for customization
        config: Pre-loaded configuration (if already available from Hydra context)

    Returns:
        bm: the Buttermilk instance

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.


    Args:
        job: Name for the specific job or task
        project: Project name (required for first session, optional for subsequent sessions)
        run_type: Type of run ("cli", "notebook", etc.) for override management
        config_dir: Path to configuration directory (defaults to packaged config)
        overrides: List of Hydra override strings for customization
        config: Pre-loaded configuration (if already available from Hydra context)

    Returns:
        Buttermilk instance ready to use

    Raises:
        RuntimeError: If project is required but not provided, or if project
                     mismatches existing execution context project.
    """
    bm, config = bootstrap_session_with_config(job=job, project=project, run_type=run_type, config_dir=config_dir, overrides=overrides, config=config)
    return bm


def bootstrap_session_with_config(
    job: str,
    project: str | None = None,
    run_type: str = "cli",
    config_dir: str | None = None,
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

    from buttermilk import set_bm

    # Resolve config directory - default to packaged config if not provided
    if not config_dir:
        config_dir = Path(__file__).parent.parent.resolve() / "conf"
        config_dir = config_dir.as_posix()

    # Prepare overrides with run-specific settings
    bootstrap_overrides = (overrides or []).copy()
    bootstrap_overrides.append(f"+run={run_type}")
    bootstrap_overrides.append(f"++run.job={job}")

    # Create bootstrapper with configuration
    bootstrapper = ConfigurationBootstrapper(config_path=config_dir, overrides=bootstrap_overrides, config=config)

    try:
        # Bootstrap full context and session
        execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

        # Validate and set project name using ExecutionContext
        validated_project = execution_context.validate_and_set_project(project)

        # Create session BM instance with validated project
        bm = asyncio.run(bootstrapper.bootstrap_session_context(name=validated_project, job=job, infrastructure=infrastructure))

        # Set the singleton BM instance
        set_bm(bm)

        logger.info(f"Starting {run_type} run for {bm.session_info.project_name} job {bm.session_info.job}")

        # Get the configuration
        config = bootstrapper.get_configuration()

        return bm, config

    except Exception as e:
        logger.error(f"Failed to initialize Buttermilk: {e}")
        raise
