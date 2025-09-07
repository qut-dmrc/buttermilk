"""Configuration bootstrapper for centralized setup of all Buttermilk infrastructure.

This module provides the ConfigurationBootstrapper class that serves as the single
entry point for all configuration management, eliminating scattered environment 
variable access and configuration initialization throughout the codebase.
"""

from __future__ import annotations

import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from buttermilk._core.execution_context import ExecutionContext, create_execution_context
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
                    from hydra import compose, initialize_config_dir
                    from pathlib import Path
                    
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
        if hasattr(config, 'observability') and config.observability:
            otel_config = config.observability.get('opentelemetry', {})
            if otel_config.get('enabled', False):
                env_vars.update({
                    'OTEL_SERVICE_NAME': otel_config.get('service_name', 'buttermilk'),
                    'OTEL_RESOURCE_ATTRIBUTES': f"service.name={otel_config.get('service_name', 'buttermilk')}",
                })
                
                if otel_config.get('endpoint'):
                    env_vars['OTEL_EXPORTER_OTLP_ENDPOINT'] = otel_config['endpoint']
        
        # Cloud provider environment setup (previously scattered in cloud.py)
        if hasattr(config, 'infrastructure') and config.infrastructure.get('clouds'):
            for cloud_config in config.infrastructure.clouds:
                if cloud_config.get('type') == 'gcp':
                    # Set GCP-specific environment variables
                    if cloud_config.get('project_id'):
                        env_vars['GOOGLE_CLOUD_PROJECT'] = cloud_config['project_id']
                    if cloud_config.get('credentials_path'):
                        env_vars['GOOGLE_APPLICATION_CREDENTIALS'] = cloud_config['credentials_path']
        
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
        if hasattr(config, 'infrastructure'):
            return config.infrastructure
        elif hasattr(config, 'bm'):
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
            # Get infrastructure configuration to extract tracing settings
            config = self._load_configuration()
            infrastructure_config = config.get('infrastructure', {})
            
            # Extract tracing configuration from infrastructure.tracing
            tracing_config = infrastructure_config.get('tracing', {})
            
            self._execution_context = create_execution_context(tracing=tracing_config)
            await self._execution_context.ensure_initialized()
            logger.info("Baseline execution context created with structured logging and tracing config")
        
        # Create infrastructure manager (may fail, but logs will be captured)
        infrastructure = self._create_infrastructure_manager()
        
        logger.info("Full application context bootstrap complete")
        return self._execution_context, infrastructure
    
    async def bootstrap_session_context(self, name: str, job: str, **kwargs) -> Any:
        """Bootstrap session-specific BM instance.
        
        Args:
            name: User-defined name for the current session or project
            job: User-defined name for the specific job or task
            **kwargs: Additional arguments for session creation
            
        Returns:
            Session-scoped BM instance
        """
        logger.info(f"Bootstrapping session context: {name}/{job}")
        
        # Ensure infrastructure is available
        infrastructure = self._create_infrastructure_manager()
        
        # Create session-scoped BM instance
        platform = kwargs.pop('platform', 'local')  # Extract platform to avoid duplicate
        session_bm = infrastructure.create_session_bm(
            name=name,
            job=job,
            platform=platform,
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
    config_path: str = "conf", 
    overrides: list[str] | None = None,
    config: DictConfig | None = None
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