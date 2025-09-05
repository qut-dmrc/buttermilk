"""Infrastructure management for shared Buttermilk components.

This module provides the InfrastructureManager class that handles creation
and management of shared infrastructure components (CloudManager, SecretsManager, 
LLMs) that can be injected into session-scoped BM instances.
"""

from __future__ import annotations

from typing import Any

import pydantic
from pydantic import BaseModel, Field

from buttermilk._core.log import logger
from buttermilk._core.utils.lazy_loading import cached_property


class InfrastructureManager(BaseModel):
    """Manages shared infrastructure components for Buttermilk.
    
    This class creates and manages shared infrastructure components that can
    be injected into multiple session-scoped BM instances. It uses service-aware
    cloud provider configurations to eliminate redundant configuration patterns.
    
    All services (secrets, logging, pubsub, tracing) are configured within
    their respective cloud provider objects, providing a clean hierarchy and
    single source of truth for each cloud's services.
    
    Attributes:
        clouds: List of service-aware cloud provider configurations.
        llms: LLM configuration data.
    """
    
    # Infrastructure configuration - service-aware cloud providers
    clouds: list[Any] = Field(
        default_factory=list,
        description="List of service-aware cloud provider configurations."
    )
    llms: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM configuration data."
    )
    
    # Private infrastructure instances
    _cloud_manager: Any = pydantic.PrivateAttr(default=None)
    _secret_manager: Any = pydantic.PrivateAttr(default=None)
    _llms_instance: Any = pydantic.PrivateAttr(default=None)
    _query_runner: Any = pydantic.PrivateAttr(default=None)
    
    @cached_property
    def cloud_manager(self) -> Any:
        """Get or create the CloudManager instance."""
        if self._cloud_manager is None:
            if not self.clouds:
                raise RuntimeError("No cloud configurations provided.")
            
            from buttermilk._core.cloud import CloudManager
            self._cloud_manager = CloudManager(clouds=self.clouds)
            logger.debug("CloudManager initialized successfully")
        
        return self._cloud_manager
    
    @cached_property 
    def secret_manager(self) -> Any:
        """Get or create the SecretsManager instance."""
        if self._secret_manager is None:
            # Find cloud provider with secrets service configured
            secrets_cloud = self._find_cloud_with_service("secrets")
            if not secrets_cloud:
                raise RuntimeError("No cloud provider with secrets service configured.")
            
            from buttermilk._core.secrets import SecretsManager
            
            # Get secrets configuration from cloud provider
            secrets_config = secrets_cloud.get_client_config("secretmanager")
            self._secret_manager = SecretsManager(**secrets_config)
            logger.debug(f"SecretsManager initialized with {secrets_cloud.type} provider")
        
        return self._secret_manager
    
    @cached_property
    def llms_instance(self) -> Any:
        """Get or create the LLMs instance."""
        if self._llms_instance is None:
            if not self.llms:
                raise RuntimeError("LLMs configuration is missing.")
            
            from buttermilk._core.llms import LLMs
            
            # Get connections from secret manager
            try:
                connections_data = self.secret_manager.get_secret(cfg_key="models_secret")
                if not connections_data:
                    raise RuntimeError("Failed to retrieve LLM connections from secret manager")
                
                logger.debug(f"Retrieved {len(connections_data)} LLM connections from secret manager")
            except Exception as e:
                logger.error(f"Failed to get LLM connections: {e}")
                raise RuntimeError(f"Cannot initialize LLMs: {e}") from e
            
            # Initialize LLMs with connections and configuration
            self._llms_instance = LLMs(
                connections=connections_data,
                **self.llms
            )
            logger.debug("LLMs instance initialized successfully")
        
        return self._llms_instance
    
    @cached_property
    def query_runner(self) -> Any:
        """Get or create the QueryRunner instance."""
        if self._query_runner is None:
            from buttermilk._core.query import QueryRunner
            self._query_runner = QueryRunner(bq_client=self.cloud_manager.bq)
            logger.debug("QueryRunner initialized successfully")
        
        return self._query_runner
    
    def _find_cloud_with_service(self, service: str) -> Any | None:
        """Find the first cloud provider that has a specific service configured.
        
        Args:
            service: Service name to look for (e.g., "secrets", "logging", "pubsub", "tracing")
            
        Returns:
            Cloud provider configuration with the service, or None if not found.
        """
        for cloud in self.clouds:
            if hasattr(cloud, 'has_service') and cloud.has_service(service):
                return cloud
        return None
    
    def get_service_config(self, service: str) -> dict[str, Any] | None:
        """Get configuration for a specific service from any configured cloud.
        
        Args:
            service: Service name (e.g., "secrets", "logging", "pubsub", "tracing")
            
        Returns:
            Service configuration dict, or None if service not configured.
        """
        cloud = self._find_cloud_with_service(service)
        if cloud:
            return cloud.get_client_config(service)
        return None
    
    def initialize_components(self) -> None:
        """Initialize all infrastructure components synchronously.
        
        This method triggers the creation of all infrastructure components
        and performs necessary authentication/setup steps.
        """
        logger.info("Initializing infrastructure components...")
        
        # Initialize cloud components if configured
        if self.clouds:
            logger.debug("Initializing cloud manager...")
            _ = self.cloud_manager
            
            # Perform cloud authentication
            self.cloud_manager.login_clouds()
            logger.debug("Cloud authentication completed")
        
        # Initialize secret manager if configured
        if self._find_cloud_with_service("secrets"):
            logger.debug("Initializing secret manager...")
            _ = self.secret_manager
        
        # Initialize LLMs if configured
        if self.llms:
            logger.debug("Initializing LLMs...")
            _ = self.llms_instance
        
        # Set up cloud logging if configured
        if self._find_cloud_with_service("logging"):
            logger.debug("Setting up cloud logging...")
            self._setup_cloud_logging()
        
        logger.info("Infrastructure initialization completed")
    
    def _setup_cloud_logging(self) -> None:
        """Set up cloud-based logging if configured."""
        logging_cloud = self._find_cloud_with_service("logging")
        if not logging_cloud:
            return
        
        try:
            from buttermilk._core.log import setup_cloud_logging
            from buttermilk._core.context import get_logging_context
            
            context_info = get_logging_context()
            logging_config = logging_cloud.get_client_config("logging")
            
            # Create a logger config object for backward compatibility
            class LoggerConfig:
                def __init__(self, config_dict):
                    self.type = logging_cloud.type
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            
            logger_cfg = LoggerConfig(logging_config)
            setup_cloud_logging(logger_cfg, self.cloud_manager, context_info)
            logger.debug(f"Cloud logging configured with {logging_cloud.type} provider")
        except Exception as e:
            logger.warning(f"Failed to setup cloud logging: {e}")
    
    def create_session_bm(
        self,
        name: str,
        job: str,
        batch_id: str | None = None,
        platform: str = "local",
        save_dir_base: str | None = None,
        **kwargs
    ) -> Any:
        """Create a session-scoped BM instance with infrastructure injection.
        
        Args:
            name: User-defined name for the current session or project.
            job: User-defined name for the specific job or task.
            batch_id: Optional batch identifier for grouping related sessions.
            platform: Platform where the session is running.
            save_dir_base: Base directory for session outputs.
            **kwargs: Additional arguments for SessionInfo.
            
        Returns:
            BM: A new session-scoped BM instance with infrastructure injected.
        """
        from buttermilk._core.bm_init import create_session_bm
        
        return create_session_bm(
            name=name,
            job=job,
            batch_id=batch_id,
            platform=platform,
            save_dir_base=save_dir_base,
            cloud_manager=self.cloud_manager if self.clouds else None,
            secret_manager=self.secret_manager if self.secret_provider else None,
            llms_instance=self.llms_instance if self.llms else None,
            **kwargs
        )
    
    def create_batch_session_bm(
        self,
        name: str,
        job: str,
        batch_id: str,
        platform: str = "local",
        save_dir_base: str | None = None,
        **kwargs
    ) -> Any:
        """Create a batch session-scoped BM instance with infrastructure injection.
        
        Args:
            name: User-defined name for the current session or project.
            job: User-defined name for the specific job or task.
            batch_id: Batch identifier that this session belongs to.
            platform: Platform where the session is running.
            save_dir_base: Base directory for session outputs.
            **kwargs: Additional arguments for SessionInfo.
            
        Returns:
            BM: A new session-scoped BM instance belonging to the batch.
        """
        return self.create_session_bm(
            name=name,
            job=job,
            batch_id=batch_id,
            platform=platform,
            save_dir_base=save_dir_base,
            **kwargs
        )


def create_infrastructure_manager(
    clouds: list[Any] | None = None,
    llms: dict[str, Any] | None = None,
    **kwargs
) -> InfrastructureManager:
    """Factory function to create an InfrastructureManager instance.
    
    This factory creates an infrastructure manager using service-aware cloud
    provider configurations. All services (secrets, logging, pubsub, tracing)
    are configured within their respective cloud provider objects.
    
    Args:
        clouds: List of service-aware cloud provider configurations.
        llms: LLM configuration data.
        **kwargs: Additional configuration arguments.
        
    Returns:
        InfrastructureManager: A new infrastructure manager instance.
    """
    return InfrastructureManager(
        clouds=clouds or [],
        llms=llms or {},
        **kwargs
    )


def create_infrastructure_from_config(config: dict[str, Any]) -> InfrastructureManager:
    """Create an InfrastructureManager from a configuration dictionary.
    
    This function handles the migration from old configuration format to new
    service-aware cloud provider format.
    
    Args:
        config: Configuration dictionary that may contain old-style separate
                service configs or new-style service-aware cloud configs.
                
    Returns:
        InfrastructureManager: A new infrastructure manager instance.
    """
    # If we have the new format with service-aware clouds, use it directly
    if 'clouds' in config and all(
        hasattr(cloud, 'secrets') or hasattr(cloud, 'logging') or 
        hasattr(cloud, 'pubsub') or hasattr(cloud, 'tracing')
        for cloud in config.get('clouds', [])
    ):
        return create_infrastructure_manager(
            clouds=config.get('clouds', []),
            llms=config.get('llms', {})
        )
    
    # Otherwise, migrate from old format
    from buttermilk._core.cloud_config import GCPConfig, SecretsServiceConfig, LoggingServiceConfig, PubSubServiceConfig, TracingServiceConfig
    
    migrated_clouds = []
    
    # Process existing clouds and add service configurations
    for cloud_config in config.get('clouds', []):
        if cloud_config.get('type') == 'gcp':
            # Create GCP config with integrated services
            gcp_config_data = {
                'type': 'gcp',
                'project_id': cloud_config.get('project_id'),
                'region': cloud_config.get('region', 'us-central1'),
                'location': cloud_config.get('location'),
                'storage_bucket': cloud_config.get('bucket'),
                'bigquery_dataset': cloud_config.get('bigquery_dataset', 'buttermilk'),
            }
            
            # Add secrets service if configured
            if 'secret_provider' in config and config['secret_provider'].get('type') == 'gcp':
                gcp_config_data['secrets'] = SecretsServiceConfig(
                    models_secret=config['secret_provider'].get('models_secret', 'dev__llm__connections'),
                    credentials_secret=config['secret_provider'].get('credentials_secret', 'dev__shared_credentials')
                )
            
            # Add logging service if configured
            if 'logger_cfg' in config and config['logger_cfg'].get('type') == 'gcp':
                gcp_config_data['logging'] = LoggingServiceConfig(
                    verbose=config['logger_cfg'].get('verbose', False)
                )
            
            # Add pubsub service if configured
            if 'pubsub' in config and config['pubsub'].get('type') == 'gcp':
                gcp_config_data['pubsub'] = PubSubServiceConfig(
                    jobs_topic=config['pubsub'].get('jobs_topic', 'jobs'),
                    jobs_subscription=config['pubsub'].get('jobs_subscription', 'jobs-sub'),
                    status_topic=config['pubsub'].get('status_topic', 'flow'),
                    status_subscription=config['pubsub'].get('status_subscription', 'flow-sub')
                )
            
            # Add tracing service if configured
            if 'tracing' in config and config['tracing'].get('otel', {}).get('enabled'):
                gcp_config_data['tracing'] = TracingServiceConfig(
                    enabled=True
                )
            
            migrated_clouds.append(GCPConfig(**gcp_config_data))
        
        else:
            # Keep other cloud types as-is
            migrated_clouds.append(cloud_config)
    
    return create_infrastructure_manager(
        clouds=migrated_clouds,
        llms=config.get('llms', {})
    )