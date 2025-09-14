"""Infrastructure management for shared Buttermilk components.

This module provides the InfrastructureManager class that handles creation
and management of shared infrastructure components (CloudManager, SecretsManager, 
LLMs) that can be injected into session-scoped BM instances.
"""

from __future__ import annotations

from typing import Any

import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf
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
    execution_context: Any = Field(
        default=None,
        description="Optional ExecutionContext to share infrastructure with."
    )
    
    # Private infrastructure instances
    _cloud_manager: Any = pydantic.PrivateAttr(default=None)
    _secret_manager: Any = pydantic.PrivateAttr(default=None)
    _llms_instance: Any = pydantic.PrivateAttr(default=None)
    _query_runner: Any = pydantic.PrivateAttr(default=None)
    
    @cached_property
    def cloud_manager(self) -> Any:
        """Get or create the CloudManager instance."""
        # Use ExecutionContext's cloud manager if available
        if self.execution_context is not None:
            return self.execution_context.cloud_manager
            
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
        # Use ExecutionContext's secret manager if available
        if self.execution_context is not None:
            return self.execution_context.secret_manager
            
        if self._secret_manager is None:
            # Find cloud provider with secrets service configured
            secrets_cloud = self._find_cloud_with_service("secrets")
            if not secrets_cloud:
                raise RuntimeError("No cloud provider with secrets service configured.")

            from buttermilk._core.keys import SecretsManager
            
            # Get secrets configuration from cloud provider
            secrets_config = secrets_cloud.get_client_config("secretmanager")
            self._secret_manager = SecretsManager(**secrets_config)
            logger.debug(f"SecretsManager initialized with {secrets_cloud.type} provider")
        
        return self._secret_manager
    
    @cached_property
    def llms_instance(self) -> Any:
        """Get or create the LLMs instance."""
        # Use ExecutionContext's LLMs instance if available
        if self.execution_context is not None:
            return self.execution_context.llms
            
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
        # Use ExecutionContext's query runner if available
        if self.execution_context is not None:
            return self.execution_context.query_runner
            
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
        # Check ExecutionContext's clouds first if available
        clouds_to_check = []
        if self.execution_context is not None:
            clouds_to_check = self.execution_context.clouds
        else:
            clouds_to_check = self.clouds
            
        for cloud in clouds_to_check:
            if hasattr(cloud, "has_service") and cloud.has_service(service):
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
        
        logger.info("Infrastructure initialization completed")
    
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
        
        # Get logger configuration if logging cloud is available
        logger_cfg = None
        logging_cloud = self._find_cloud_with_service("logging")
        if logging_cloud:
            try:
                from buttermilk._core.config import LoggerConfig
                logging_config = logging_cloud.get_client_config("logging")
                
                # Create a proper LoggerConfig object
                logger_cfg = LoggerConfig(
                    type=logging_cloud.type,
                    **logging_config
                )
            except Exception as e:
                logger.warning(f"Failed to create logger config: {e}")
        
        return create_session_bm(
            name=name,
            job=job,
            batch_id=batch_id,
            platform=platform,
            save_dir_base=save_dir_base,
            cloud_manager=self.cloud_manager if self.clouds else None,
            secret_manager=self.secret_manager if self._find_cloud_with_service("secrets") else None,
            llms_instance=self.llms_instance,
            query_runner=self.query_runner if self.clouds else None,
            logger_cfg=logger_cfg,
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


def create_infrastructure_from_config(config: DictConfig | dict[str, Any]) -> InfrastructureManager:
    """Create an InfrastructureManager from a configuration dictionary.
    
    This function properly instantiates cloud provider configurations using Hydra
    to ensure they have the required methods for service detection.
    
    Args:
        config: Configuration dictionary containing cloud and LLM configurations.
                
    Returns:
        InfrastructureManager: A new infrastructure manager instance with hydrated clouds.
    """
    hydrated_clouds = []
    
    # Process cloud configurations and instantiate them properly
    for cloud_config in config.get("clouds", []):
        try:
            # If it's a DictConfig, instantiate it using Hydra
            if isinstance(cloud_config, DictConfig):
                hydrated_cloud = hydra.utils.instantiate(cloud_config)
                hydrated_clouds.append(hydrated_cloud)
            else:
                # Already instantiated - use as-is
                hydrated_clouds.append(cloud_config)
        except Exception as e:
            logger.error(f"Failed to instantiate cloud config {cloud_config}: {e}")
            raise RuntimeError(f"Cannot instantiate cloud provider: {e}") from e
    
    # Also instantiate LLM configs if they're DictConfigs
    llms_config = config.get("llms", {})
    if isinstance(llms_config, DictConfig):
        try:
            # Convert DictConfig to regular dict for LLMs
            llms_config = OmegaConf.to_container(llms_config, resolve=True)
        except Exception as e:
            logger.error(f"Failed to convert LLMs config: {e}")
            raise RuntimeError(f"Cannot process LLMs configuration: {e}") from e
    
    return create_infrastructure_manager(
        clouds=hydrated_clouds,
        llms=llms_config
    )
