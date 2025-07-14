"""Unified cloud configuration system for consistent provider management.

This module provides a consistent configuration structure for all cloud providers,
replacing the current inconsistent approach where different services have different
configuration patterns.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class CloudProviderConfig(BaseModel, ABC):
    """Base configuration for all cloud providers.
    
    Provides common fields and validation that all cloud providers should support.
    """

    type: str = Field(description="Cloud provider type")
    project_id: Optional[str] = Field(
        default=None,
        description="Primary project/account identifier"
    )
    region: Optional[str] = Field(
        default=None,
        description="Default region for resources"
    )
    credentials: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific credential configuration"
    )

    model_config = {
        "extra": "allow",  # Allow provider-specific fields
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    @abstractmethod
    def get_client_config(self, service: str) -> Dict[str, Any]:
        """Get configuration for specific service client.
        
        Args:
            service: Service name (e.g., 'bigquery', 'storage', 'pubsub')
            
        Returns:
            Configuration dict for the service client
        """
        pass


class GCPConfig(CloudProviderConfig):
    """Google Cloud Platform configuration."""

    type: Literal["gcp"] = "gcp"
    project_id: Optional[str] = Field(
        default=None,
        description="GCP Project ID (auto-detected from GOOGLE_CLOUD_PROJECT)"
    )
    quota_project_id: Optional[str] = Field(
        default=None,
        description="Quota project for billing (defaults to project_id)"
    )
    region: str = Field(
        default="us-central1",
        description="Default GCP region"
    )
    location: Optional[str] = Field(
        default=None,
        description="Default location (defaults to region)"
    )

    # Service-specific configurations
    storage_bucket: Optional[str] = Field(
        default=None,
        description="Default GCS bucket for storage operations"
    )
    bigquery_dataset: str = Field(
        default="buttermilk",
        description="Default BigQuery dataset"
    )

    @model_validator(mode="after")
    def set_defaults_from_env(self) -> "GCPConfig":
        """Set defaults from environment variables."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

        if not self.quota_project_id:
            self.quota_project_id = self.project_id

        if not self.location:
            self.location = self.region

        return self

    def get_client_config(self, service: str) -> Dict[str, Any]:
        """Get GCP service client configuration."""
        base_config = {
            "project": self.project_id,
            "location": self.location,
        }

        service_configs = {
            "bigquery": {
                **base_config,
                "default_query_job_config": {
                    "use_legacy_sql": False,
                },
            },
            "storage": {
                **base_config,
                "default_bucket": self.storage_bucket,
            },
            "pubsub": {
                **base_config,
            },
            "logging": {
                **base_config,
                "resource": {
                    "type": "global",
                    "labels": {"project_id": self.project_id},
                },
            },
            "secretmanager": {
                **base_config,
            },
        }

        return service_configs.get(service, base_config)


class VertexAIConfig(CloudProviderConfig):
    """Vertex AI specific configuration (extends GCP)."""

    type: Literal["vertex"] = "vertex"
    project_id: Optional[str] = Field(
        default=None,
        description="GCP Project ID for Vertex AI"
    )
    region: str = Field(
        default="us-central1",
        description="Vertex AI region"
    )
    location: Optional[str] = Field(
        default=None,
        description="Vertex AI location (defaults to region)"
    )

    @model_validator(mode="after")
    def set_defaults_from_env(self) -> "VertexAIConfig":
        """Set defaults from environment variables."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

        if not self.location:
            self.location = self.region

        return self

    def get_client_config(self, service: str) -> Dict[str, Any]:
        """Get Vertex AI client configuration."""
        return {
            "project": self.project_id,
            "location": self.location,
        }


class AWSConfig(CloudProviderConfig):
    """Amazon Web Services configuration."""

    type: Literal["aws"] = "aws"
    account_id: Optional[str] = Field(
        default=None,
        alias="project_id",  # Map to common field
        description="AWS Account ID"
    )
    region: str = Field(
        default="us-east-1",
        description="Default AWS region"
    )

    def get_client_config(self, service: str) -> Dict[str, Any]:
        """Get AWS service client configuration."""
        base_config = {
            "region_name": self.region,
        }

        service_configs = {
            "s3": base_config,
            "secretsmanager": base_config,
            "cloudwatch": base_config,
        }

        return service_configs.get(service, base_config)


class AzureConfig(CloudProviderConfig):
    """Microsoft Azure configuration."""

    type: Literal["azure"] = "azure"
    subscription_id: Optional[str] = Field(
        default=None,
        alias="project_id",  # Map to common field
        description="Azure Subscription ID"
    )
    resource_group: Optional[str] = Field(
        default=None,
        description="Default resource group"
    )
    region: str = Field(
        default="eastus",
        description="Default Azure region"
    )

    def get_client_config(self, service: str) -> Dict[str, Any]:
        """Get Azure service client configuration."""
        return {
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "location": self.region,
        }


class SecretProviderConfig(BaseModel):
    """Configuration for secret management providers."""

    type: Literal["gcp", "aws", "azure", "local"] = Field(
        description="Secret provider type"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Cloud project/account ID for secrets"
    )
    models_secret: Optional[str] = Field(
        default=None,
        description="Secret name for LLM API keys"
    )
    credentials_secret: Optional[str] = Field(
        default=None,
        description="Secret name for shared credentials"
    )

    @model_validator(mode="after")
    def set_project_from_env(self) -> "SecretProviderConfig":
        """Set project from environment if not specified."""
        if not self.project_id and self.type == "gcp":
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self


class LoggerConfig(BaseModel):
    """Configuration for cloud logging providers."""

    type: Literal["gcp", "aws", "azure", "local"] = Field(
        description="Logging provider type"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Cloud project ID for logging"
    )
    location: Optional[str] = Field(
        default=None,
        description="Logging location/region"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )

    @model_validator(mode="after")
    def set_project_from_env(self) -> "LoggerConfig":
        """Set project from environment if not specified."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self


class PubSubConfig(BaseModel):
    """Configuration for pub/sub messaging providers."""

    type: Literal["gcp", "aws", "azure"] = Field(
        description="Pub/Sub provider type"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Cloud project ID for pub/sub"
    )
    jobs_topic: str = Field(
        default="jobs",
        description="Topic name for job messages"
    )
    jobs_subscription: str = Field(
        default="jobs-sub",
        description="Subscription name for job messages"
    )
    status_topic: str = Field(
        default="flow",
        description="Topic name for status messages"
    )
    status_subscription: str = Field(
        default="flow-sub",
        description="Subscription name for status messages"
    )

    @model_validator(mode="after")
    def set_project_from_env(self) -> "PubSubConfig":
        """Set project from environment if not specified."""
        if not self.project_id and self.type == "gcp":
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self


class RunInfoConfig(BaseModel):
    """Configuration for run execution information."""

    platform: Literal["local", "cloud", "batch"] = Field(
        default="local",
        description="Execution platform"
    )
    flow_api: Optional[str] = Field(
        default=None,
        description="Base URL for flow API"
    )
    save_dir_base: Optional[str] = Field(
        default=None,
        description="Base directory/URI for saving results"
    )


class TracingConfig(BaseModel):
    """Configuration for experiment tracing."""

    enabled: bool = Field(
        default=True,
        description="Enable tracing"
    )
    provider: Literal["weave", "wandb", "mlflow"] = Field(
        default="weave",
        description="Tracing provider"
    )


# Union type for all cloud providers
CloudProvider = Union[GCPConfig, VertexAIConfig, AWSConfig, AzureConfig]


class InfrastructureConfig(BaseModel):
    """Complete infrastructure configuration for Buttermilk.
    
    This replaces the current scattered cloud configuration approach
    with a unified, composable system.
    """

    # Core cloud providers
    clouds: List[CloudProvider] = Field(
        default_factory=list,
        description="List of configured cloud providers"
    )

    # Service configurations
    secret_provider: Optional[SecretProviderConfig] = Field(
        default=None,
        description="Secret management configuration"
    )
    logger_cfg: Optional[LoggerConfig] = Field(
        default=None,
        description="Logging configuration"
    )
    pubsub: Optional[PubSubConfig] = Field(
        default=None,
        description="Pub/Sub messaging configuration"
    )

    # Execution configuration
    run_info: RunInfoConfig = Field(
        default_factory=RunInfoConfig,
        description="Run execution configuration"
    )
    tracing: TracingConfig = Field(
        default_factory=TracingConfig,
        description="Experiment tracing configuration"
    )

    def get_cloud_config(self, provider_type: str) -> Optional[CloudProvider]:
        """Get configuration for a specific cloud provider."""
        for cloud in self.clouds:
            if cloud.type == provider_type:
                return cloud
        return None

    def get_primary_cloud(self) -> Optional[CloudProvider]:
        """Get the primary (first) cloud provider configuration."""
        return self.clouds[0] if self.clouds else None
