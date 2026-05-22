"""Unified cloud configuration system for consistent provider management.

This module provides a consistent configuration structure for all cloud providers,
replacing the current inconsistent approach where different services have different
configuration patterns.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


class CloudProviderConfig(BaseModel, ABC):
    """Base configuration for all cloud providers.

    Provides common fields and validation that all cloud providers should support.
    """

    type: str = Field(description="Cloud provider type")
    project_id: str | None = Field(default=None, description="Primary project/account identifier")
    region: str | None = Field(default=None, description="Default region for resources")
    credentials: dict[str, Any] = Field(default_factory=dict, description="Provider-specific credential configuration")

    model_config = {
        "extra": "allow",  # Allow provider-specific fields
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    @abstractmethod
    def get_client_config(self, service: str) -> dict[str, Any]:
        """Get configuration for specific service client.

        Args:
            service: Service name (e.g., 'bigquery', 'storage', 'pubsub')

        Returns:
            Configuration dict for the service client
        """


class SecretsServiceConfig(BaseModel):
    """Configuration for secrets management service."""

    models_secret: str = Field(default="dev__llm__connections", description="Secret name for LLM API keys")
    credentials_secret: str = Field(
        default="dev__shared_credentials",
        description="Secret name for shared credentials",
    )


class LoggingServiceConfig(BaseModel):
    """Configuration for cloud logging service."""

    verbose: bool = Field(default=False, description="Enable verbose logging")


class TracingServiceConfig(BaseModel):
    """Configuration for OpenTelemetry tracing service."""

    enabled: bool = Field(default=True, description="Enable OpenTelemetry tracing")


class VertexServiceConfig(BaseModel):
    """Vertex AI service configuration."""

    enabled: bool = Field(default=True, description="Enable Vertex AI service")


class GCPConfig(CloudProviderConfig):
    """Google Cloud Platform configuration with integrated services."""

    type: Literal["gcp"] = "gcp"
    project_id: str | None = Field(
        default=None,
        description="GCP Project ID (auto-detected from GOOGLE_CLOUD_PROJECT)",
    )
    quota_project_id: str | None = Field(default=None, description="Quota project for billing (defaults to project_id)")
    region: str = Field(default="us-central1", description="Default GCP region")
    location: str | None = Field(default=None, description="Default location (defaults to region)")

    # Core storage configurations
    storage_bucket: str | None = Field(default=None, description="Default GCS bucket for storage operations")
    bigquery_dataset: str = Field(default="buttermilk", description="Default BigQuery dataset")

    # Integrated service configurations
    secrets: SecretsServiceConfig | None = Field(default=None, description="Secrets management configuration")
    logging: LoggingServiceConfig | None = Field(default=None, description="Cloud logging configuration")
    tracing: TracingServiceConfig | None = Field(default=None, description="OpenTelemetry tracing configuration")
    vertex: VertexServiceConfig | None = Field(default=None, description="Vertex AI service configuration")

    @model_validator(mode="after")
    def set_defaults_from_env(self) -> "GCPConfig":
        """Set defaults from environment variables and populate service configs."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

        if not self.quota_project_id:
            self.quota_project_id = self.project_id

        if not self.location:
            self.location = self.region

        return self

    def get_client_config(self, service: str) -> dict[str, Any]:
        """Get GCP service client configuration."""
        base_config = {
            "project_id": self.project_id,
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
            "logging": {
                **base_config,
                "resource": {
                    "type": "global",
                    "labels": {"project_id": self.project_id},
                },
                **(self.logging.model_dump() if self.logging else {}),
            },
            "secretmanager": {
                **base_config,
                "type": "gcp",
                **(self.secrets.model_dump() if self.secrets else {}),
            },
            "tracing": {
                **base_config,
                **(self.tracing.model_dump() if self.tracing else {}),
            },
            "vertex": {
                **base_config,
                "bucket": self.storage_bucket,
                **(self.vertex.model_dump() if self.vertex else {}),
            },
        }

        return service_configs.get(service, base_config)

    def has_service(self, service: str) -> bool:
        """Check if this cloud provider has a specific service configured."""
        service_map = {
            "secrets": self.secrets,
            "logging": self.logging,
            "tracing": self.tracing,
            "vertex": self.vertex,
        }
        return service_map.get(service) is not None


class AWSConfig(CloudProviderConfig):
    """Amazon Web Services configuration."""

    type: Literal["aws"] = "aws"
    account_id: str | None = Field(
        default=None,
        alias="project_id",  # Map to common field
        description="AWS Account ID",
    )
    region: str = Field(default="us-east-1", description="Default AWS region")

    def get_client_config(self, service: str) -> dict[str, Any]:
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
    subscription_id: str | None = Field(
        default=None,
        alias="project_id",  # Map to common field
        description="Azure Subscription ID",
    )
    resource_group: str | None = Field(default=None, description="Default resource group")
    region: str = Field(default="eastus", description="Default Azure region")

    def get_client_config(self, service: str) -> dict[str, Any]:
        """Get Azure service client configuration."""
        return {
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "location": self.region,
        }


class LoggerConfig(BaseModel):
    """Configuration for cloud logging providers."""

    type: Literal["gcp", "aws", "azure", "local"] = Field(description="Logging provider type")
    project_id: str | None = Field(default=None, description="Cloud project ID for logging")
    location: str | None = Field(default=None, description="Logging location/region")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    console: bool = Field(
        default=True,
        description="Enable console logging to stderr. Logs go to stderr by default (Python best practice), making buttermilk MCP-compatible without configuration.",
    )

    @model_validator(mode="after")
    def set_project_from_env(self) -> "LoggerConfig":
        """Set project from environment if not specified."""
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        return self


class RunInfoConfig(BaseModel):
    """Configuration for run execution information."""

    platform: Literal["local", "cloud", "batch"] = Field(default="local", description="Execution platform")
    flow_api: str | None = Field(default=None, description="Base URL for flow API")
    save_dir_base: str | None = Field(default=None, description="Base directory/URI for saving results")


class TracingConfig(BaseModel):
    """Configuration for experiment tracing.

    Note: This config is deprecated. Tracing is now configured via
    infrastructure.tracing in the main config using OpenTelemetry.
    """

    enabled: bool = Field(default=True, description="Enable tracing (deprecated)")
    provider: Literal["otel"] = Field(default="otel", description="Tracing provider (otel only)")


# Union type for all cloud providers
CloudProvider = Union[GCPConfig, AWSConfig, AzureConfig]
