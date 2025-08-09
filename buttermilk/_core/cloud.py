"""Cloud provider client management and connection utilities."""

import os
from typing import Any

from google.auth import default
from google.auth.credentials import Credentials as GoogleCredentials
from google.cloud import bigquery, storage
from google.cloud.logging_v2.client import Client as CloudLoggingClient

from buttermilk._core.config import CloudProviderCfg
from buttermilk._core.log import logger
from buttermilk._core.utils.lazy_loading import cached_property, refreshable_cached_property


class CloudManager:
    """Manages cloud provider connections and client instances."""

    def __init__(self, clouds: list[CloudProviderCfg]) -> None:
        """Initialize the cloud manager with cloud configuration.

        Args:
            clouds: List of cloud provider configurations

        """
        self.clouds = clouds or []
        self._gcp_project = ""

        # Find GCP cloud config for initialization
        self.gcp_cloud_cfg = next(
            (c for c in self.clouds if c and hasattr(c, "type") and c.type == "gcp"),
            None,
        )

        # Initialize environment variables if GCP config exists
        if self.gcp_cloud_cfg:
            # Try both 'project_id' (config field) and 'project' (legacy field)
            project_id = getattr(self.gcp_cloud_cfg, "project_id", None) or getattr(self.gcp_cloud_cfg, "project", None)
            quota_project_id = getattr(self.gcp_cloud_cfg, "quota_project_id", project_id)

            if project_id:
                os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)

            if quota_project_id:
                os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT", quota_project_id)

    def _needs_credentials_refresh(self, credentials: GoogleCredentials) -> bool:
        """Check if credentials need to be refreshed."""
        return hasattr(credentials, "valid") and not credentials.valid

    @refreshable_cached_property
    def gcp_credentials(self) -> GoogleCredentials:
        """Get Google Cloud Platform credentials.

        Returns:
            Authenticated Google credentials

        Raises:
            RuntimeError: If credentials cannot be obtained

        """
        if not self.gcp_cloud_cfg:
            raise RuntimeError("No GCP cloud configuration found")

        # Try both 'project_id' (config field) and 'project' (legacy field)
        project_id = getattr(self.gcp_cloud_cfg, "project_id", None) or getattr(self.gcp_cloud_cfg, "project", None)
        quota_project_id = getattr(self.gcp_cloud_cfg, "quota_project_id", project_id)

        if not project_id:
            raise RuntimeError("GCP project ID not specified in configuration")

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]

        try:
            credentials, project = default(
                quota_project_id=quota_project_id,
                scopes=scopes,
            )

            # Store project ID for other clients to use
            self._gcp_project = project

            return credentials
        except Exception as e:
            raise RuntimeError(f"Failed to obtain GCP credentials: {e}") from e

    def get_access_token(self) -> str:
        """Get a valid access token from GCP credentials, refreshing if needed.
        
        Returns:
            str: A valid OAuth2 access token

        """
        creds = self.gcp_credentials

        # Refresh if needed
        if not creds.valid:
            from google.auth.transport.requests import Request
            request = Request()
            creds.refresh(request)

        return creds.token

    @cached_property
    def gcs(self) -> Any | None:
        """Get Google Cloud Storage client instance.

        Returns:
            Authenticated GCS client or None if Google Cloud not available

        Raises:
            RuntimeError: If client initialization fails

        """
        if not self._gcp_project:
            # Ensure credentials are loaded to get project ID
            _ = self.gcp_credentials

        try:
            return storage.Client(
                project=self._gcp_project,
                credentials=self.gcp_credentials,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCS client: {e}") from e

    @cached_property
    def bq(self) -> Any | None:
        """Get Google BigQuery client instance.

        Returns:
            Authenticated BigQuery client or None if Google Cloud not available

        Raises:
            RuntimeError: If client initialization fails

        """
        if not self._gcp_project:
            # Ensure credentials are loaded to get project ID
            _ = self.gcp_credentials

        try:
            return bigquery.Client(
                project=self._gcp_project,
                credentials=self.gcp_credentials,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BigQuery client: {e}") from e

    def gcs_log_client(self, logger_cfg: CloudProviderCfg) -> Any | None:
        """Get Google Cloud Logging client instance.

        Args:
            logger_cfg: Logger configuration with project information

        Returns:
            Authenticated Cloud Logging client

        Raises:
            RuntimeError: If client initialization fails

        """
        if not logger_cfg:
            raise RuntimeError("Logger config needed for GCS Log Client")

        project = logger_cfg.project_id
        if not project:
            raise RuntimeError("Logger config missing 'project' attribute")

        try:
            return CloudLoggingClient(
                project=project,
                credentials=self.gcp_credentials,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Cloud Logging client: {e}") from e

    def login_clouds(self) -> None:
        """Initialize cloud provider connections."""
        for cloud in self.clouds:
            if not cloud or not hasattr(cloud, "type"):
                continue  # Skip invalid cloud entries

            if cloud.type == "vertex":
                self._init_vertex_ai(cloud)

    def _init_vertex_ai(self, cloud: CloudProviderCfg) -> None:
        """Initialize Vertex AI connection."""
        from vertexai import init as aiplatform_init

        # Ensure required attributes exist
        project_id = getattr(cloud, "project_id", None)
        location = getattr(cloud, "location", None)
        bucket = getattr(cloud, "bucket", None)

        if project_id and location and bucket:
            try:
                aiplatform_init(
                    project=project_id,
                    location=location,
                    staging_bucket=bucket,
                )
                logger.info(f"Initialized Vertex AI: project={project_id}, location={location}")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}")
        else:
            raise FatalError(
                "Unable to complete Vertex AI initialization due to missing project, location, or bucket in config.",
            )
