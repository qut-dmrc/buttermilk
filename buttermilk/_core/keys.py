import json
from typing import Any

# Optional Azure imports - fail gracefully if not available
try:
    from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]  # azure-identity: no stubs available
    from azure.keyvault.secrets import SecretClient  # type: ignore[import-not-found]  # azure-keyvault-secrets: no stubs available

    AZURE_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None  # type: ignore[assignment,misc]
    SecretClient = None  # type: ignore[assignment,misc]
    AZURE_AVAILABLE = False

# Optional Google Cloud imports - fail gracefully if not available
try:
    from google.cloud import secretmanager

    GCP_SECRETS_AVAILABLE = True
except ImportError:
    secretmanager = None  # type: ignore[assignment]
    GCP_SECRETS_AVAILABLE = False

import os

from buttermilk._core.config import CloudProviderCfg
from buttermilk._core.utils.lazy_loading import cached_property
from buttermilk.utils.utils import load_json_flexi


class SecretsManager(CloudProviderCfg):
    _path: str = ""
    vault: str | None = None  # Azure Key Vault URL (required when type == "azure")

    @cached_property
    def client(self) -> Any:
        """Initialize secrets manager client with project ID - deferred until first secret access."""
        client = None

        if self.type == "gcp":
            if not GCP_SECRETS_AVAILABLE:
                raise ImportError("Google Cloud Secret Manager not available. Install google-cloud-secret-manager.")
            client = secretmanager.SecretManagerServiceClient()
            self._path = f"projects/{self.project_id}/secrets"
        elif self.type == "azure":
            if not AZURE_AVAILABLE:
                raise ImportError("Azure Key Vault not available. Install azure-identity and azure-keyvault-secrets.")
            client = SecretClient(
                vault_url=self.vault,
                credential=DefaultAzureCredential(),
            )
        return client

    # @lru_cache
    def get_secret(
        self,
        secret_name: str | None = None,
        secret_class: str | None = None,
        cfg_key: str | None = None,  # Get the secret name from the config passed in earlier
        version: str = "latest",
    ) -> str:
        """Retrieve latest version of a secret by ID"""

        # First, check environment variables.
        if cfg_key and (env_var := os.environ.get(cfg_key)):
            return env_var

        secret_name = secret_name or secret_class or (getattr(self, cfg_key) if cfg_key is not None else None)
        if not secret_name:
            raise ValueError("get_secret requires one of secret_name, secret_class, or cfg_key to resolve a secret name")

        _client = self.client

        name = "/".join([x for x in [self._path, secret_name, "versions", version or "latest"] if x])

        try:
            response = _client.access_secret_version(request={"name": name})
            response = response.payload.data.decode("UTF-8")
        except Exception as e:
            raise OSError(f"Unable to access secret {name}: {e}, {e.args}")

        try:
            response = load_json_flexi(response)

        except json.JSONDecodeError:
            pass
        return response
