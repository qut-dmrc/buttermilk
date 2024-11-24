from functools import lru_cache
from typing import Self

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from google.cloud import secretmanager
from pydantic import model_validator

from buttermilk._core.config import CloudProviderCfg


class SecretsManager(CloudProviderCfg):
    _path: str
    _client: secretmanager.SecretManagerServiceClient | SecretClient

    @model_validator(mode="after")
    def _load_client(self) -> Self:
        """Initialize secrets manager client with project ID"""
        if self.type == "gcp":
            self._client = secretmanager.SecretManagerServiceClient()
            self._path = f"projects/{self.project}/secrets/"
        elif self.type == "azure":
            self._client = SecretClient(
                vault_url=self.vault,
                credential=DefaultAzureCredential(),
            )
        return self

    @lru_cache
    def get_secret(
        self,
        secret_name: str = None,
        secret_class: str = None,
        version: str = "latest",
    ) -> str | None:
        """Retrieve latest version of a secret by ID"""
        if secret_class and not secret_name:
            secret_name = getattr(self, secret_class)
        try:
            name = f"{self._path}/{secret_name}/versions/{version}"
            response = self._client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            return None
