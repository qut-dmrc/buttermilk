from functools import lru_cache
import json
from typing import Self

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from google.cloud import secretmanager
from pydantic import model_validator

from buttermilk._core.config import CloudProviderCfg
from buttermilk import logger

class SecretsManager(CloudProviderCfg):
    _path: str = ''

    @property
    def client(self):
        """Initialize secrets manager client with project ID"""
        client = None

        if self.type == "gcp":
            client = secretmanager.SecretManagerServiceClient()
            self._path = f"projects/{self.project}/secrets"
        elif self.type == "azure":
            client = SecretClient(
                vault_url=self.vault,
                credential=DefaultAzureCredential(),
            )
        return client

    # @lru_cache
    def get_secret(
        self,
        secret_name: str = None,
        secret_class: str = None,
        version: str = "latest",
    ) -> str | None:
        """Retrieve latest version of a secret by ID"""
        if secret_class and not secret_name:
            secret_name = secret_class
        
        _client = self.client

        name = "/".join([x for x in [self._path, secret_name, "versions", version or "latest"] if x ])

        try:
            response = _client.access_secret_version(request={"name": name})
            response = response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Unable to access secret {name}: {e}")
            return None

        try:
            response = json.loads(response)
            
        except json.JSONDecodeError:
            pass    
        return response