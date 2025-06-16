import json

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from google.cloud import secretmanager

from buttermilk._core.config import CloudProviderCfg
from buttermilk._core.utils.lazy_loading import cached_property
from buttermilk.utils.utils import load_json_flexi


class SecretsManager(CloudProviderCfg):
    _path: str = ""

    @cached_property
    def client(self):
        """Initialize secrets manager client with project ID - deferred until first secret access."""
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
        cfg_key: str = None,  # Get the secret name from the config passed in earlier
        version: str = "latest",
    ) -> str:
        """Retrieve latest version of a secret by ID"""
        secret_name = secret_name or secret_class or getattr(self, cfg_key)

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
