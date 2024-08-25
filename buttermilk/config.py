# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

from functools import cached_property
import json
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from typing import Any, Optional
from pathlib import Path
import pydantic

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import BaseModel, Field, root_validator, validator

CONFIG_CACHE_PATH='~/.cache/buttermilk/.models.json'
class Config(BaseModel):
    credential_auzre: Any = Field(default_factory=DefaultAzureCredential)
    cfg: DictConfig = Field(default_factory=lambda: Config.get_config())
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_config(cls, config_dir: str = 'conf', config_name: str = 'config') -> DictConfig:
        """Load the configuration from the given directory and file.

        Args:
            config_dir: The directory containing the configuration file.
            config_name: The name of the configuration file.

        Returns:
            The configuration dictionary.
        """
        with initialize(config_path=config_dir):
            cfg= compose(config_name=config_name)
        return cfg

    @cached_property
    def _connections_azure(self) -> Any:
        # Model definitions are stored in Azure Secrets, and loaded here.

        try:
            contents = Path(CONFIG_CACHE_PATH).read_text()
        except Exception as e:
            pass
        auth = DefaultAzureCredential()
        vault_uri = self.cfg['project']['azure']['vault']
        models_secret = self.cfg['project']['models_secret']
        secrets = SecretClient(
                vault_uri,
                credential=auth
            )
        contents = secrets.get_secret(models_secret).value
        if not contents:
            raise ValueError("Could not load secrets from Azure vault")
        try:
            Path(CONFIG_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(CONFIG_CACHE_PATH).write_text(contents)
        except:
            pass

        connections = json.loads(contents)
        connections = { conn['name']: conn for conn in connections }

        return connections
