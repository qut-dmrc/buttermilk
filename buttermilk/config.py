# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

import json
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from typing import Any, Optional
from pathlib import Path
import pydantic

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import BaseModel, Field

SECRET_MODELS_PATH = 'connections'

class Config(BaseModel):
    credential_auzre: Any = Field(default_factory=DefaultAzureCredential)
    cfg: dict = None

    @pydantic.field_validator
    def get_config(self, config_dir: str = 'conf', config_name: str = 'config') -> dict:
        """Load the configuration from the given directory and file.

        Args:
            config_dir: The directory containing the configuration file.
            config_name: The name of the configuration file.

        Returns:
            The configuration dictionary.
        """
        with initialize(config_path=config_dir):
            self.cfg = compose(config_name=config_name)
            

    def _connections_azure(self) -> Any:
        # Model definitions are stored in Azure Secrets, and loaded here.

        try:
            contents = Path("conf/models.json").read_text()
        except:
            pass
        auth = DefaultAzureCredential()
        secrets = SecretClient(
                SECRET_MODELS_PATH,
                credential=auth
            )
        contents = secrets.get_secret(SECRET_MODELS_PATH).value
        if not contents:
            raise ValueError("Could not load secrets from Azure vault")
        try:
            Path("conf/models.json").write_text(contents)
        except:
            pass

        connections = json.loads(contents)
        connections = { conn['name']: conn for conn in connections }

        return connections
