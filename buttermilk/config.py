# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

def get_config(config_dir: str = 'conf', config_name: str = 'config') -> dict:
    """Load the configuration from the given directory and file.

    Args:
        config_dir: The directory containing the configuration file.
        config_name: The name of the configuration file.

    Returns:
        The configuration dictionary.
    """
    with initialize(config_path=config_dir):
        cfg = compose(config_name=config_name)
        return cfg
