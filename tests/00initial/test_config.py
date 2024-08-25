
from buttermilk.config import Config


def test_config_obj(config):
    conf = config.get_config()
    assert "secret_name" in conf['project']

def test_config_models_azure(config: Config):
    models = config._connections_azure
    assert models


