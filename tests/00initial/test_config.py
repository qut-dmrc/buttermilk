
from buttermilk.buttermilk import BM


def test_config_obj(bm: BM):
    assert "secret_name" in bm.cfg['project']

def test_config_models_azure(bm):
    models = bm._connections_azure
    assert models


