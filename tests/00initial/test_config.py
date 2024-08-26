
from buttermilk.buttermilk import BM


def test_config_obj(bm: BM):
    assert "secret_provider" in bm._cfg['project']
    assert "models_secret" in bm._cfg['project']

def test_config_models_azure(bm: BM):
    models = bm._connections_azure
    assert models

def test_save_dir():
    TEST_DIR = "/tmp/test.txt"
    bm = BM(save_dir=TEST_DIR)
    assert bm.save_dir ==TEST_DIR