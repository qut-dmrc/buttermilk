
from buttermilk.buttermilk import BM


def test_has_test_info(bm: BM):
    assert bm.cfg.name == "buttermilk"
    assert bm.cfg.job == "testing"

def test_config_obj(bm: BM):
    assert "secret_provider" in bm.cfg.project
    assert "models_secret" in bm.cfg.project

def test_config_models_azure(bm: BM):
    models = bm._connections_azure
    assert models

def test_save_dir(bm):
    assert bm.save_dir.startswith("gs://")
    uri = bm.save(["test"])
    assert uri.startswith(bm.save_dir)


def test_singleton():
    obj1 = BM()
    obj2 = BM()

    assert id(obj1) == id(obj2), "variables contain different instances."

def test_singleton_from_fixture(bm):
    obj2 = BM()

    assert id(bm) == id(obj2), "variables contain different instances."

def test_time_to_instantiate():
    import time
    start = time.time()
    obj = BM()
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f} seconds")
    assert time_taken < 1, "Took too long to instantiate BM"
