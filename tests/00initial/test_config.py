
from cloudpathlib import AnyPath

from buttermilk.bm import BM


def test_has_test_info(bm: BM):
    assert bm.cfg.name == "buttermilk"
    assert bm.cfg.job == "testing"
    assert bm.save_dir is not None
    assert bm.save_dir != ""


def test_config_llms(bm: BM):
    models = bm.llms
    assert models


def test_save_dir(bm):
    assert "runs/buttermilk/testing/" in bm.save_dir
    assert AnyPath(bm.save_dir)


def test_singleton(bm):
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


def test_model_post_init_saves_config(bm, mock_save):
    mock_save.assert_called_once()
    args, kwargs = mock_save.call_args

    saved_data = kwargs.get("data")

    assert len(saved_data) == 2

    assert isinstance(saved_data[0], dict)  # Ensure the first item is the config
    assert isinstance(saved_data[1], dict)  # Ensure the second item is the run metadata

    assert kwargs.get("basename") == "config"
    assert kwargs.get("extension") == "json"
