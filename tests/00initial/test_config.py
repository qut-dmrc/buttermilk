
from cloudpathlib import AnyPath

from buttermilk.bm import BM


def test_has_test_info(bm: BM):
    assert bm.run_info.name == "buttermilk"
    assert bm.run_info.job == "testing"
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
