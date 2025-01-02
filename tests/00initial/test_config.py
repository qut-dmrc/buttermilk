
from cloudpathlib import CloudPath

from buttermilk.bm import BM
from buttermilk.utils.save import upload_text


def test_has_test_info(bm: BM):
    assert bm.cfg.name == "buttermilk"
    assert bm.cfg.job == "testing"
    assert bm.save_dir is not None
    assert bm.save_dir != ""
    assert bm.save_dir.startswith("gs://")

def test_config_llms(bm: BM):
    models = bm.llms
    assert models


def test_save_dir(bm):
    assert bm.save_dir.startswith("gs://")
    uri = bm.save(["test"])
    assert uri.startswith(bm.save_dir)
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    uploaded.unlink(missing_ok=False)


def test_upload_text(bm):
    save_dir = bm.save_dir
    uri = upload_text(data="test data", save_dir=save_dir, extension=".txt")
    assert uri.startswith("gs://")
    assert uri.startswith(bm.save_dir)
    assert uri.endswith(".txt")
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == "test data"
    uploaded.unlink(missing_ok=False)


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
