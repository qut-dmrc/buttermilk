from datatools.gcloud import GCloud


def test_singleton():
    obj1 = GCloud()
    obj2 = GCloud()

    assert id(obj1) == id(obj2), "variables contain different instances."

def test_singleton_from_fixture(gc):
    obj2 = GCloud()

    assert id(gc) == id(obj2), "variables contain different instances."

def test_time_to_instantiate_gc():
    import time
    start = time.time()
    gc = GCloud()
    end = time.time()
    assert (end - start) < 1, "Took too long to instantiate GCloud"

def test_has_test_info(gc: GCloud):
    assert gc.name == "testing"
    assert gc.job == "testing"


def test_save_binary(gc: GCloud):
    with open("datatools/tests/data/sample_image.png", "rb") as img:
        uri = gc.upload_binary(img)
        assert uri is not None

    gc.gcs._delete_resource(uri)
    pass


def test_logger_initialised(gc: GCloud):
    obj = gc.logger
    assert obj is not None
    assert obj.already_setup
    assert len(obj.handlers) >= 2
