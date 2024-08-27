

def test_save_binary(gc):
    with open("datatools/tests/data/sample_image.png", "rb") as img:
        uri = gc.upload_binary(img)
        assert uri is not None

    gc.gcs._delete_resource(uri)
    pass


def test_logger_initialised(gc):
    obj = gc.logger
    assert obj is not None
    assert obj.already_setup
    assert len(obj.handlers) >= 2
