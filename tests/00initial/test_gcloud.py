from hashlib import md5

from cloudpathlib import AnyPath, CloudPath
from google.cloud import aiplatform
from shortuuid import uuid

from buttermilk.utils.save import upload_binary, upload_text
from buttermilk.utils.utils import read_file

bm = bm


def test_logger_initialised(bm):
    obj = bm.logger
    assert obj is not None
    assert len(obj.handlers) >= 1


def test_save(bm):
    uri = bm.save(data=["test data"], extension=".txt")
    assert uri.startswith(bm.save_dir)
    assert uri.endswith(".txt")
    uploaded = AnyPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == '"test data"'
    uploaded.unlink(missing_ok=False)


def test_upload_text(bm):
    uri = f"gs://{bm.clouds[0].bucket}/test_data/{uuid}.txt"
    return_uri = upload_text(data="test data", uri=uri)
    assert return_uri == uri
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == "test data"
    uploaded.unlink(missing_ok=False)


def test_save_binary(bm):
    uri = f"gs://{bm.clouds[0].bucket}/test_data/{uuid}.txt"
    try:
        with open("tests/data/Rijksmuseum_(25621972346).jpg", "rb") as img:
            return_uri = upload_binary(img, uri=uri)
            assert return_uri == uri
            img.seek(0)
            uploaded_bytes = img.read()
            assert uri is not None

        downloaded = read_file(uri)
        assert md5(downloaded).hexdigest() == md5(uploaded_bytes).hexdigest()
    finally:
        # delete
        CloudPath(uri).unlink()


def test_vertex_setup(bm):
    # Try a simple operation like listing models
    models = aiplatform.Model.list()
    assert models is not None
