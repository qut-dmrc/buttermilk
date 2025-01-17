from hashlib import md5

from cloudpathlib import CloudPath

from buttermilk.utils.save import upload_binary, upload_text
from buttermilk.utils.utils import read_file


def test_logger_initialised(bm):
    obj = bm.logger
    assert obj is not None
    assert len(obj.handlers) >= 2


def test_save(bm):
    uri = bm.save(data=["test data"], extension=".txt")
    assert uri.startswith("gs://")
    assert uri.startswith(bm.save_dir)
    assert uri.endswith(".txt")
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == '{"0": "test data"}'
    uploaded.unlink(missing_ok=False)


def test_upload_text(bm):
    uri = bm.save_dir + "/test_data"
    uri = upload_text(data="test data", uri=uri, extension=".txt")
    assert uri.startswith(bm.save_dir)
    assert uri.endswith(".txt")
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == "test data"
    uploaded.unlink(missing_ok=False)


def test_save_binary(bm):
    save_dir = bm.save_dir
    assert save_dir

    try:
        with open("tests/data/Rijksmuseum_(25621972346).jpg", "rb") as img:
            uri = upload_binary(img, save_dir=save_dir)
            img.seek(0)
            uploaded_bytes = img.read()
            assert uri is not None

        downloaded = read_file(uri)
        assert md5(downloaded).hexdigest() == md5(uploaded_bytes).hexdigest()
    finally:
        # delete
        CloudPath(uri).unlink()
