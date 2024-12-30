from hashlib import md5
from cloudpathlib import CloudPath
from google.cloud.storage.client import Client
import pytest

from buttermilk.bm import BM
from buttermilk.utils.save import upload_binary
from buttermilk.utils.utils import read_file

@pytest.fixture
def gcs() -> Client:
    from google.cloud import bigquery, storage
    return storage.Client()

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
    pass


def test_logger_initialised(bm):
    obj = bm.logger
    assert obj is not None
    assert len(obj.handlers) >= 2
