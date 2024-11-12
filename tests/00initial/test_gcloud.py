from google.cloud.storage.client import Client
import pytest

from buttermilk.bm import BM
from buttermilk.utils.save import upload_binary
from buttermilk.utils.utils import read_file

@pytest.fixture
def gcs() -> Client:
    from google.cloud import bigquery, storage
    return storage.Client()

# Presumably this fails where default credentials have not yet been saved.
def test_save_binary_no_bm_init(gcs):
    try:
        with open("datatools/tests/data/sample_image.png", "rb") as img:
            uri = upload_binary(img)
            uploaded_bytes = img.seek(0).to_bytes()
            assert uri is not None

        downloaded = read_file(uri)
        assert downloaded == img
    finally:
        # delete
        gcs._delete_resource(uri)
    pass

def test_save_binary(bm: BM, gcs):
    assert bm
    test_save_binary_no_bm_init(gcs=gcs)


def test_logger_initialised(bm):
    obj = bm.logger
    assert obj is not None
    assert len(obj.handlers) >= 2
