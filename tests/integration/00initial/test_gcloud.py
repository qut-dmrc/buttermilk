from hashlib import md5

import pytest
from cloudpathlib import AnyPath, CloudPath
from google.cloud import aiplatform
from shortuuid import uuid

from buttermilk._core.log import logger
from buttermilk.utils.save import upload_binary, upload_text
from buttermilk.utils.utils import read_file


def test_logger_initialised(bm):
    obj = logger
    assert obj is not None
    # The logger is a structlog logger, not a standard logger, so it doesn't have handlers
    # Just verify it's properly initialized and can log
    assert hasattr(obj, 'info')
    assert hasattr(obj, 'error')
    assert hasattr(obj, 'debug')


def test_save(bm):
    uri = bm.save(data=["test data"], extension=".txt")
    assert uri.startswith(bm.session_info.save_dir)
    assert uri.endswith(".txt")
    uploaded = AnyPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    # The data is saved as a JSON-serialized list, not a single string
    assert read_text == '["test data"]'
    uploaded.unlink(missing_ok=False)


def test_upload_text():
    """Test moved to tests/integration/test_gcloud_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_gcloud_infrastructure.py")


def test_save_binary():
    """Test moved to tests/integration/test_gcloud_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_gcloud_infrastructure.py")


def test_vertex_setup():
    """Test moved to tests/integration/test_gcloud_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_gcloud_infrastructure.py")


def test_genai_sync_client():
    """Test moved to tests/integration/test_gcloud_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_gcloud_infrastructure.py")


def test_genai_location():
    """Test moved to tests/integration/test_gcloud_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_gcloud_infrastructure.py")
    

@pytest.mark.anyio
async def test_genai_async_client():
    """Test moved to tests/integration/test_gcloud_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_gcloud_infrastructure.py")
