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
    assert len(obj.handlers) >= 1


def test_save(bm):
    uri = bm.save(data=["test data"], extension=".txt")
    assert uri.startswith(bm.run_info.save_dir)
    assert uri.endswith(".txt")
    uploaded = AnyPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == '"test data"'
    uploaded.unlink(missing_ok=False)


def test_upload_text(bm):
    import pytest
    
    # Skip test if no clouds configured or no bucket available
    if not bm.clouds or not hasattr(bm.clouds[0], "bucket") or not bm.clouds[0].bucket:
        pytest.skip("No cloud bucket configured for test")
        
    uri = f"gs://{bm.clouds[0].bucket}/test_data/{uuid}.txt"
    return_uri = upload_text(data="test data", uri=uri)
    assert return_uri == uri
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == "test data"
    uploaded.unlink(missing_ok=False)


def test_save_binary(bm):
    import pytest
    
    # Skip test if no clouds configured or no bucket available
    if not bm.clouds or not hasattr(bm.clouds[0], "bucket") or not bm.clouds[0].bucket:
        pytest.skip("No cloud bucket configured for test")
        
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


def test_genai_sync_client(bm):
    # Ensure the GenAI client is set up correctly
    client = bm.genai
    assert client is not None
    assert hasattr(client, "models")
    assert hasattr(client.models, "list")

    # List models to verify the client works
    models = list(client.models.list())
    model_names = [m.name for m in models]
    assert "publishers/google/models/gemini-2.5-flash" in model_names
    assert "publishers/google/models/imagen-4.0-ultra-generate-001" in model_names


def test_genai_location(bm):
    # Ensure the GenAI client is set up correctly
    client = bm.genai
    

@pytest.mark.anyio
async def test_genai_async_client(bm):
    # Ensure the GenAI client is set up correctly
    client = bm.genai.aio

    # List models to verify the client works
    models = await client.models.list()
    model_names = [m.name for m in models]
    assert "publishers/google/models/gemini-2.5-flash" in model_names
    assert "publishers/google/models/imagen-4.0-ultra-generate-001" in model_names
