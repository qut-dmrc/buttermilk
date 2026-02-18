"""Integration tests for Google Cloud infrastructure functionality.

These tests verify that cloud upload/download, GenAI, and Vertex AI functionality
works correctly with real infrastructure.
"""

from hashlib import md5

import pytest
from cloudpathlib import CloudPath
from google.cloud import aiplatform
from shortuuid import uuid

from buttermilk.utils.save import upload_binary, upload_text
from buttermilk.utils.utils import read_file


@pytest.mark.slow
def test_save_binary(real_bm):
    """Test binary file upload to cloud storage."""
    # Integration test must fail if cloud manager not properly configured
    assert real_bm.cloud_manager is not None, "CloudManager must be configured for integration tests"
    assert real_bm.cloud_manager.clouds, "Cloud configurations must be available for integration tests"

    # Get the first cloud provider and ensure storage bucket exists
    cloud = real_bm.cloud_manager.clouds[0]
    assert hasattr(cloud, "storage_bucket"), "Cloud must have storage_bucket attribute"
    assert cloud.storage_bucket, "Storage bucket must be configured for integration tests"

    uri = f"gs://{cloud.storage_bucket}/test_data/{uuid}.txt"
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


def test_upload_text(real_bm):
    """Test text upload to cloud storage."""
    # Integration test must fail if cloud manager not properly configured
    assert real_bm.cloud_manager is not None, "CloudManager must be configured for integration tests"
    assert real_bm.cloud_manager.clouds, "Cloud configurations must be available for integration tests"

    # Get the first cloud provider and ensure storage bucket exists
    cloud = real_bm.cloud_manager.clouds[0]
    assert hasattr(cloud, "storage_bucket"), "Cloud must have storage_bucket attribute"
    assert cloud.storage_bucket, "Storage bucket must be configured for integration tests"

    uri = f"gs://{cloud.storage_bucket}/test_data/{uuid}.txt"
    return_uri = upload_text(data="test data", uri=uri)
    assert return_uri == uri
    uploaded = CloudPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    assert read_text == "test data"
    uploaded.unlink(missing_ok=False)


def test_vertex_setup():
    """Test Vertex AI setup and model listing."""
    # Try a simple operation like listing models
    models = aiplatform.Model.list()
    assert models is not None


def test_genai_sync_client(real_bm):
    """Test GenAI synchronous client functionality."""
    # Ensure the GenAI client is set up correctly
    client = real_bm.genai
    assert client is not None
    assert hasattr(client, "models")
    assert hasattr(client.models, "list")

    # List models to verify the client works
    models = list(client.models.list())
    model_names = [m.name for m in models]
    assert "publishers/google/models/gemini-2.5-flash" in model_names
    assert "publishers/google/models/imagen-4.0-ultra-generate-001" in model_names


def test_genai_location(real_bm):
    """Test GenAI client location configuration."""
    # Ensure the GenAI client is set up correctly
    client = real_bm.genai
    assert client is not None


@pytest.mark.anyio
async def test_genai_async_client(real_bm):
    """Test GenAI asynchronous client functionality."""
    # Ensure the GenAI client is set up correctly
    client = real_bm.genai.aio

    # List models to verify the client works
    models = await client.models.list()
    model_names = [m.name for m in models]
    assert "publishers/google/models/gemini-2.5-flash" in model_names
    assert "publishers/google/models/imagen-4.0-ultra-generate-001" in model_names
