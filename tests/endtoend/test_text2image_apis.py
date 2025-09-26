# filepath: /src/buttermilk/tests/images/test_text2image_integrations.py
from hashlib import sha256
from io import BytesIO

import pytest
from cloudpathlib import CloudPath
from PIL import Image, ImageStat

from buttermilk._core.image import ImageRecord
from buttermilk.agents.imagegen import ImageClients

pytestmark = pytest.mark.anyio

CLIENTS = [x for x in ImageClients]

TEST_PROMPT = "Two Bangladeshi women working at a coffee shop in Dhaka, Bangladesh."
TEST_NEGATIVE_PROMPT = "TRADITIONAL ATTIRE"

ALLOWED_SCHEMES = ("gs://", "s3://", "az://")
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _is_nontrivial_image(img: Image.Image) -> bool:
    try:
        w, h = img.size
        if w < 16 or h < 16:
            return False
        # Use variance to detect non-uniform images
        stat = ImageStat.Stat(img.convert("L"))
        return stat.var[0] > 0.0
    except Exception:
        return False


@pytest.mark.parametrize("client", CLIENTS)
async def test_generated_image_is_valid_and_nontrivial(client):
    image_client = client()
    result = await image_client.generate_image(
        text=TEST_PROMPT,
        negative_prompt=TEST_NEGATIVE_PROMPT,
    )

    assert isinstance(result, ImageRecord)
    assert isinstance(result.image, Image.Image)
    assert _is_nontrivial_image(result.image)

    assert result.uri.startswith(ALLOWED_SCHEMES)
    cp = CloudPath(result.uri)
    assert cp.exists()
    assert cp.suffix.lower() in ALLOWED_SUFFIXES

    # Download and verify the stored artifact is a valid image with non-zero bytes
    with cp.open("rb") as f:
        blob = f.read()
    assert len(blob) > 1024  # basic sanity check the file isn't empty/tiny

    img2 = Image.open(BytesIO(blob))
    img2.load()  # force decode
    assert isinstance(img2, Image.Image)
    assert _is_nontrivial_image(img2)


@pytest.mark.parametrize("client", CLIENTS)
async def test_image_can_roundtrip_to_bytes_and_reopen(client):
    image_client = client()
    result = await image_client.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEGATIVE_PROMPT,
    )

    # Round-trip current in-memory image to bytes and back
    buf = BytesIO()
    # Default to PNG if format is missing
    fmt = result.image.format or "PNG"
    result.image.save(buf, format=fmt)
    data = buf.getvalue()
    assert len(data) > 1024

    reopened = Image.open(BytesIO(data))
    reopened.load()
    assert isinstance(reopened, Image.Image)
    assert _is_nontrivial_image(reopened)


@pytest.mark.parametrize("client", CLIENTS)
async def test_cloud_artifact_matches_in_memory_dimensions(client):
    image_client = client()
    result = await image_client.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEGATIVE_PROMPT,
    )

    w_mem, h_mem = result.image.size

    cp = CloudPath(result.uri)
    with cp.open("rb") as f:
        img2 = Image.open(f)
        img2.load()
    w_disk, h_disk = img2.size

    # Dimensions should be reasonable and typically match
    assert w_mem >= 16 and h_mem >= 16
    assert w_disk >= 16 and h_disk >= 16
    # Be lenient but expect equality in most pipelines
    assert (w_mem, h_mem) == (w_disk, h_disk)


@pytest.mark.parametrize("client", CLIENTS)
async def test_allows_none_negative_prompt_and_still_produces_image(client):
    image_client = client()
    result = await image_client.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=None,
    )
    assert isinstance(result, ImageRecord)
    assert isinstance(result.image, Image.Image)
    assert _is_nontrivial_image(result.image)
    assert result.uri.startswith(ALLOWED_SCHEMES)
    assert CloudPath(result.uri).exists()


@pytest.mark.parametrize("client", CLIENTS)
async def test_cloud_artifact_content_hash_is_stable_for_single_download(client):
    # Ensures the stored object is readable consistently (not necessarily deterministic generation)
    image_client = client()
    result = await image_client.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEGATIVE_PROMPT,
    )

    cp = CloudPath(result.uri)
    with cp.open("rb") as f:
        b1 = f.read()
    with cp.open("rb") as f:
        b2 = f.read()

    assert sha256(b1).hexdigest() == sha256(b2).hexdigest()
    assert len(b1) > 1024
