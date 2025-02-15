import pytest
from PIL import Image

from buttermilk._core.runner_types import MediaObj, RecordInfo
from buttermilk.llms import LLMCapabilities
from buttermilk.utils.media import download_and_convert

pytestmark = pytest.mark.anyio


@pytest.mark.anyio
async def test_record_remote_load(multimodal_record):
    assert multimodal_record


def test_record_no_keywords():
    record = RecordInfo(text="test")
    assert not record.uri
    assert len(record.components) == 1
    assert record.components[0].mime == "text/plain"
    assert record.components[0].base_64 is None
    assert record.all_text == "test"
    assert len(record.record_id) >= 8


@pytest.mark.anyio
async def test_from_path_valid():
    # Use a small test image
    test_image_path = "tests/data/sadrobot.jpg"

    record = await download_and_convert(
        test_image_path,
        mime="image/jpeg",
        title="Test Image",
    )
    assert record.uri == test_image_path
    assert record.metadata["title"] == "Test Image"
    assert len(record.components) == 1
    assert record.components[0].mime == "image/jpeg"
    assert record.components[0].base_64 is not None


@pytest.mark.anyio
async def test_from_uri_valid():
    # Use a small test image
    test_image_path = "https://picsum.photos/64"

    record = await download_and_convert(
        test_image_path,
        mime="image/jpeg",
        title="Test Image",
    )
    assert record.uri == test_image_path
    assert record.metadata["title"] == "Test Image"
    assert len(record.components) == 1
    assert record.components[0].mime == "image/jpeg"
    assert record.components[0].base_64 is not None


ARTICLES = [
    (
        "abc-gaza",
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
        """He said it was a "relief" to hear the news of the ceasefire "which we were calling for, for the last 15 months".""",
    ),
    (
        "guardian-somethingmiriam",
        "https://www.theguardian.com/tv-and-radio/2024/apr/25/she-was-tough-but-it-broke-her-why-theres-something-about-miriam-was-reality-tvs-most-shameful-low",
        """it wasn’t a joke. It was Miriam and her life.""",
    ),
]


@pytest.mark.parametrize("id, url, test_str", ARTICLES)
async def test_from_uri_article(id, url, test_str):
    record = await download_and_convert(
        url,
    )
    assert test_str in record.all_text


@pytest.mark.anyio
async def test_from_uri_invalid():
    record = await download_and_convert("invalid_uri", mime="image/png")
    assert not record.uri
    assert record.all_text == "invalid_uri"


@pytest.mark.anyio
async def test_from_object_valid():
    # Create a dummy image object
    image = Image.new("RGB", (100, 100))
    obj = image.tobytes()

    record = await download_and_convert(
        obj,
        mime="image/png",
        title="Test Image from Object",
    )

    assert record.metadata["title"] == "Test Image from Object"
    assert len(record.components) == 1
    assert record.components[0].mime == "image/png"
    assert record.components[0].base_64 is not None


@pytest.mark.anyio
async def test_record_update():
    image = Image.new("RGB", (100, 100))
    record = await download_and_convert(
        image.tobytes(),
        mime="image/png",
        title="Test Image",
    )
    result = dict(
        category="test",
        prediction=True,
        result=0.9,
        labels=["test_label"],
        reasons=["test_reason"],
    )
    record.update_from(result)
    assert record.category == "test"
    assert record.prediction == True
    assert record.result == 0.9
    assert record.labels == ["test_label"]
    assert record.reasons == ["test_reason"]

    record.update_from(result.model_dump(), fields=["result"])
    assert record.result == 0.9
    assert "category" not in record.model_dump()


def test_as_openai_message_with_media(image_bytes: bytes):
    message = RecordInfo(data=[MediaObj(mime="image/png", content=image_bytes), "test"])
    openai_message = message.as_openai_message(
        model_capabilities=LLMCapabilities(image=True),
    )
    assert openai_message["role"] == "user"
    assert len(openai_message["content"]) == 2
    assert openai_message["content"][0]["type"] == "image_url"
    assert openai_message["content"][1]["type"] == "text"


def test_as_openai_message_with_media_and_role(image_bytes: bytes):
    message = RecordInfo(data=[MediaObj(mime="image/png", content=image_bytes)])
    openai_message = message.as_openai_message(
        role="system",
        model_capabilities=LLMCapabilities(image=True),
    )
    assert openai_message["role"] == "system"
    assert openai_message["content"][0]["type"] == "image_url"
    assert len(openai_message["content"]) == 1


def test_as_openai_message_with_text():
    message = RecordInfo(text="test")
    openai_message = message.as_openai_message(
        role="system",
        model_capabilities=LLMCapabilities(),
    )
    assert openai_message["role"] == "system"
    assert openai_message["content"] == [{"type": "text", "text": "test"}]
    assert len(openai_message["content"]) == 1


def test_as_openai_message_no_media_no_text():
    with pytest.raises(OSError):
        message = RecordInfo()
    openai_message = message.as_openai_message(model_capabilities=LLMCapabilities())
    assert len(openai_message["content"]) == 0
