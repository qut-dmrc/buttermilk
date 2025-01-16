import pytest
from PIL import Image

from buttermilk._core.runner_types import MediaObj, RecordInfo, Result
from buttermilk.llms import LLMCapabilities

pytestmark = pytest.mark.anyio


def test_record_no_keywords():
    record = RecordInfo("test")
    assert not record.uri
    assert len(record._components) == 1
    assert record._components[0].mime == "text/plain"
    assert record._components[0].base_64 is None
    assert record.text == "test"
    assert len(record.record_id) >= 8


@pytest.mark.asyncio
async def test_from_path_valid():
    # Use a small test image
    test_image_path = "tests/data/sadrobot.jpg"

    record = await RecordInfo.from_path(
        test_image_path,
        mimetype="image/jpeg",
        title="Test Image",
    )
    assert record.uri == test_image_path
    assert record.metadata["title"] == "Test Image"
    assert len(record._components) == 1
    assert record._components[0].mime == "image/jpeg"
    assert record._components[0].base_64 is not None


@pytest.mark.asyncio
async def test_from_uri_valid():
    # Use a small test image
    test_image_path = "https://picsum.photos/64"

    record = await RecordInfo.from_uri(
        test_image_path,
        mimetype="image/jpeg",
        title="Test Image",
    )
    assert record.uri == test_image_path
    assert record.metadata["title"] == "Test Image"
    assert len(record._components) == 1
    assert record._components[0].mime == "image/jpeg"
    assert record._components[0].base_64 is not None


@pytest.mark.asyncio
async def test_from_uri_article():
    record = await RecordInfo.from_uri(
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
    )
    assert (
        """He said it was a "relief" to hear the news of the ceasefire "which we were calling for, for the last 15 months"."""
        in record.all_text
    )
    assert (
        """Nasser Mashni, the president of the Australian Palestine Advocacy Network, says he felt changed as an Australian."""
        in record.all_text
    )


@pytest.mark.asyncio
@pytest.mark.xfail
async def test_from_uri_invalid():
    await RecordInfo.from_uri("invalid_uri", mimetype="image/png")


@pytest.mark.asyncio
async def test_from_object_valid():
    # Create a dummy image object
    image = Image.new("RGB", (100, 100))
    obj = image.tobytes()

    record = await RecordInfo.from_object(
        obj,
        mimetype="image/png",
        title="Test Image from Object",
    )

    assert record.metadata["title"] == "Test Image from Object"
    assert len(record._components) == 1
    assert record._components[0].mime == "image/png"
    assert record._components[0].base_64 is not None


@pytest.mark.asyncio
async def test_record_update():
    image = Image.new("RGB", (100, 100))
    record = await RecordInfo.from_object(
        image.tobytes(),
        mimetype="image/png",
        title="Test Image",
    )
    result = Result(
        category="test",
        prediction=True,
        result=0.9,
        labels=["test_label"],
        reasons=["test_reason"],
    )
    record.update_from(result=result)
    assert record.category == "test"
    assert record.prediction == True
    assert record.result == 0.9
    assert record.labels == ["test_label"]
    assert record.reasons == ["test_reason"]

    record.update_from(result=result, fields=["result"])
    assert record.result == 0.9
    assert "category" not in record.model_dump()


def test_as_openai_message_with_media(image_bytes: bytes):
    message = RecordInfo(data=[MediaObj(mime="image/png", data=image_bytes), "test"])
    openai_message = message.as_openai_message(
        model_capabilities=LLMCapabilities(image=True),
    )
    assert openai_message["role"] == "user"
    assert len(openai_message["content"]) == 2
    assert openai_message["content"][0]["type"] == "image_url"
    assert openai_message["content"][1]["type"] == "text"


def test_as_openai_message_with_media_and_role(image_bytes: bytes):
    message = RecordInfo(data=[MediaObj(mime="image/png", data=image_bytes)])
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


def test_as_openai_message_no_media_no_text():
    message = RecordInfo()
    with pytest.raises(OSError):
        message.as_openai_message(model_capabilities=LLMCapabilities())
