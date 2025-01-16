import pytest
from PIL import Image

from buttermilk._core.runner_types import MediaObj, RecordInfo, Result


@pytest.mark.asyncio
async def test_from_uri_valid():
    # Use a small test image
    test_image_path = "tests/data/sadrobot.jpg"

    record = await RecordInfo.from_uri(
        test_image_path,
        mimetype="image/jpeg",
        title="Test Image",
    )
    assert record.uri == test_image_path
    assert record.metadata["title"] == "Test Image"
    assert len(record.components) == 1
    assert record.components[0].mime == "image/jpg"
    assert record.components[0].base_64 is not None


@pytest.mark.asyncio
async def test_from_uri_invalid():
    record = await RecordInfo.from_uri(
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
    )
    assert (
        """He said it was a "relief" to hear the news of the ceasefire "which we were calling for, for the last 15 months"."""
        in record.text
    )
    assert (
        """Nasser Mashni, the president of the Australian Palestine Advocacy Network, says he felt changed as an Australian."""
        in record.text
    )


@pytest.mark.asyncio
async def test_from_uri_article():
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
    assert len(record.components) == 1
    assert record.components[0].mime == "image/png"
    assert record.components[0].base_64 is not None


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


def test_as_openai_message_with_media():
    message = RecordInfo(components=[MediaObj()], text="test")
    openai_message = message.as_openai_message()
    assert openai_message["role"] == "user"
    assert openai_message["content"] == [
        {"type": "media", "media": "test"},
        {"type": "text", "text": "test"},
    ]


def test_as_openai_message_with_media_and_role(image_bytes: bytes):
    message = RecordInfo(components=[MediaObj(mime="image/png", data=image_bytes)])
    openai_message = message.as_openai_message(role="system")
    assert openai_message["role"] == "system"
    assert openai_message["content"][0]["type"] == "image/png"
    assert openai_message["content"][0]["image_url"]


def test_as_openai_message_with_text():
    message = RecordInfo(text="test")
    openai_message = message.as_openai_message(role="system")
    assert openai_message["role"] == "system"
    assert openai_message["content"] == [{"type": "text", "text": "test"}]


def test_as_openai_message_no_media_no_text():
    message = RecordInfo()
    with pytest.raises(OSError):
        message.as_openai_message()
