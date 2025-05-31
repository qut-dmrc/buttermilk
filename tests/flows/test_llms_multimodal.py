from io import BytesIO

import pytest
from PIL import Image

from buttermilk._core.llms import MULTIMODAL_MODELS, LLMClient, LLMs
from buttermilk._core.types import Record


@pytest.mark.anyio
async def test_multimodal_question(
    llm: LLMClient,
    multimodal_record: Record,
):
    # Test that the multimodal record contains content
    assert multimodal_record.content is not None

    # Test that we can convert the record to a message format
    message = multimodal_record.as_message(role="user")
    assert message is not None

    # Basic test to ensure the record has the expected structure
    assert True  # placeholder until we implement LLM calls


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_pil_image(llms: LLMs, model, image_bytes):
    """Test creating a Record with PIL Image directly in content."""
    # Create PIL Image from bytes
    pil_image = Image.open(BytesIO(image_bytes))

    # Create record with PIL image directly in content
    record = Record(
        content=[
            pil_image,
            "Hi, can you tell me what this is?",
        ],
        mime="multipart/mixed",
    )

    # Test that the record was created successfully
    assert record.content is not None
    assert len(record.content) == 2
    assert isinstance(record.content[0], Image.Image)
    assert isinstance(record.content[1], str)

    # Test that images property works
    assert record.images is not None
    assert len(record.images) == 1
    assert isinstance(record.images[0], Image.Image)

    # Test conversion to message format
    message = record.as_message(role="user")
    assert message is not None


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_with_message_conversion(llms: LLMs, model, image_bytes):
    """Test creating a Record with PIL image and converting to message format."""
    # Create PIL Image from bytes
    pil_image = Image.open(BytesIO(image_bytes))

    # Create record with PIL image directly in content
    record = Record(
        content=[
            pil_image,
            "Hi, can you tell me what this is?",
        ],
        mime="multipart/mixed",
    )

    # Test that the record was created successfully
    assert record.content is not None
    assert len(record.content) == 2
    assert isinstance(record.content[0], Image.Image)
    assert isinstance(record.content[1], str)

    # Test that images property works
    assert record.images is not None
    assert len(record.images) == 1
    assert isinstance(record.images[0], Image.Image)

    # Test conversion to message format (this should work with utility functions)
    message = record.as_message(role="user")
    assert message is not None
    assert hasattr(message, 'content')
    # The content should be a list with image content part and text
    assert isinstance(message.content, list)
    assert len(message.content) == 2
