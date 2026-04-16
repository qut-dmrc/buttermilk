"""Unit tests for ImageGenerationProcessor."""

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.agents.imagegen import VertexImagen3Fast
from buttermilk.processors.image_generation import ImageGenerationProcessor


def test_image_generation_processor_instantiation():
    """Test that ImageGenerationProcessor can be instantiated with a client class."""
    processor = ImageGenerationProcessor(client_class=VertexImagen3Fast)

    assert processor.client_class == VertexImagen3Fast


@pytest.mark.anyio
async def test_image_generation_processor_with_empty_content():
    """Test that processor fails-fast with empty content."""
    processor = ImageGenerationProcessor(client_class=VertexImagen3Fast)

    record = BaseRecord(
        record_id="test-003",
        content="",
        metadata={"session_id": "sess789"},
    )

    # Should fail-fast with empty prompt
    with pytest.raises(ValueError, match="empty content"):
        async for _ in processor.process(ProcessingContext(session_id="generate", record=record)):
            pass


@pytest.mark.anyio
async def test_image_generation_processor_with_none_content():
    """Test that processor fails-fast with None content."""
    processor = ImageGenerationProcessor(client_class=VertexImagen3Fast)

    record = BaseRecord(
        record_id="test-004",
        content=None,
        metadata={"session_id": "sess789"},
    )

    # Should fail-fast with None prompt
    with pytest.raises(ValueError, match="empty content"):
        async for _ in processor.process(ProcessingContext(session_id="generate", record=record)):
            pass
