"""Unit tests for BatchExpansionProcessor."""

import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.agents.imagegen import VertexImagen3Fast, VertexImagen4Fast
from buttermilk.processors.batch_expansion import BatchExpansionProcessor


@pytest.mark.anyio
async def test_batch_expansion_multiplies_by_reps_and_models():
    """Test expansion creates reps × models records."""
<<<<<<< HEAD
    processor = BatchExpansionProcessor(repetitions=3, models=[VertexImagen3Fast, VertexImagen4Fast])
=======
    processor = BatchExpansionProcessor(
        repetitions=3, models=[VertexImagen3Fast, VertexImagen4Fast]
    )
>>>>>>> origin/stable

    record = BaseRecord(
        record_id="prompt-001",
        content="test prompt",
        metadata={"session_id": "sess123"},
    )

    results = [r async for r in processor.process(record, processor_stage="expand")]

    assert len(results) == 6  # 3 reps × 2 models
    assert all(r.content == "test prompt" for r in results)
    assert all("model_class" in r.metadata for r in results)
    assert all("repetition" in r.metadata for r in results)
    assert all("session_id" in r.metadata for r in results)

    # Verify unique record_ids
    record_ids = [r.record_id for r in results]
    assert len(set(record_ids)) == 6


@pytest.mark.anyio
async def test_batch_expansion_preserves_metadata():
    """Test that original metadata is preserved."""
    processor = BatchExpansionProcessor(repetitions=1, models=[VertexImagen3Fast])

    record = BaseRecord(
        record_id="test",
        content="prompt",
        metadata={"session_id": "s1", "scenario": "office", "custom": "value"},
    )

    results = [r async for r in processor.process(record, processor_stage="expand")]

    assert results[0].metadata["session_id"] == "s1"
    assert results[0].metadata["scenario"] == "office"
    assert results[0].metadata["custom"] == "value"


@pytest.mark.anyio
async def test_batch_expansion_adds_model_metadata():
    """Test that model-specific metadata is added."""
<<<<<<< HEAD
    processor = BatchExpansionProcessor(repetitions=2, models=[VertexImagen3Fast, VertexImagen4Fast])

    record = BaseRecord(record_id="test", content="prompt", metadata={"session_id": "s1"})
=======
    processor = BatchExpansionProcessor(
        repetitions=2, models=[VertexImagen3Fast, VertexImagen4Fast]
    )

    record = BaseRecord(
        record_id="test", content="prompt", metadata={"session_id": "s1"}
    )
>>>>>>> origin/stable

    results = [r async for r in processor.process(record, processor_stage="expand")]

    # Check that we have the right model classes (as class objects, not strings)
    model_classes = [r.metadata["model_class"] for r in results]
    assert model_classes.count(VertexImagen3Fast) == 2
    assert model_classes.count(VertexImagen4Fast) == 2

    # Check that model_prefix is set
    assert all("model_prefix" in r.metadata for r in results)
    assert any("imagen3fast" in r.metadata["model_prefix"] for r in results)
    assert any("imagen4fast" in r.metadata["model_prefix"] for r in results)


@pytest.mark.anyio
async def test_batch_expansion_repetition_indices():
    """Test that repetition indices are correct."""
    processor = BatchExpansionProcessor(repetitions=3, models=[VertexImagen3Fast])

<<<<<<< HEAD
    record = BaseRecord(record_id="test", content="prompt", metadata={"session_id": "s1"})
=======
    record = BaseRecord(
        record_id="test", content="prompt", metadata={"session_id": "s1"}
    )
>>>>>>> origin/stable

    results = [r async for r in processor.process(record, processor_stage="expand")]

    repetitions = [r.metadata["repetition"] for r in results]
    assert repetitions == [0, 1, 2]
