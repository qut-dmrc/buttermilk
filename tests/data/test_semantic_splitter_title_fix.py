"""Test that SemanticSplitter correctly handles records with and without title attributes.

This test verifies the fix for the bug where SemanticSplitter unconditionally accessed
doc.title, but now uses getattr with a fallback to record_id.
"""

import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.data.vector import SemanticSplitter


@pytest.mark.anyio
async def test_semantic_splitter_with_record_without_title():
    """Test SemanticSplitter handles records without title attribute gracefully.

    After the fix, this should use record_id as a fallback when title is None.
    """
    # Create a BaseRecord which has title property but returns None
    record = BaseRecord(
        record_id="test_record_1",
        dataset="test_dataset",
        content="This is test content that should be chunked. " * 50,  # Make it long enough to chunk
    )

    # Create SemanticSplitter instance
    splitter = SemanticSplitter(breakpoint_threshold_amount=0.5, max_chunk_size=100, min_chunk_size=10)

    # Process the record - returns record with chunks attached
    processed_records = []
    async for processed_record in splitter.process(record):
        processed_records.append(processed_record)

    # Should have one processed record
    assert len(processed_records) == 1
    processed = processed_records[0]

    # Verify we got chunks
    assert hasattr(processed, "chunks")
    assert len(processed.chunks) > 0

    # Verify the chunks have document_title set to record_id (fallback)
    for chunk in processed.chunks:
        assert hasattr(chunk, "document_title")
        assert chunk.document_title == "test_record_1"  # Should use record_id as fallback


@pytest.mark.anyio
async def test_semantic_splitter_with_title_in_metadata():
    """Test SemanticSplitter uses title from metadata when available."""
    # Create a BaseRecord with title in metadata
    record = BaseRecord(
        record_id="test_record_2", dataset="test_dataset", content="This is test content with a title. " * 50, metadata={"title": "My Document Title"}
    )

    splitter = SemanticSplitter(breakpoint_threshold_amount=0.5, max_chunk_size=100, min_chunk_size=10)

    # Process the record
    processed_records = []
    async for processed_record in splitter.process(record):
        processed_records.append(processed_record)

    assert len(processed_records) == 1
    processed = processed_records[0]

    # Verify we got chunks with the correct title
    assert len(processed.chunks) > 0
    for chunk in processed.chunks:
        assert chunk.document_title == "My Document Title"


@pytest.mark.anyio
async def test_semantic_splitter_mixed_records():
    """Test SemanticSplitter handles mix of records with and without titles."""
    records = [
        # Record with title in metadata
        BaseRecord(record_id="with_title", dataset="test", content="Content with title. " * 30, metadata={"title": "Document with Title"}),
        # Record without title (will use record_id)
        BaseRecord(record_id="without_title", dataset="test", content="Content without title. " * 30),
    ]

    splitter = SemanticSplitter(breakpoint_threshold_amount=0.5, max_chunk_size=100, min_chunk_size=10)

    all_processed = []
    for record in records:
        async for processed_record in splitter.process(record):
            all_processed.append(processed_record)

    # Should have processed both records
    assert len(all_processed) == 2

    # Check chunks from first record use the title
    first_record = all_processed[0]
    assert all(c.document_title == "Document with Title" for c in first_record.chunks)

    # Check chunks from second record use record_id as fallback
    second_record = all_processed[1]
    assert all(c.document_title == "without_title" for c in second_record.chunks)
