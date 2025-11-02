"""Tests for ChromaDB batch size handling in vector operations.

This test suite verifies that ChromaDBEmbeddings correctly handles large numbers
of chunks by batching upsert operations to respect ChromaDB's maximum batch size of 5,461.

Uses real ChromaDBEmbeddings with temporary storage (no mocks of internal code).
Mocks only external embedding API (system boundary).
"""

import tempfile
from pathlib import Path

import pytest

from buttermilk._core.types import Record
from buttermilk.data.vector import ChromaDBEmbeddings, ChunkedDocument


class TestChromaDBBatching:
    """Test ChromaDB batching for large upsert operations."""

    @pytest.mark.anyio
    async def test_store_chunks_handles_large_batch(self):
        """Test that _store_chunks_for_record handles >5,461 chunks via batching.

        ChromaDB has a maximum batch size of 5,461 items. This test verifies that
        when upserting 6,000 chunks, the operation succeeds by batching appropriately.

        Before fix: ValueError: Batch size of 6000 is greater than max batch size of 5461
        After fix: Should succeed by splitting into batches
        """
        # ARRANGE: Create ChromaDBEmbeddings with temporary storage
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_chromadb"

            embeddings = ChromaDBEmbeddings(
                persist_directory=str(db_path),
                collection_name="test_batching",
                embedding_model="text-embedding-004",
                dimensionality=768,
                concurrency=1,
                read_only=False,
            )

            # Initialize collection
            await embeddings.ensure_cache_initialized()

            # ARRANGE: Create Record with 6,000 chunks (exceeds 5,461 limit)
            num_chunks = 6000
            chunks = []

            for i in range(num_chunks):
                chunk = ChunkedDocument(
                    chunk_id=f"TEST_CHUNK_{i}",
                    chunk_text=f"This is test chunk number {i} with some content.",
                    embedding=[0.1] * 768,  # Realistic embedding dimension
                    document_title="Large Test Document",
                    chunk_index=i,
                    document_id="LARGE_TEST_DOC",
                    metadata={"chunk_type": "content", "test": True},
                )
                chunks.append(chunk)

            test_record = Record(
                record_id="LARGE_TEST_RECORD",
                dataset="test",
                content="Large document with many chunks",
                metadata={"title": "Test Large Document"},
                chunks=chunks,
            )

            # ACT: Attempt to store all chunks
            # Before fix: This will raise ValueError about batch size
            # After fix: This should succeed by batching
            await embeddings._store_chunks_for_record(test_record)

            # ASSERT: Verify all chunks were stored successfully
            # Query collection to verify chunk count
            collection = embeddings.collection
            result = collection.get()

            assert len(result["ids"]) == num_chunks, f"Expected {num_chunks} chunks to be stored, but found {len(result['ids'])}"

            # Verify some chunks have correct metadata
            assert "TEST_CHUNK_0" in result["ids"]
            assert "TEST_CHUNK_5999" in result["ids"]
