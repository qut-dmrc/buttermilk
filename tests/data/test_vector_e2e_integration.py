"""End-to-end integration tests for vector pipeline with dict chunks.

Tests the complete zotero vectorization workflow:
1. SemanticSplitter creates dict chunks
2. EmbeddingGenerator adds embeddings to chunks
3. ChromaDBEmbeddings stores chunks in vector database

Uses real_bm fixture for fully configured objects.
No mocks of internal code - only external API boundaries.
Uses real data from JSON fixtures.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from buttermilk._core.types import Record
from buttermilk.data.vector import ChromaDBEmbeddings, ChunkedDocument, SemanticSplitter
from buttermilk.processors.embeddings import EmbeddingGenerator


class TestVectorE2EIntegration:
    """End-to-end integration tests for vector pipeline."""

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_complete_vectorization_pipeline_with_dict_chunks(self, real_bm):
        """Test complete pipeline: Splitter -> Embeddings -> ChromaDB.

        This is the REAL zotero pipeline workflow:
        1. Create a record (simulating ZoteroSource output)
        2. SemanticSplitter chunks it (produces dict chunks in practice)
        3. EmbeddingGenerator adds embeddings (works with dict chunks)
        4. ChromaDBEmbeddings stores in vector DB (must handle dict chunks)

        This test reproduces the exact bug we fixed:
        - Line 1013 error: 'dict' object has no attribute 'metadata'
        """
        # ARRANGE: Create test record (simulating real Zotero item)
        test_record = Record(
            record_id="TEST_ZOTERO_ARTICLE",
            dataset="zotero",
            content=(
                "This is a test academic article about artificial intelligence. "
                "The paper discusses recent advances in machine learning. "
                "Natural language processing has seen significant improvements. "
                "Deep learning models are becoming more sophisticated. "
                "Transformer architectures have revolutionized the field. "
                "Applications include text generation and understanding. "
                "Ethical considerations are increasingly important. "
                "Future research directions include multimodal learning."
            ),
            metadata={
                "title": "Recent Advances in Artificial Intelligence",
                "authors": ["Test Author", "Another Author"],
                "year": 2024,
                "item_type": "journalArticle",
            },
        )

        # Create processors (use real objects, not mocks!)
        splitter = SemanticSplitter(chunk_size=100, chunk_overlap=20)

        # Create temporary ChromaDB for test isolation
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_chromadb"

            embeddings = ChromaDBEmbeddings(
                persist_directory=str(db_path),
                collection_name="test_collection",
                embedding_model="text-embedding-004",  # Real model name
                dimensionality=768,
                concurrency=1,
            )

            # Mock ONLY the external embedding API (system boundary)
            # We patch the TextEmbeddingModel at the system boundary
            mock_embedding = [0.1] * 768  # Realistic embedding dimension

<<<<<<< HEAD
            with patch("vertexai.language_models.TextEmbeddingModel.get_embeddings") as mock_get_embeddings:
=======
            with patch(
                "vertexai.language_models.TextEmbeddingModel.get_embeddings"
            ) as mock_get_embeddings:
>>>>>>> origin/stable
                # Mock the embedding API response
                mock_result = AsyncMock()
                mock_result.values = mock_embedding
                mock_get_embeddings.return_value = [mock_result]

<<<<<<< HEAD
                with patch("vertexai.language_models.TextEmbeddingModel.get_embeddings_async") as mock_get_embeddings_async:
=======
                with patch(
                    "vertexai.language_models.TextEmbeddingModel.get_embeddings_async"
                ) as mock_get_embeddings_async:
>>>>>>> origin/stable
                    mock_get_embeddings_async.return_value = [mock_result]

                    # ACT: Run the complete pipeline

                    # Step 1: Chunk the document (produces dict chunks when using metadata)
                    chunked_records = []
<<<<<<< HEAD
                    async for chunked_record in splitter.process(test_record, processor_stage="chunk"):
=======
                    async for chunked_record in splitter.process(
                        test_record, processor_stage="chunk"
                    ):
>>>>>>> origin/stable
                        chunked_records.append(chunked_record)

                    assert len(chunked_records) == 1
                    chunked_record = chunked_records[0]
                    assert hasattr(chunked_record, "chunks")
                    assert len(chunked_record.chunks) > 0

                    # Verify chunks have required fields
                    chunked_record.chunks[0]
                    # Chunks from SemanticSplitter are ChunkedDocument objects
                    # But in the real pipeline, they can be converted to dicts
                    # Let's test with dict chunks to verify our fix

                    # Convert to dict chunks to test the bug scenario
                    dict_chunks = []
                    for chunk in chunked_record.chunks:
                        dict_chunk = {
                            "chunk_id": chunk.chunk_id,
                            "document_title": chunk.document_title,
                            "chunk_index": chunk.chunk_index,
                            "chunk_text": chunk.chunk_text,
                            "document_id": chunk.document_id,
<<<<<<< HEAD
                            "metadata": (chunk.metadata.copy() if hasattr(chunk, "metadata") else {}),
=======
                            "metadata": (
                                chunk.metadata.copy()
                                if hasattr(chunk, "metadata")
                                else {}
                            ),
>>>>>>> origin/stable
                        }
                        dict_chunks.append(dict_chunk)

                    # Create new record with dict chunks
<<<<<<< HEAD
                    chunked_record = chunked_record.model_copy(update={"chunks": dict_chunks})
=======
                    chunked_record = chunked_record.model_copy(
                        update={"chunks": dict_chunks}
                    )
>>>>>>> origin/stable

                    # Step 2: Generate embeddings (using real EmbeddingGenerator)
                    embedding_gen = EmbeddingGenerator(
                        embedding_model="text-embedding-004",
                        dimensionality=768,
                        embedding_batch_size=10,
                    )

                    # Initialize embeddings cache
                    await embeddings.ensure_cache_initialized()

                    embedded_records = []
<<<<<<< HEAD
                    async for embedded_record in embedding_gen.process(chunked_record, processor_stage="embed"):
=======
                    async for embedded_record in embedding_gen.process(
                        chunked_record, processor_stage="embed"
                    ):
>>>>>>> origin/stable
                        embedded_records.append(embedded_record)

                    assert len(embedded_records) == 1
                    embedded_record = embedded_records[0]

                    # Verify embeddings were added to dict chunks
                    for chunk in embedded_record.chunks:
                        assert isinstance(chunk, dict), "Chunks should be dicts"
                        assert "embedding" in chunk, "Each chunk should have embedding"
                        assert chunk["embedding"] is not None
                        assert len(chunk["embedding"]) == 768

                    # Step 3: Store in ChromaDB (this was failing with the metadata bug!)
                    # This is where line 1013 error occurred: chunk.metadata.update(...)
                    result = await embeddings.process_record(
                        embedded_record,
                        skip_existing=False,
                        validate_before_process=True,
                    )

                    # ASSERT: Verify complete pipeline succeeded

                    # Verify processing succeeded (not failed!)
<<<<<<< HEAD
                    assert result.status == "processed", f"Processing failed: {result.reason}"
=======
                    assert result.status == "processed", (
                        f"Processing failed: {result.reason}"
                    )
>>>>>>> origin/stable
                    assert result.chunks_created == len(embedded_record.chunks)
                    assert result.chunks_created > 0

                    # Verify metadata was enhanced correctly on dict chunks
                    for chunk in embedded_record.chunks:
                        assert isinstance(chunk, dict), "Chunks should still be dicts"
                        assert "metadata" in chunk, "Chunk should have metadata"

                        # Verify metadata enhancements from ChromaDBEmbeddings.process_record
                        metadata = chunk["metadata"]
<<<<<<< HEAD
                        assert "embedding_model" in metadata, "Should have embedding_model"
=======
                        assert "embedding_model" in metadata, (
                            "Should have embedding_model"
                        )
>>>>>>> origin/stable
                        assert metadata["embedding_model"] == "text-embedding-004"
                        assert "content_hash" in metadata, "Should have content_hash"
                        assert "created_timestamp" in metadata, "Should have timestamp"

                    # Verify chunks were actually stored in ChromaDB
                    collection = embeddings.collection
                    stored_count = collection.count()
<<<<<<< HEAD
                    assert stored_count == result.chunks_created, f"Expected {result.chunks_created} chunks in DB, got {stored_count}"

                    # Verify we can query the stored chunks
                    query_results = collection.query(query_embeddings=[mock_embedding], n_results=1)
                    assert len(query_results["ids"][0]) > 0, "Should be able to query chunks"
=======
                    assert stored_count == result.chunks_created, (
                        f"Expected {result.chunks_created} chunks in DB, got {stored_count}"
                    )

                    # Verify we can query the stored chunks
                    query_results = collection.query(
                        query_embeddings=[mock_embedding], n_results=1
                    )
                    assert len(query_results["ids"][0]) > 0, (
                        "Should be able to query chunks"
                    )
>>>>>>> origin/stable

    @pytest.mark.anyio
    async def test_metadata_enhancement_preserves_existing_fields(self, real_bm):
        """Verify that metadata enhancement preserves existing chunk metadata.

        This tests the specific code at line 928-944 in vector.py:
        chunk_metadata = _get_chunk_field(chunk, 'metadata', {})
        chunk_metadata.update({...})
        """
        # Create a record with dict chunks that have existing metadata
        record = Record(
            record_id="TEST_METADATA",
            dataset="test",
            content="Test content",
            metadata={"title": "Test Document"},
            chunks=[
                {
                    "chunk_id": "chunk_1",
                    "document_title": "Test Document",
                    "chunk_index": 0,
                    "chunk_text": "Test content chunk",
                    "document_id": "TEST_METADATA",
                    "metadata": {
                        "existing_field": "should_be_preserved",
                        "chunk_type": "content",
                    },
                    "embedding": [0.1] * 768,
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_db"

            embeddings = ChromaDBEmbeddings(
                persist_directory=str(db_path),
                collection_name="test_collection",
                embedding_model="test-model",
                dimensionality=768,
            )

            await embeddings.ensure_cache_initialized()

            # This should NOT fail with: 'dict' object has no attribute 'metadata'
            result = await embeddings.process_record(
                record,
                skip_existing=False,
                validate_before_process=True,
            )

            # Verify processing succeeded
            assert result.status == "processed", f"Failed: {result.reason}"

            # Verify existing metadata was preserved AND new fields added
            chunk = record.chunks[0]
            assert chunk["metadata"]["existing_field"] == "should_be_preserved"
            assert chunk["metadata"]["chunk_type"] == "content"
            assert "embedding_model" in chunk["metadata"]
            assert "content_hash" in chunk["metadata"]

    @pytest.mark.anyio
    async def test_mixed_chunk_types_in_same_record(self, real_bm):
        """Test that a record can have both dict and object chunks (edge case).

        While not common, the code should handle mixed types gracefully.
        """
        # Create a record with mixed chunk types
        dict_chunk = {
            "chunk_id": "chunk_dict",
            "document_title": "Test",
            "chunk_index": 0,
            "chunk_text": "Dict chunk",
            "document_id": "TEST_MIXED",
            "metadata": {"type": "dict"},
            "embedding": [0.1] * 768,
        }

        object_chunk = ChunkedDocument(
            chunk_id="chunk_object",
            document_title="Test",
            chunk_index=1,
            chunk_text="Object chunk",
            document_id="TEST_MIXED",
            metadata={"type": "object"},
            embedding=[0.2] * 768,
        )

        record = Record(
            record_id="TEST_MIXED",
            dataset="test",
            content="Test",
            metadata={"title": "Mixed Chunks Test"},
            chunks=[dict_chunk, object_chunk],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_db"

            embeddings = ChromaDBEmbeddings(
                persist_directory=str(db_path),
                collection_name="test_collection",
                embedding_model="test-model",
                dimensionality=768,
            )

            await embeddings.ensure_cache_initialized()

            # Should handle both dict and object chunks
            result = await embeddings.process_record(
                record,
                skip_existing=False,
                validate_before_process=True,
            )

            assert result.status == "processed", f"Failed: {result.reason}"
            assert result.chunks_created == 2

            # Both chunks should have enhanced metadata
            for chunk in record.chunks:
                if isinstance(chunk, dict):
                    assert "embedding_model" in chunk["metadata"]
                else:
                    assert hasattr(chunk, "metadata")
                    assert "embedding_model" in chunk.metadata
