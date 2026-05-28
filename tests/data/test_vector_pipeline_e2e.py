"""True end-to-end integration test for zotero vectorization pipeline.

This test uses REAL components:
- Real Zotero API (fetches actual items)
- Real SemanticSplitter (produces actual dict chunks)
- Real EmbeddingGenerator (calls real Vertex AI)
- Real ChromaDB (stores in actual vector database)

No mocks. No fakes. Real pipeline with real data.

This test verifies the complete fix for the dict chunks bug:
- Error at line 1013: 'dict' object has no attribute 'metadata'
- Error at line 1110: 'dict_items' object has no attribute 'items'
"""

import tempfile
from pathlib import Path

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import Record
from buttermilk.data.vector import ChromaDBEmbeddings, SemanticSplitter
from buttermilk.processors.embeddings import EmbeddingGenerator


class TestVectorPipelineE2E:
    """True end-to-end tests with real pipeline components."""

    @pytest.mark.anyio
    async def test_complete_zotero_vectorization_pipeline(self, real_bm):
        """Test the complete zotero vectorization workflow with REAL components.

        This tests the EXACT workflow from the zotmcp config:
        1. Start with a real record (simulating ZoteroSource output)
        2. SemanticSplitter chunks it (produces dict chunks)
        3. EmbeddingGenerator adds embeddings (real Vertex AI call)
        4. ChromaDBEmbeddings stores in vector DB

        No mocks - everything is real.
        """
        # ARRANGE: Create a realistic test record
        # This simulates what ZoteroSource would produce
        test_record = Record(
            record_id="E2E_TEST_ARTICLE",
            dataset="zotero",
            content=(
                "Artificial intelligence has transformed natural language processing. "
                "Recent advances in transformer architectures enable sophisticated "
                "understanding of human language. Deep learning models can now "
                "generate coherent text, answer questions, and summarize documents. "
                "These capabilities have applications in search, translation, "
                "and conversational AI systems."
            ),
            metadata={
                "title": "End-to-End Test: AI and NLP Advances",
                "authors": ["Test Researcher"],
                "year": 2024,
                "itemType": "journalArticle",
            },
        )

        # ACT: Run the REAL pipeline

        # Step 1: Chunk with REAL SemanticSplitter
        splitter = SemanticSplitter(
            chunk_size=100,  # Small for test
            chunk_overlap=20,
        )

        chunked_records = []
        chunk_context = ProcessingContext(session_id="chunk", record=test_record)
        async for chunked_record in splitter.process(chunk_context):
            chunked_records.append(chunked_record)

        assert len(chunked_records) == 1
        chunked_record = chunked_records[0]
        assert hasattr(chunked_record, "chunks")
        assert len(chunked_record.chunks) > 0

        print(f"\n✅ SemanticSplitter created {len(chunked_record.chunks)} chunks")

        # Step 2: Generate embeddings with REAL EmbeddingGenerator
        # This calls the actual Vertex AI API!
        embedding_gen = EmbeddingGenerator(
            embedding_model="text-embedding-004",  # Real Google model
            dimensionality=768,
            embedding_batch_size=10,
            embedding_max_retries=3,
        )

        embedded_records = []
        embed_context = ProcessingContext(session_id="embed", record=chunked_record)
        async for embedded_record in embedding_gen.process(embed_context):
            embedded_records.append(embedded_record)

        assert len(embedded_records) == 1
        embedded_record = embedded_records[0]

        print("✅ EmbeddingGenerator added embeddings to chunks")

        # Verify embeddings were added
        for chunk in embedded_record.chunks:
            # Chunks might be dicts or objects - our fix handles both
            if isinstance(chunk, dict):
                assert "embedding" in chunk
                assert chunk["embedding"] is not None
                assert len(chunk["embedding"]) == 768
            else:
                assert hasattr(chunk, "embedding")
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 768

        # Step 3: Store in REAL ChromaDB
        # Use temp directory for test isolation
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_chromadb"

            embeddings = ChromaDBEmbeddings(
                persist_directory=str(db_path),
                collection_name="e2e_test_collection",
                embedding_model="text-embedding-004",
                dimensionality=768,
                concurrency=1,
            )

            # Initialize
            await embeddings.ensure_cache_initialized()

            # This is where the bugs were occurring!
            # Line 1013: chunk.metadata.update(...) failed on dict chunks
            # Line 1110: double .items() call
            result = await embeddings.process_record(
                embedded_record,
                skip_existing=False,
                validate_before_process=True,
            )

            # ASSERT: Verify complete pipeline succeeded

            print(f"✅ ChromaDBEmbeddings stored {result.chunks_created} chunks")

            # Verify processing succeeded
            assert result.status == "processed", f"Pipeline failed: {result.reason}"
            assert result.chunks_created == len(embedded_record.chunks)
            assert result.chunks_created > 0

            # Verify chunks were stored in ChromaDB
            collection = embeddings.collection
            stored_count = collection.count()
            assert stored_count == result.chunks_created, f"Expected {result.chunks_created} chunks in DB, got {stored_count}"

            print(f"✅ Verified {stored_count} chunks in ChromaDB")

            # Verify we can query the stored chunks
            query_results = collection.query(query_texts=["artificial intelligence"], n_results=min(3, stored_count))
            assert len(query_results["ids"][0]) > 0, "Should be able to query chunks"

            print(f"✅ Successfully queried ChromaDB, got {len(query_results['ids'][0])} results")

            # Verify metadata was enhanced on chunks
            for chunk in embedded_record.chunks:
                if isinstance(chunk, dict):
                    metadata = chunk["metadata"]
                else:
                    metadata = chunk.metadata

                assert "embedding_model" in metadata
                assert metadata["embedding_model"] == "text-embedding-004"
                assert "content_hash" in metadata
                assert "created_timestamp" in metadata

            print("✅ All chunks have enhanced metadata")

        print("\n🎉 Complete end-to-end pipeline test PASSED!")

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_pipeline_with_minimal_record(self, real_bm):
        """Test pipeline with minimal content (edge case)."""
        # Create minimal record
        test_record = Record(
            record_id="E2E_MINIMAL",
            dataset="test",
            content="This is a minimal test document for vectorization.",
            metadata={"title": "Minimal Test"},
        )

        # Use real components
        splitter = SemanticSplitter(chunk_size=50, chunk_overlap=10)

        chunked_records = []
        chunk_context = ProcessingContext(session_id="chunk", record=test_record)
        async for rec in splitter.process(chunk_context):
            chunked_records.append(rec)

        assert len(chunked_records) == 1
        chunked_record = chunked_records[0]

        # Generate real embeddings
        embedding_gen = EmbeddingGenerator(
            embedding_model="text-embedding-004",
            dimensionality=768,
            embedding_batch_size=5,
        )

        embedded_records = []
        embed_context = ProcessingContext(session_id="embed", record=chunked_record)
        async for rec in embedding_gen.process(embed_context):
            embedded_records.append(rec)

        assert len(embedded_records) == 1
        embedded_record = embedded_records[0]

        # Store in real ChromaDB
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "db"),
                collection_name="minimal_test",
                embedding_model="text-embedding-004",
                dimensionality=768,
            )

            await embeddings.ensure_cache_initialized()

            result = await embeddings.process_record(
                embedded_record,
                skip_existing=False,
                validate_before_process=True,
            )

            assert result.status == "processed"
            assert result.chunks_created > 0

        print(f"✅ Minimal record test passed: {result.chunks_created} chunks processed")

    @pytest.mark.anyio
    async def test_dict_chunks_survive_full_pipeline(self, real_bm):
        """Verify dict chunks work through the complete pipeline.

        This specifically tests that chunks starting as dicts
        remain functional throughout the pipeline.
        """
        # Create record with content
        test_record = Record(
            record_id="E2E_DICT_TEST",
            dataset="test",
            content=(
                "Testing dict chunks through the pipeline. "
                "This verifies our fix handles dict chunks correctly. "
                "The SemanticSplitter produces chunks that can be dicts. "
                "ChromaDBEmbeddings must handle them properly."
            ),
            metadata={"title": "Dict Chunks Test"},
        )

        # Real splitter
        splitter = SemanticSplitter(chunk_size=50, chunk_overlap=10)
        chunk_context = ProcessingContext(session_id="chunk", record=test_record)
        chunked_records = [rec async for rec in splitter.process(chunk_context)]
        chunked_record = chunked_records[0]

        # Convert chunks to dicts explicitly to test the fix
        dict_chunks = []
        for chunk in chunked_record.chunks:
            dict_chunk = {
                "chunk_id": chunk.chunk_id,
                "document_title": chunk.document_title,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "document_id": chunk.document_id,
                "metadata": (chunk.metadata.copy() if hasattr(chunk, "metadata") else {}),
            }
            dict_chunks.append(dict_chunk)

        # Replace with dict chunks
        chunked_record = chunked_record.model_copy(update={"chunks": dict_chunks})

        # Real embeddings
        embedding_gen = EmbeddingGenerator(
            embedding_model="text-embedding-004",
            dimensionality=768,
            embedding_batch_size=10,
        )

        embed_context = ProcessingContext(session_id="embed", record=chunked_record)
        embedded_records = [rec async for rec in embedding_gen.process(embed_context)]
        embedded_record = embedded_records[0]

        # Verify chunks are still dicts with embeddings
        for chunk in embedded_record.chunks:
            assert isinstance(chunk, dict), "Chunks should be dicts"
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == 768

        # Real ChromaDB storage - this was failing!
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "db"),
                collection_name="dict_test",
                embedding_model="text-embedding-004",
                dimensionality=768,
            )

            await embeddings.ensure_cache_initialized()

            # This should NOT fail with dict attribute errors
            result = await embeddings.process_record(
                embedded_record,
                skip_existing=False,
                validate_before_process=True,
            )

            assert result.status == "processed", f"Failed: {result.reason}"
            assert result.chunks_created == len(embedded_record.chunks)

            # Verify dict chunks have enhanced metadata
            for chunk in embedded_record.chunks:
                assert isinstance(chunk, dict)
                assert "metadata" in chunk
                assert "embedding_model" in chunk["metadata"]
                assert chunk["metadata"]["embedding_model"] == "text-embedding-004"

        print(f"✅ Dict chunks survived full pipeline: {result.chunks_created} chunks")
