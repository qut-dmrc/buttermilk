
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from buttermilk.data.vector import ChromaDBEmbeddings

@pytest.mark.anyio
async def test_chromadb_readonly_query_dimensionality_fix():
    """Verify that read_only=True still allows queries with correct dimensionality.
    
    This test ensures that the embedding function is correctly initialized and
    attached to the collection even in read-only mode, preventing ChromaDB
    from falling back to the default 384-dim embedder.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chromadb_readonly"
        
        # 1. Setup: Create a 768-dim collection in write mode
        # We must use a dimensionality different from ChromaDB default (384)
        target_dim = 768
        collection_name = "readonly_fix_test"
        
        embeddings_write = ChromaDBEmbeddings(
            persist_directory=str(db_path),
            collection_name=collection_name,
            embedding_model="text-embedding-004",
            dimensionality=target_dim,
            read_only=False
        )
        await embeddings_write.ensure_cache_initialized()
        
        # Mock embedding model to allow adding a record
        mock_embedding = [0.1] * target_dim
        with patch("vertexai.language_models.TextEmbeddingModel.get_embeddings") as mock_get_embeddings:
            mock_result = AsyncMock()
            mock_result.values = mock_embedding
            mock_get_embeddings.return_value = [mock_result]
            
            with patch("vertexai.language_models.TextEmbeddingModel.get_embeddings_async") as mock_get_embeddings_async:
                mock_get_embeddings_async.return_value = [mock_result]
                
                # Add a record to fix the collection dimensionality
                embeddings_write.collection.add(
                    ids=["doc1"],
                    embeddings=[mock_embedding],
                    documents=["This is a test document"]
                )
        
        # 2. Test: Access in read_only mode and perform a query_texts
        embeddings_readonly = ChromaDBEmbeddings(
            persist_directory=str(db_path),
            collection_name=collection_name,
            embedding_model="text-embedding-004",
            dimensionality=target_dim,
            read_only=True
        )
        await embeddings_readonly.ensure_cache_initialized()
        
        # ASSERT: Embedding function should be initialized despite read_only=True
        assert embeddings_readonly._embedding_function is not None, "Embedding function should be initialized in read-only mode"
        
        # We need to mock bm.genai because GeminiEmbeddingFunction uses it lazily
        from unittest.mock import MagicMock
        mock_genai = MagicMock()
        mock_embed_response = MagicMock()
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = mock_embedding
        mock_embed_response.embeddings = [mock_embedding_obj]
        mock_genai.models.embed_content.return_value = mock_embed_response
        
        # Mock get_bm to return a mock BM object with our mock genai client
        mock_bm_obj = MagicMock()
        mock_bm_obj.genai = mock_genai
        
        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_obj):
            # This would fail with "Got 384, expected 768" if fix was not applied
            results = embeddings_readonly.collection.query(
                query_texts=["search query"],
                n_results=1
            )
            
            # Verify results
            assert results["ids"][0][0] == "doc1"
            assert results["documents"][0][0] == "This is a test document"
            
            # Verify our mock was called (confirming our embedder was used)
            mock_genai.models.embed_content.assert_called_once()
            call_args = mock_genai.models.embed_content.call_args
            assert call_args.kwargs["config"]["output_dimensionality"] == target_dim

@pytest.mark.anyio
async def test_chromadb_search_tool_readonly_fix():
    """Verify that ChromaDBSearchTool works correctly with read_only=True."""
    from buttermilk.tools.chromadb_search import ChromaDBSearchTool
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_search_tool"
        target_dim = 768
        collection_name = "search_tool_test"
        
        # 1. Setup existing collection
        embeddings_write = ChromaDBEmbeddings(
            persist_directory=str(db_path),
            collection_name=collection_name,
            embedding_model="text-embedding-004",
            dimensionality=target_dim,
            read_only=False
        )
        await embeddings_write.ensure_cache_initialized()
        
        mock_embedding = [0.1] * target_dim
        with patch("vertexai.language_models.TextEmbeddingModel.get_embeddings") as mock_get_embeddings:
            mock_result = AsyncMock()
            mock_result.values = mock_embedding
            mock_get_embeddings.return_value = [mock_result]
            with patch("vertexai.language_models.TextEmbeddingModel.get_embeddings_async") as mock_get_embeddings_async:
                mock_get_embeddings_async.return_value = [mock_result]
                embeddings_write.collection.add(
                    ids=["doc1"],
                    embeddings=[mock_embedding],
                    documents=["Search tool content"],
                    metadatas=[{"document_id": "doc1", "document_title": "Test Title"}]
                )
        
        # 2. Test Search Tool in read-only mode
        search_tool = ChromaDBSearchTool(
            persist_directory=str(db_path),
            collection_name=collection_name,
            embedding_model="text-embedding-004",
            dimensionality=target_dim,
            read_only=True
        )
        
        from unittest.mock import MagicMock
        mock_genai = MagicMock()
        mock_embed_response = MagicMock()
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = mock_embedding
        mock_embed_response.embeddings = [mock_embedding_obj]
        mock_genai.models.embed_content.return_value = mock_embed_response
        
        mock_bm_obj = MagicMock()
        mock_bm_obj.genai = mock_genai
        
        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_obj):
            # This would fail without the fix
            results = await search_tool.search("test query", n_results=1)
            
            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].content == "Search tool content"
            mock_genai.models.embed_content.assert_called_once()

@pytest.mark.anyio
async def test_chromadb_readonly_still_gates_writes():
    """Verify that read_only=True still prevents write operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chromadb_gates"
        
        embeddings = ChromaDBEmbeddings(
            persist_directory=str(db_path),
            collection_name="gates_test",
            embedding_model="text-embedding-004",
            dimensionality=768,
            read_only=True
        )
        await embeddings.ensure_cache_initialized()
        
        from buttermilk._core.types import Record
        test_record = Record(record_id="test", dataset="test", content="content")
        
        # process_record should raise RuntimeError in read-only mode
        with pytest.raises(RuntimeError, match="read-only mode"):
            await embeddings.process_record(test_record)
        
        # upsert_document_chunks should also raise RuntimeError
        with pytest.raises(RuntimeError, match="read-only mode"):
            await embeddings.upsert_document_chunks([test_record])
