"""Integration tests for Zotero vector database functionality.

Tests the complete pipeline from Zotero data retrieval through vectorization
and storage in ChromaDB, including:
1. Record creation from Zotero items
2. PDF download
3. Full-text extraction from PDFs
4. Annotation extraction from PDFs (when implemented)
5. ChromaDB collection creation
6. Vectorization with different chunk sizes
7. Batch upsert operations
8. Interruption handling and resume functionality
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk.data.vector import (
    ChromaDBEmbeddings,
)
from buttermilk.libs.zotero import ZotDownloader

# Test fixtures and data
MOCK_ZOTERO_ITEM = {
    "key": "TEST123",
    "data": {
        "title": "Test Research Paper",
        "DOI": "10.1234/test.2024.001",
        "url": "https://example.com/paper.pdf",
        "itemType": "journalArticle",
        "creators": [{"firstName": "John", "lastName": "Doe"}],
        "date": "2024-01-01",
        "abstractNote": "This is a test abstract for the paper.",
    },
    "links": {
        "attachment": {
            "href": "https://api.zotero.org/groups/123/items/ATTACH123/file",
            "attachmentType": "application/pdf",
        }
    },
}

MOCK_FULLTEXT_RESPONSE = {
    "content": """This is the full text content of the test paper.
    It contains multiple paragraphs and sections that will be chunked.
    
    Section 1: Introduction
    This section introduces the main concepts.
    
    Section 2: Methods
    This section describes the methodology used.
    
    Section 3: Results
    This section presents the findings.
    
    Section 4: Conclusion
    This section summarizes the key points.""",
    "indexedChars": 500,
    "totalChars": 500,
}

MOCK_ANNOTATIONS = [
    {
        "key": "ANNOT1",
        "data": {
            "annotationType": "highlight",
            "annotationText": "Important finding",
            "annotationComment": "This is a key result",
            "annotationPageLabel": "5",
        },
    },
    {
        "key": "ANNOT2",
        "data": {
            "annotationType": "note",
            "annotationText": "Methodology question",
            "annotationComment": "Need to review this approach",
            "annotationPageLabel": "3",
        },
    },
]


class TestZoteroVectorIntegration:
    """Test suite for Zotero vector database integration."""

    @pytest.fixture
    async def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as save_dir:
            with tempfile.TemporaryDirectory() as vector_dir:
                yield save_dir, vector_dir

    @pytest.fixture
    async def mock_zotero_api(self):
        """Mock Zotero API responses."""
        with patch("buttermilk.libs.zotero.zotero.Zotero") as mock_zot:
            instance = mock_zot.return_value
            instance.items.return_value = [MOCK_ZOTERO_ITEM]
            instance.links = {"next": None}
            instance.fulltext_item.return_value = MOCK_FULLTEXT_RESPONSE
            instance.dump = Mock()  # Mock PDF download
            yield instance

    @pytest.fixture
    async def mock_embeddings(self):
        """Mock embedding generation."""
        with patch("buttermilk.data.vector.TextEmbeddingModel") as mock_model:
            instance = mock_model.from_pretrained.return_value
            # Mock embedding results
            mock_embedding = [0.1] * 768  # 768-dimensional embedding
            instance.get_embeddings_async = AsyncMock(return_value=[Mock(values=mock_embedding)])
            yield instance

    @pytest.fixture
    async def mock_chromadb(self):
        """Mock ChromaDB client."""
        with patch("buttermilk.data.vector.chromadb.PersistentClient") as mock_client:
            instance = mock_client.return_value
            collection = Mock()
            collection.count.return_value = 0
            collection.get.return_value = {"ids": []}
            collection.upsert = Mock()
            instance.get_or_create_collection.return_value = collection
            instance.get_collection.return_value = collection
            instance.list_collections.return_value = []
            yield instance, collection

    @pytest.mark.anyio
    async def test_record_creation_from_zotero(self, temp_dirs, mock_zotero_api):
        """Test 1: Creating records from Zotero items."""
        save_dir, _ = temp_dirs
        downloader = ZotDownloader(save_dir=save_dir, library="test_library")
        downloader._zot = mock_zotero_api

        records = []
        async for record in downloader.get_all_records():
            records.append(record)

        assert len(records) == 1
        record = records[0]
        assert record.record_id == "TEST123"
        assert record.metadata["title"] == "Test Research Paper"
        assert record.metadata["doi_or_url"] == "10.1234/test.2024.001"
        assert "zotero_data" in record.metadata
        assert record.file_path.endswith("TEST123.pdf")

    @pytest.mark.anyio
    async def test_pdf_download(self, temp_dirs, mock_zotero_api):
        """Test 2: PDF download functionality."""
        save_dir, _ = temp_dirs
        downloader = ZotDownloader(save_dir=save_dir, library="test_library")
        downloader._zot = mock_zotero_api

        # Mock the dump method to create a fake PDF file
        def create_fake_pdf(key, path):
            Path(path).write_bytes(b"Fake PDF content")

        mock_zotero_api.dump.side_effect = create_fake_pdf

        records = []
        async for record in downloader.get_all_records():
            records.append(record)

        # Check that PDF was downloaded
        pdf_path = Path(save_dir) / "TEST123.pdf"
        assert pdf_path.exists()
        assert pdf_path.read_bytes() == b"Fake PDF content"

        # Check that dump was called with correct parameters
        mock_zotero_api.dump.assert_called_once()

    @pytest.mark.anyio
    async def test_fulltext_extraction(self, temp_dirs, mock_zotero_api):
        """Test 3: Full-text extraction from Zotero."""
        save_dir, _ = temp_dirs
        downloader = ZotDownloader(save_dir=save_dir, library="test_library")
        downloader._zot = mock_zotero_api

        records = []
        async for record in downloader.get_all_records():
            records.append(record)

        assert len(records) == 1
        record = records[0]
        assert record.content == MOCK_FULLTEXT_RESPONSE
        assert record.text_content == MOCK_FULLTEXT_RESPONSE["content"]

    @pytest.mark.anyio
    async def test_annotation_extraction_not_implemented(self, temp_dirs, mock_zotero_api):
        """Test 4: Annotation extraction (expecting failure as not implemented)."""
        save_dir, _ = temp_dirs
        downloader = ZotDownloader(save_dir=save_dir, library="test_library")
        downloader._zot = mock_zotero_api

        # Add mock for annotations endpoint (future implementation)
        mock_zotero_api.children.return_value = MOCK_ANNOTATIONS

        records = []
        async for record in downloader.get_all_records():
            records.append(record)

        # Currently, annotations are not extracted
        # This test documents expected future behavior
        record = records[0]
        # TODO: When implemented, check for annotations in metadata
        assert "annotations" not in record.metadata

    @pytest.mark.anyio
    async def test_chromadb_collections_creation(self, temp_dirs, mock_zotero_api, mock_embeddings, mock_chromadb):
        """Test 5: ChromaDB collection creation for different record types."""
        save_dir, vector_dir = temp_dirs
        _, mock_collection = mock_chromadb

        # Create vector store with multi-field config for different content types
        multi_config = MultiFieldEmbeddingConfig(
            content_field="text_content",
            chunk_size=1000,
            chunk_overlap=200,
            additional_fields=[
                {"source_field": "abstract", "chunk_type": "abstract", "min_length": 50},
                {"source_field": "annotations", "chunk_type": "annotation", "min_length": 20},
            ],
        )

        vector_store = ChromaDBEmbeddings(
            persist_directory=vector_dir, collection_name="test_zotero", embedding_model="text-embedding-005", dimensionality=768
        )

        # Ensure cache is initialized
        await vector_store.ensure_cache_initialized()

        # Create test record with multiple content types
        test_record = Record(
            record_id="TEST123",
            content=MOCK_FULLTEXT_RESPONSE["content"],
            file_path=f"{save_dir}/TEST123.pdf",
            metadata={
                "title": "Test Paper",
                "abstract": "This is a test abstract that should be embedded separately.",
                "annotations": ["Annotation 1: Important finding", "Annotation 2: Method note"],
            },
        )

        # Process the record
        result = await vector_store.process_record(test_record)

        assert result.status == "processed"
        assert result.chunks_created > 0

        # Verify different chunk types were created
        chunk_types = result.metadata.get("chunk_types", {})
        assert "content" in chunk_types  # Main content chunks
        assert "abstract" in chunk_types  # Abstract as separate chunk

    @pytest.mark.anyio
    async def test_vectorization_different_chunk_sizes(self, temp_dirs, mock_embeddings, mock_chromadb):
        """Test 6: Vectorization with different chunk sizes."""
        _, vector_dir = temp_dirs
        _, mock_collection = mock_chromadb

        # Test different chunk size configurations
        chunk_configs = [
            (500, 100),  # Small chunks
            (2000, 500),  # Medium chunks
            (8000, 1000),  # Large chunks
        ]

        for chunk_size, chunk_overlap in chunk_configs:
            # Create vector store with specific chunk size
            vector_store = ChromaDBEmbeddings(
                persist_directory=vector_dir,
                collection_name=f"test_chunks_{chunk_size}",
                embedding_model="text-embedding-005",
                dimensionality=768,
            )

            await vector_store.ensure_cache_initialized()

            # Create test record with long content
            long_content = " ".join([f"Paragraph {i}. " * 50 for i in range(20)])
            test_record = Record(
                record_id=f"TEST_CHUNK_{chunk_size}",
                content=long_content,
                metadata={"chunk_size_test": chunk_size},
            )

            # Create chunks manually to test chunking
            chunks = vector_store.create_multi_field_chunks_for_record(test_record)

            # Verify chunks respect size limits
            for chunk in chunks:
                assert len(chunk.chunk_text) <= chunk_size + chunk_overlap
                if chunk.chunk_index > 0:
                    # Verify overlap exists (except for first chunk)
                    assert len(chunk.chunk_text) > chunk_size - chunk_overlap

            logger.info("Created chunks", num_chunks=len(chunks), chunk_size=chunk_size)

    @pytest.mark.anyio
    async def test_batch_upsert(self, temp_dirs, mock_embeddings, mock_chromadb):
        """Test 7: Batch upsert functionality."""
        _, vector_dir = temp_dirs
        _, mock_collection = mock_chromadb

        vector_store = ChromaDBEmbeddings(
            persist_directory=vector_dir,
            collection_name="test_batch",
            embedding_model="text-embedding-005",
            dimensionality=768,
            sync_batch_size=5,  # Small batch for testing
        )

        await vector_store.ensure_cache_initialized()

        # Create multiple test records
        test_records = []
        for i in range(10):
            record = Record(
                record_id=f"BATCH_TEST_{i}",
                content=f"Test content for record {i}",
                metadata={"batch_index": i},
            )
            test_records.append(record)

        # Process batch
        result = await vector_store.process_batch(
            test_records,
            mode="safe",
            max_failures=2,
        )

        assert result.total_records == 10
        assert result.successful_count == 10
        assert result.failed_count == 0
        assert result.processing_time_ms > 0

        # Verify upsert was called
        assert mock_collection.upsert.call_count > 0

    @pytest.mark.anyio
    async def test_interruption_and_resume(self, temp_dirs, mock_zotero_api, mock_embeddings, mock_chromadb):
        """Test 8: Interruption handling and resume functionality."""
        save_dir, vector_dir = temp_dirs
        _, mock_collection = mock_chromadb

        # Setup Zotero downloader with vector store
        downloader = ZotDownloader(save_dir=save_dir, library="test_library")
        downloader._zot = mock_zotero_api

        vector_store = ChromaDBEmbeddings(
            persist_directory=vector_dir,
            collection_name="test_resume",
            embedding_model="text-embedding-005",
            dimensionality=768,
        )

        await vector_store.ensure_cache_initialized()

        # Link vector store to downloader for deduplication
        downloader.set_vector_store(vector_store)

        # First run - process some items
        mock_zotero_api.items.return_value = [{**MOCK_ZOTERO_ITEM, "key": f"ITEM{i}"} for i in range(5)]

        processed_first_run = 0
        async for record in downloader.get_all_records():
            result = await vector_store.process_record(record)
            if result.status == "processed":
                processed_first_run += 1
            if processed_first_run >= 3:
                break  # Simulate interruption

        assert processed_first_run == 3

        # Second run - resume with If-Modified-Since-Version
        # Mock that some items already exist
        def mock_get_existing(where, **kwargs):
            doc_id = where.get("$and", [{}])[0].get("document_id", "")
            if doc_id in ["ITEM0", "ITEM1", "ITEM2"]:
                return {"ids": [f"chunk_{doc_id}"]}
            return {"ids": []}

        mock_collection.get.side_effect = mock_get_existing

        # Resume processing
        processed_second_run = 0
        skipped_second_run = 0

        async for record in downloader.get_all_records():
            # Vector store should skip already processed items
            should_skip, reason = await vector_store._should_skip_record(record)
            if should_skip:
                skipped_second_run += 1
            else:
                result = await vector_store.process_record(record)
                if result.status == "processed":
                    processed_second_run += 1

        # Should skip the 3 already processed, process the remaining 2
        assert skipped_second_run >= 3
        assert processed_second_run == 2

    @pytest.mark.anyio
    async def test_last_modified_version_handling(self, temp_dirs, mock_zotero_api):
        """Test 9: Last-Modified-Version and If-Modified-Since-Version handling."""
        save_dir, _ = temp_dirs
        downloader = ZotDownloader(save_dir=save_dir, library="test_library")
        downloader._zot = mock_zotero_api

        # Mock version headers
        mock_zotero_api.request.headers = {"Last-Modified-Version": "1234"}

        # First request - no If-Modified-Since-Version
        items = []
        async for record in downloader.get_all_records():
            items.append(record)

        # TODO: Implement version tracking in ZotDownloader
        # Currently not implemented - this documents expected behavior

        # Expected future behavior:
        # 1. Store Last-Modified-Version from response
        # 2. Use If-Modified-Since-Version in subsequent requests
        # 3. Handle 304 Not Modified responses

    @pytest.mark.anyio
    async def test_remote_chromadb_sync(self, temp_dirs, mock_embeddings, mock_chromadb):
        """Test 10: Remote ChromaDB sync functionality."""
        _, vector_dir = temp_dirs
        _, mock_collection = mock_chromadb

        # Mock GCS operations
        with patch("buttermilk.utils.utils.ensure_chromadb_cache") as mock_cache:
            with patch("buttermilk.utils.utils.upload_chromadb_cache") as mock_upload:
                mock_cache.return_value = Path(vector_dir)

                vector_store = ChromaDBEmbeddings(
                    persist_directory="gs://test-bucket/chromadb",
                    collection_name="test_sync",
                    embedding_model="text-embedding-005",
                    dimensionality=768,
                    sync_batch_size=2,
                    sync_interval_minutes=1,
                )

                await vector_store.ensure_cache_initialized()

                # Process multiple records to trigger sync
                for i in range(5):
                    record = Record(
                        record_id=f"SYNC_TEST_{i}",
                        content=f"Content {i}",
                    )
                    await vector_store.process_record(record)

                # Force final sync
                await vector_store.finalize_processing()

                # Verify sync was called
                assert mock_upload.called
                logger.info("Upload called", call_count=mock_upload.call_count)


@pytest.mark.anyio
async def test_full_pipeline_integration():
    """End-to-end test of the complete Zotero to vector database pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "zotero"
        vector_dir = Path(temp_dir) / "chromadb"
        save_dir.mkdir()
        vector_dir.mkdir()

        # This would be a full integration test with real services
        # Currently marked as a template for manual testing
        pytest.skip("Full integration test requires real Zotero API and Vertex AI access")

        # Example of what full integration would look like:
        """
        # Initialize components
        downloader = ZotDownloader(
            save_dir=str(save_dir),
            library=os.environ.get("ZOTERO_LIBRARY_ID")
        )
        
        vector_store = ChromaDBEmbeddings(
            persist_directory=str(vector_dir),
            collection_name="zotero_integration_test",
            embedding_model="text-embedding-005",
            dimensionality=768
        )
        
        await vector_store.ensure_cache_initialized()
        downloader.set_vector_store(vector_store)
        
        # Process records
        processed = 0
        async for record in downloader.get_all_records(limit=10):
            result = await vector_store.process_record(record)
            if result.status == "processed":
                processed += 1
                logger.info("Processed record", record_id=record.record_id, chunks_created=result.chunks_created)
        
        # Finalize
        await vector_store.finalize_processing()
        
        assert processed > 0
        logger.info("Successfully processed records", processed_count=processed)
        """


if __name__ == "__main__":
    # Run specific test for debugging
    asyncio.run(test_full_pipeline_integration())
