"""Comprehensive test suite for the enhanced Record class.

This test suite ensures that the enhanced Record class maintains all existing
functionality while adding new vector capabilities, and that the migration
from InputDocument works correctly.
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Import our enhanced classes (in real implementation, these would be proper imports)
from enhanced_record_design import EnhancedRecord, ChunkData, RecordMigration
from migration_strategy import CompatibilityLayer, MigrationConfig, MigrationPhase
from vector_processing_updates import EnhancedChromaDBEmbeddings, EnhancedVectorSearch
from configuration_integration import (
    AdvancedMultiFieldConfig, ConditionalFieldConfig, 
    ConfigurationTemplates, ConfigurationValidator
)

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


# ========== FIXTURES ==========

@pytest.fixture
def sample_record_data():
    """Sample data for creating test records."""
    return {
        "record_id": "test_record_001",
        "content": "This is a test document with some interesting content about machine learning and AI.",
        "metadata": {
            "title": "Test Document",
            "author": "Test Author",
            "source": "test_suite",
            "abstract": "A brief abstract about the document content and its purpose.",
            "keywords": "machine learning, AI, test, document"
        },
        "mime": "text/plain",
        "uri": "https://example.com/test-document"
    }


@pytest.fixture
def sample_legacy_input_document():
    """Sample InputDocument for testing migration."""
    from buttermilk.data.vector import InputDocument, ChunkedDocument
    
    chunks = [
        ChunkedDocument(
            chunk_id="chunk_001",
            document_title="Test Document",
            chunk_index=0,
            chunk_text="This is the first chunk of text.",
            document_id="test_record_001",
            embedding=[0.1, 0.2, 0.3],
            metadata={"chunk_type": "content"}
        ),
        ChunkedDocument(
            chunk_id="chunk_002", 
            document_title="Test Document",
            chunk_index=1,
            chunk_text="This is the second chunk of text.",
            document_id="test_record_001",
            embedding=[0.4, 0.5, 0.6],
            metadata={"chunk_type": "content"}
        )
    ]
    
    return InputDocument(
        record_id="test_record_001",
        file_path="/path/to/test.pdf",
        record_path="/path/to/record",
        chunks_path="/path/to/chunks.parquet",
        full_text="This is the first chunk of text. This is the second chunk of text.",
        chunks=chunks,
        title="Test Document",
        metadata={"author": "Test Author", "source": "test"}
    )


@pytest.fixture
def multi_field_config():
    """Sample multi-field embedding configuration."""
    return AdvancedMultiFieldConfig(
        content_field="content",
        conditional_fields=[
            ConditionalFieldConfig(
                source_field="title",
                chunk_type="title",
                min_length=5,
                max_length=200
            ),
            ConditionalFieldConfig(
                source_field="abstract",
                chunk_type="abstract",
                min_length=20,
                max_length=1000
            ),
            ConditionalFieldConfig(
                source_field="keywords",
                chunk_type="keywords",
                min_length=5,
                max_length=500
            )
        ],
        chunk_size=500,
        chunk_overlap=100
    )


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3] * 1024  # Mock 3072-dimensional embedding
    mock_model.get_embeddings_async = AsyncMock(return_value=[mock_embedding])
    return mock_model


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for testing."""
    mock_collection = MagicMock()
    mock_collection.upsert = MagicMock()
    mock_collection.query = MagicMock(return_value={
        "documents": [["Test document content"]],
        "metadatas": [[{"document_id": "test_001", "chunk_type": "content"}]],
        "distances": [[0.1]],
        "ids": [["chunk_001"]]
    })
    return mock_collection


# ========== ENHANCED RECORD TESTS ==========

class TestEnhancedRecord:
    """Test suite for the EnhancedRecord class."""
    
    def test_create_enhanced_record(self, sample_record_data):
        """Test creating an EnhancedRecord with basic data."""
        record = EnhancedRecord(**sample_record_data)
        
        assert record.record_id == sample_record_data["record_id"]
        assert record.content == sample_record_data["content"]
        assert record.metadata == sample_record_data["metadata"]
        assert record.text_content == sample_record_data["content"]
        assert record.title == "Test Document"
        assert len(record.chunks) == 0
        
    def test_enhanced_record_auto_populate_full_text(self):
        """Test that full_text is auto-populated from content."""
        record = EnhancedRecord(content="Test content string")
        assert record.full_text == "Test content string"
        assert record.text_content == "Test content string"
        
    def test_enhanced_record_with_explicit_full_text(self):
        """Test EnhancedRecord with explicitly provided full_text."""
        record = EnhancedRecord(
            content="Short content",
            full_text="This is the full text content that is longer than the main content"
        )
        assert record.text_content == "This is the full text content that is longer than the main content"
        
    def test_add_chunk(self, sample_record_data):
        """Test adding chunks to an EnhancedRecord."""
        record = EnhancedRecord(**sample_record_data)
        
        # Add chunks
        chunk1 = record.add_chunk("First chunk text", chunk_type="content")
        chunk2 = record.add_chunk("Title text", chunk_type="title", source_field="title")
        
        assert len(record.chunks) == 2
        assert chunk1.chunk_index == 0
        assert chunk2.chunk_index == 1
        assert chunk1.chunk_type == "content"
        assert chunk2.chunk_type == "title"
        assert chunk2.source_field == "title"
        
    def test_chunk_operations(self, sample_record_data):
        """Test chunk filtering and querying operations."""
        record = EnhancedRecord(**sample_record_data)
        
        # Add various chunk types
        record.add_chunk("Content chunk 1", chunk_type="content")
        record.add_chunk("Content chunk 2", chunk_type="content")
        record.add_chunk("Title chunk", chunk_type="title", source_field="title")
        record.add_chunk("Abstract chunk", chunk_type="abstract", source_field="abstract")
        
        # Test filtering
        content_chunks = record.get_chunks_by_type("content")
        title_chunks = record.get_chunks_by_type("title")
        title_source_chunks = record.get_chunks_by_source("title")
        
        assert len(content_chunks) == 2
        assert len(title_chunks) == 1
        assert len(title_source_chunks) == 1
        assert record.chunk_types == {"content", "title", "abstract"}
        
    def test_computed_properties(self, sample_record_data):
        """Test computed properties of EnhancedRecord."""
        record = EnhancedRecord(**sample_record_data)
        
        # Initially no embeddings
        assert not record.has_embeddings
        assert record.embedding_count == 0
        
        # Add chunks with embeddings
        chunk1 = record.add_chunk("Text 1")
        chunk1.embedding = [0.1, 0.2, 0.3]
        
        chunk2 = record.add_chunk("Text 2")
        # chunk2 has no embedding
        
        chunk3 = record.add_chunk("Text 3")
        chunk3.embedding = [0.4, 0.5, 0.6]
        
        assert record.has_embeddings
        assert record.embedding_count == 2
        
    def test_multimodal_content(self):
        """Test EnhancedRecord with multimodal content."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        record = EnhancedRecord(
            content=["Text content", test_image, "More text"],
            alt_text="Image description"
        )
        
        assert record.images is not None
        assert len(record.images) == 1
        assert record.text_content == ""  # No string content, should fall back to alt_text
        
        # Test with alt_text fallback
        record_with_alt = EnhancedRecord(
            content=["Text content", test_image],
            alt_text="Combined text and image description"
        )
        assert "Combined text and image description" in record_with_alt.as_markdown()
        
    def test_serialization(self, sample_record_data):
        """Test serialization and deserialization of EnhancedRecord."""
        record = EnhancedRecord(**sample_record_data)
        record.add_chunk("Test chunk", chunk_type="content")
        record.chunks[0].embedding = [0.1, 0.2, 0.3]
        
        # Test model_dump
        data = record.model_dump()
        assert "record_id" in data
        assert "chunks" in data
        assert "vector_metadata" in data
        
        # Test reconstruction
        new_record = EnhancedRecord(**data)
        assert new_record.record_id == record.record_id
        assert len(new_record.chunks) == len(record.chunks)
        assert new_record.chunks[0].embedding == record.chunks[0].embedding


# ========== MIGRATION TESTS ==========

class TestMigration:
    """Test suite for migration functionality."""
    
    def test_input_document_to_enhanced_record(self, sample_legacy_input_document):
        """Test converting InputDocument to EnhancedRecord."""
        enhanced_record = EnhancedRecord.from_input_document(sample_legacy_input_document)
        
        assert enhanced_record.record_id == sample_legacy_input_document.record_id
        assert enhanced_record.title == sample_legacy_input_document.title
        assert enhanced_record.full_text == sample_legacy_input_document.full_text
        assert enhanced_record.file_path == sample_legacy_input_document.file_path
        assert len(enhanced_record.chunks) == len(sample_legacy_input_document.chunks)
        
        # Check chunk conversion
        for i, chunk in enumerate(enhanced_record.chunks):
            original_chunk = sample_legacy_input_document.chunks[i]
            assert chunk.chunk_id == original_chunk.chunk_id
            assert chunk.chunk_text == original_chunk.chunk_text
            assert chunk.embedding == original_chunk.embedding
            
    def test_enhanced_record_to_input_document(self, sample_record_data):
        """Test converting EnhancedRecord back to InputDocument."""
        record = EnhancedRecord(**sample_record_data)
        record.add_chunk("Test chunk 1", chunk_type="content")
        record.add_chunk("Test chunk 2", chunk_type="title")
        
        input_doc = record.as_input_document()
        
        assert input_doc.record_id == record.record_id
        assert input_doc.title == record.title
        assert input_doc.full_text == record.text_content
        assert len(input_doc.chunks) == len(record.chunks)
        
    def test_compatibility_layer(self, sample_record_data, sample_legacy_input_document):
        """Test the compatibility layer for normalizing records."""
        # Test with EnhancedRecord
        enhanced = EnhancedRecord(**sample_record_data)
        normalized_enhanced = CompatibilityLayer.normalize_record(enhanced)
        assert isinstance(normalized_enhanced, EnhancedRecord)
        assert normalized_enhanced.record_id == enhanced.record_id
        
        # Test with InputDocument
        normalized_input = CompatibilityLayer.normalize_record(sample_legacy_input_document)
        assert isinstance(normalized_input, EnhancedRecord)
        assert normalized_input.record_id == sample_legacy_input_document.record_id
        
        # Test with dict-like object
        dict_record = {"record_id": "dict_test", "content": "test content"}
        normalized_dict = CompatibilityLayer.normalize_record(dict_record)
        assert isinstance(normalized_dict, EnhancedRecord)
        assert normalized_dict.record_id == "dict_test"
        
    def test_batch_migration(self, sample_record_data, sample_legacy_input_document):
        """Test batch migration of multiple records."""
        records = [
            EnhancedRecord(**sample_record_data),
            sample_legacy_input_document,
            {"record_id": "dict_record", "content": "dict content"}
        ]
        
        migrated = CompatibilityLayer.batch_normalize_records(records)
        
        assert len(migrated) == 3
        assert all(isinstance(r, EnhancedRecord) for r in migrated)
        assert migrated[0].record_id == sample_record_data["record_id"]
        assert migrated[1].record_id == sample_legacy_input_document.record_id
        assert migrated[2].record_id == "dict_record"


# ========== VECTOR PROCESSING TESTS ==========

class TestVectorProcessing:
    """Test suite for vector processing with EnhancedRecord."""
    
    @patch('buttermilk.data.vector.TextEmbeddingModel.from_pretrained')
    @patch('chromadb.PersistentClient')
    async def test_enhanced_chromadb_embeddings(
        self, 
        mock_chromadb_client,
        mock_embedding_model_class,
        sample_record_data,
        multi_field_config,
        mock_embedding_model,
        mock_chroma_collection
    ):
        """Test enhanced ChromaDB embeddings with multi-field support."""
        # Setup mocks
        mock_embedding_model_class.return_value = mock_embedding_model
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb_client.return_value = mock_client
        
        # Create embedder
        embedder = EnhancedChromaDBEmbeddings(
            collection_name="test_collection",
            persist_directory="./test_chromadb",
            multi_field_config=multi_field_config
        )
        
        # Mock the collection property
        embedder._collection = mock_chroma_collection
        
        # Create test record
        record = EnhancedRecord(**sample_record_data)
        
        # Process record
        with patch.object(embedder, 'collection', mock_chroma_collection):
            processed_record = await embedder.process_enhanced_record_async(record)
        
        # Verify results
        assert processed_record is not None
        assert len(processed_record.chunks) > 0
        
        # Should have chunks for content, title, abstract, and keywords
        chunk_types = processed_record.chunk_types
        expected_types = {"content", "title", "abstract", "keywords"}
        assert expected_types.issubset(chunk_types)
        
        # Verify embedding calls were made
        mock_embedding_model.get_embeddings_async.assert_called()
        
        # Verify ChromaDB upsert was called
        mock_chroma_collection.upsert.assert_called()
        
    def test_multi_field_chunking(self, sample_record_data, multi_field_config):
        """Test multi-field chunking functionality."""
        embedder = EnhancedChromaDBEmbeddings(
            collection_name="test",
            persist_directory="./test",
            multi_field_config=multi_field_config
        )
        
        record = EnhancedRecord(**sample_record_data)
        chunks = embedder.create_multi_field_chunks_enhanced(record)
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Check that different field types are represented
        chunk_types = {chunk.chunk_type for chunk in chunks}
        expected_types = {"content", "title", "abstract", "keywords"}
        assert expected_types.issubset(chunk_types)
        
        # Verify content chunks
        content_chunks = [c for c in chunks if c.chunk_type == "content"]
        assert len(content_chunks) > 0
        
        # Verify metadata chunks (title, abstract, keywords)
        title_chunks = [c for c in chunks if c.chunk_type == "title"]
        abstract_chunks = [c for c in chunks if c.chunk_type == "abstract"]
        keywords_chunks = [c for c in chunks if c.chunk_type == "keywords"]
        
        assert len(title_chunks) == 1
        assert len(abstract_chunks) == 1
        assert len(keywords_chunks) == 1
        
        # Verify chunk content
        assert title_chunks[0].chunk_text == sample_record_data["metadata"]["title"]
        assert abstract_chunks[0].chunk_text == sample_record_data["metadata"]["abstract"]
        assert keywords_chunks[0].chunk_text == sample_record_data["metadata"]["keywords"]
        
    async def test_vector_search(self, mock_chroma_collection, mock_embedding_model):
        """Test enhanced vector search functionality."""
        embedder = EnhancedChromaDBEmbeddings(
            collection_name="test",
            persist_directory="./test"
        )
        embedder._embedding_model = mock_embedding_model
        embedder._collection = mock_chroma_collection
        
        search = EnhancedVectorSearch(embedder)
        
        # Test search by content type
        results = await search.search_by_content_type(
            query="machine learning",
            content_types=["content", "abstract"],
            limit=5
        )
        
        assert len(results) > 0
        assert "document" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]
        
        # Verify ChromaDB query was called with correct parameters
        mock_chroma_collection.query.assert_called()
        call_args = mock_chroma_collection.query.call_args
        assert "where" in call_args.kwargs
        assert call_args.kwargs["where"]["content_type"]["$in"] == ["content", "abstract"]


# ========== CONFIGURATION TESTS ==========

class TestConfiguration:
    """Test suite for configuration integration."""
    
    def test_configuration_templates(self):
        """Test predefined configuration templates."""
        # Test academic papers template
        academic_config = ConfigurationTemplates.academic_papers()
        assert academic_config.content_field == "content"
        assert len(academic_config.conditional_fields) > 0
        assert any(field.chunk_type == "title" for field in academic_config.conditional_fields)
        assert any(field.chunk_type == "abstract" for field in academic_config.conditional_fields)
        
        # Test news articles template
        news_config = ConfigurationTemplates.news_articles()
        assert news_config.chunk_size == 800
        assert any(field.chunk_type == "headline" for field in news_config.conditional_fields)
        
        # Test technical documentation template
        tech_config = ConfigurationTemplates.technical_documentation()
        assert tech_config.chunk_size == 600
        assert any(field.chunk_type == "api_reference" for field in tech_config.conditional_fields)
        
    def test_configuration_validation(self, multi_field_config):
        """Test configuration validation."""
        # Valid configuration should pass
        issues = ConfigurationValidator.validate_multi_field_config(multi_field_config)
        assert len(issues) == 0
        
        # Invalid configuration should fail
        invalid_config = AdvancedMultiFieldConfig(
            chunk_size=-100,  # Invalid
            chunk_overlap=1000,  # Larger than chunk_size
            min_embedding_quality=1.5  # Out of range
        )
        
        issues = ConfigurationValidator.validate_multi_field_config(invalid_config)
        assert len(issues) > 0
        assert any("chunk_size must be positive" in issue for issue in issues)
        assert any("chunk_overlap must be less than chunk_size" in issue for issue in issues)
        
    def test_field_embedding_strategy(self, multi_field_config):
        """Test field embedding strategy determination."""
        from configuration_integration import ConfigurationUtils
        
        # Test content that should be embedded
        strategy = ConfigurationUtils.get_field_embedding_strategy(
            multi_field_config,
            "abstract",
            "This is a long enough abstract that meets the minimum length requirements for embedding."
        )
        
        assert strategy["should_embed"] is True
        assert strategy["chunk_type"] == "abstract"
        
        # Test content that's too short
        strategy = ConfigurationUtils.get_field_embedding_strategy(
            multi_field_config,
            "abstract",
            "Too short"  # Less than min_length of 20
        )
        
        assert strategy["should_embed"] is False
        assert "too short" in strategy["reasons"][0].lower()


# ========== PERFORMANCE TESTS ==========

class TestPerformance:
    """Test suite for performance considerations."""
    
    async def test_large_record_processing(self, multi_field_config):
        """Test processing of large records with many chunks."""
        # Create a large record
        large_content = "This is a test sentence. " * 1000  # ~25KB of text
        
        record = EnhancedRecord(
            content=large_content,
            metadata={
                "title": "Large Document",
                "abstract": "This is a large document for testing performance. " * 20,
                "keywords": "performance, testing, large, document"
            }
        )
        
        embedder = EnhancedChromaDBEmbeddings(
            collection_name="test",
            persist_directory="./test",
            multi_field_config=multi_field_config
        )
        
        # Test chunking performance
        chunks = embedder.create_multi_field_chunks_enhanced(record)
        
        assert len(chunks) > 10  # Should create many chunks
        assert all(len(chunk.chunk_text) <= multi_field_config.chunk_size + 100 for chunk in chunks)
        
    def test_memory_efficiency(self, sample_record_data):
        """Test memory efficiency of EnhancedRecord."""
        # Create many records
        records = []
        for i in range(1000):
            data = sample_record_data.copy()
            data["record_id"] = f"record_{i}"
            record = EnhancedRecord(**data)
            records.append(record)
        
        # Basic memory usage test - just ensure we can create many records
        assert len(records) == 1000
        assert all(isinstance(r, EnhancedRecord) for r in records)
        
    async def test_concurrent_processing(self, sample_record_data, multi_field_config):
        """Test concurrent processing of multiple records."""
        # Create multiple records
        records = []
        for i in range(10):
            data = sample_record_data.copy()
            data["record_id"] = f"concurrent_record_{i}"
            records.append(EnhancedRecord(**data))
        
        embedder = EnhancedChromaDBEmbeddings(
            collection_name="test",
            persist_directory="./test",
            multi_field_config=multi_field_config,
            concurrency=3
        )
        
        # Process records concurrently
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(embedder.create_multi_field_chunks_enhanced, record)
            )
            for record in records
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all processed successfully
        assert len(results) == 10
        assert all(len(chunks) > 0 for chunks in results)


# ========== FILE I/O TESTS ==========

class TestFileIO:
    """Test suite for file I/O operations."""
    
    def test_parquet_write_read(self, sample_record_data):
        """Test writing and reading EnhancedRecord to/from Parquet."""
        record = EnhancedRecord(**sample_record_data)
        
        # Add chunks with embeddings
        chunk1 = record.add_chunk("First chunk", chunk_type="content")
        chunk1.embedding = [0.1, 0.2, 0.3] * 1024  # 3072-dim embedding
        
        chunk2 = record.add_chunk("Title chunk", chunk_type="title", source_field="title")
        chunk2.embedding = [0.4, 0.5, 0.6] * 1024
        
        embedder = EnhancedChromaDBEmbeddings(
            collection_name="test",
            persist_directory="./test",
            dimensionality=3072
        )
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            embedder._write_record_to_parquet(record, tmp_path)
            
            # Verify file was created
            assert tmp_path.exists()
            
            # Read back and verify
            table = pq.read_table(tmp_path)
            assert table.num_rows == 2  # Two chunks
            
            # Check columns
            expected_columns = [
                "chunk_id", "document_id", "document_title", "chunk_index",
                "chunk_text", "chunk_type", "source_field", "embedding", "chunk_metadata"
            ]
            assert all(col in table.column_names for col in expected_columns)
            
            # Check metadata
            metadata = table.schema.metadata
            assert b'record_id' in metadata
            assert b'format_version' in metadata
            assert metadata[b'format_version'] == b'enhanced_record_v1'
            
            # Clean up
            tmp_path.unlink()


# ========== INTEGRATION TESTS ==========

class TestIntegration:
    """Integration tests for complete workflows."""
    
    async def test_end_to_end_processing(self, sample_record_data, multi_field_config):
        """Test complete end-to-end processing workflow."""
        # 1. Create record
        record = EnhancedRecord(**sample_record_data)
        
        # 2. Setup embedder with mocks
        with patch('buttermilk.data.vector.TextEmbeddingModel.from_pretrained') as mock_model_class, \
             patch('chromadb.PersistentClient') as mock_client_class:
            
            # Setup mocks
            mock_embedding_model = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 3072
            mock_embedding_model.get_embeddings_async = AsyncMock(return_value=[mock_embedding])
            mock_model_class.return_value = mock_embedding_model
            
            mock_collection = MagicMock()
            mock_collection.upsert = MagicMock()
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            embedder = EnhancedChromaDBEmbeddings(
                collection_name="integration_test",
                persist_directory="./test_chromadb",
                multi_field_config=multi_field_config
            )
            embedder._collection = mock_collection
            
            # 3. Process record
            with patch.object(embedder, 'collection', mock_collection):
                processed_record = await embedder.process_enhanced_record_async(record)
            
            # 4. Verify complete workflow
            assert processed_record is not None
            assert len(processed_record.chunks) > 0
            assert processed_record.has_embeddings
            
            # Verify all expected chunk types are present
            chunk_types = processed_record.chunk_types
            expected_types = {"content", "title", "abstract", "keywords"}
            assert expected_types.issubset(chunk_types)
            
            # Verify embeddings were generated
            mock_embedding_model.get_embeddings_async.assert_called()
            
            # Verify data was stored
            mock_collection.upsert.assert_called()
            
    def test_legacy_compatibility_workflow(self, sample_legacy_input_document):
        """Test complete workflow with legacy InputDocument compatibility."""
        # 1. Start with legacy InputDocument
        input_doc = sample_legacy_input_document
        
        # 2. Convert to EnhancedRecord
        enhanced_record = CompatibilityLayer.normalize_record(input_doc)
        
        # 3. Verify conversion preserved data
        assert enhanced_record.record_id == input_doc.record_id
        assert enhanced_record.title == input_doc.title
        assert len(enhanced_record.chunks) == len(input_doc.chunks)
        
        # 4. Convert back to InputDocument
        converted_back = enhanced_record.as_input_document()
        
        # 5. Verify round-trip conversion
        assert converted_back.record_id == input_doc.record_id
        assert converted_back.title == input_doc.title
        assert len(converted_back.chunks) == len(input_doc.chunks)
        
        # Verify chunk data is preserved
        for orig_chunk, conv_chunk in zip(input_doc.chunks, converted_back.chunks):
            assert orig_chunk.chunk_id == conv_chunk.chunk_id
            assert orig_chunk.chunk_text == conv_chunk.chunk_text
            assert orig_chunk.embedding == conv_chunk.embedding


# ========== EXAMPLE TEST RUNNER ==========

if __name__ == "__main__":
    # Example of how to run specific test cases
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_create_enhanced_record"
    ])