"""Tests for ChromaDBProcessor - Unified Processor for ChromaDB uploads.

This module tests the new ChromaDBProcessor that implements the Processor protocol.
The ChromaDBProcessor is the unified replacement for ChromaDBUploader, designed to:
- Process individual records with embedded chunks
- Upload embeddings to ChromaDB collection
- Handle remote storage with sync capabilities
- Support finalize_processing() for final sync

Tests follow the fail-fast philosophy and use REAL data patterns (no mocking internal code).
Only mock external services (ChromaDB) at the system boundary.
"""

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import Processor
from buttermilk._core.types import BaseRecord
from buttermilk.processors.unified_processors import ChromaDBProcessor


class TestChromaDBProcessorPydanticModel:
    """Test ChromaDBProcessor Pydantic model structure and validation."""

    def test_chromadb_processor_is_pydantic_model(self):
        """Verify ChromaDBProcessor is a Pydantic model."""
        # Verify it's a Pydantic model
        assert hasattr(ChromaDBProcessor, "model_validate")

    def test_chromadb_processor_has_required_fields(self):
        """Verify ChromaDBProcessor has necessary fields."""
        # Create a minimal processor
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Verify required fields
        assert processor.collection_name == "test_collection"
        assert processor.persist_directory == "/tmp/chromadb_test"

    def test_chromadb_processor_has_optional_fields(self):
        """Verify ChromaDBProcessor has optional configuration fields."""
        # Create processor with optional fields
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
            sync_batch_size=100,
            sync_interval_minutes=5,
            disable_auto_sync=True,
            upsert_batch_size=500,
        )

        # Verify optional fields
        assert processor.sync_batch_size == 100
        assert processor.sync_interval_minutes == 5
        assert processor.disable_auto_sync is True
        assert processor.upsert_batch_size == 500

    def test_chromadb_processor_has_default_values(self):
        """Verify ChromaDBProcessor has correct default values for optional fields."""
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Verify defaults
        assert processor.sync_batch_size == 50
        assert processor.sync_interval_minutes == 10
        assert processor.disable_auto_sync is False
        assert processor.upsert_batch_size == 1000


class TestChromaDBProcessorProtocol:
    """Test ChromaDBProcessor implements Processor protocol."""

    def test_chromadb_processor_class_exists(self):
        """Verify ChromaDBProcessor class is defined."""
        # Verify class exists
        assert ChromaDBProcessor is not None

    def test_chromadb_processor_inherits_from_unified_processor(self):
        """Verify ChromaDBProcessor inherits from UnifiedProcessor."""
        from buttermilk._core.unified_processor import UnifiedProcessor

        # Verify inheritance
        assert issubclass(ChromaDBProcessor, UnifiedProcessor)

    def test_chromadb_processor_implements_processor_protocol(self):
        """Verify ChromaDBProcessor satisfies Processor protocol."""
        # Create instance directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Verify protocol methods exist
        assert hasattr(processor, "process")
        assert hasattr(processor, "finalize_processing")

        # Verify it's recognized as Processor
        assert isinstance(processor, Processor)

    def test_chromadb_processor_stores_fields_correctly(self):
        """Verify processor stores fields correctly."""
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
            sync_batch_size=100,
        )

        # Verify fields are stored
        assert processor.collection_name == "test_collection"
        assert processor.persist_directory == "/tmp/chromadb_test"
        assert processor.sync_batch_size == 100


class TestChromaDBProcessorSingleRecordProcessing:
    """Test single record processing functionality with mock ChromaDB."""

    @pytest.mark.anyio
    async def test_chromadb_processor_processes_record_with_embedded_chunks(self):
        """Verify ChromaDBProcessor uploads record chunks to ChromaDB."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Create test record with embedded chunks
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="test-record-1",
                content="test content",
                chunks=[
                    {
                        "chunk_id": "chunk-1-0",
                        "chunk_text": "chunk 1 text",
                        "chunk_index": 0,
                        "document_id": "test-record-1",
                        "document_title": "test document",
                        "embedding": [0.1] * 768,
                        "metadata": {"content_type": "text/plain"},
                    },
                    {
                        "chunk_id": "chunk-1-1",
                        "chunk_text": "chunk 2 text",
                        "chunk_index": 1,
                        "document_id": "test-record-1",
                        "document_title": "test document",
                        "embedding": [0.2] * 768,
                        "metadata": {"content_type": "text/plain"},
                    },
                ],
            ),
        )

        # Mock ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.upsert = AsyncMock()

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("buttermilk.processors.unified_processors.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client

            # Process record
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            # Verify record was processed
            assert len(outputs) == 1
            output_record = outputs[0]

            # Verify record was returned (passthrough)
            assert output_record.record_id == "test-record-1"

            # Verify metadata was updated
            assert "chromadb_upload" in output_record.metadata
            upload_metadata = output_record.metadata["chromadb_upload"]
            assert upload_metadata["status"] == "uploaded"
            assert upload_metadata["chunks_uploaded"] == 2
            assert upload_metadata["collection"] == "test_collection"

    @pytest.mark.anyio
    async def test_chromadb_processor_handles_records_without_chunks(self):
        """Verify processor handles records without chunks gracefully."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Create record without chunks
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="no-chunks",
                content="content without chunks",
            ),
        )

        # Process record (should pass through without error)
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield the record unchanged
        assert len(outputs) == 1
        assert outputs[0].record_id == "no-chunks"

    @pytest.mark.anyio
    async def test_chromadb_processor_handles_chunks_without_embeddings(self):
        """Verify processor handles chunks without embeddings gracefully."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Create record with chunks but no embeddings
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="no-embeddings",
                content="test content",
                chunks=[
                    {"chunk_text": "chunk 1", "chunk_index": 0},
                    {"chunk_text": "chunk 2", "chunk_index": 1},
                ],
            ),
        )

        # Process record (should pass through without error)
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield the record unchanged (no upload happened)
        assert len(outputs) == 1
        assert outputs[0].record_id == "no-embeddings"

    @pytest.mark.anyio
    async def test_chromadb_processor_enriches_context_metadata(self):
        """Verify processor adds upload metadata to context."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Create record with embedded chunks
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="metadata-test",
                content="test content",
                chunks=[
                    {
                        "chunk_id": "chunk-1",
                        "chunk_text": "chunk 1",
                        "chunk_index": 0,
                        "document_id": "metadata-test",
                        "document_title": "test",
                        "embedding": [0.1] * 768,
                    }
                ],
            ),
        )

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.upsert = AsyncMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("buttermilk.processors.unified_processors.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client

            # Process record
            outputs = []
            async for output in processor.process(context):
                outputs.append(output)

            # Verify context metadata was updated
            assert "chromadb_stats" in context.metadata
            stats = context.metadata["chromadb_stats"]

            # Verify stats contain useful information
            assert stats["chunks_uploaded"] == 1
            assert stats["collection_name"] == "test_collection"
            assert "processing_time_ms" in stats


class TestChromaDBProcessorRemoteStorage:
    """Test remote storage and sync functionality."""

    @pytest.mark.skip(reason="BM singleton cannot be mocked in tests due to buttermilk lazy loading")
    @pytest.mark.anyio
    async def test_chromadb_processor_handles_remote_storage_path(self):
        """Verify processor handles remote storage paths (gs://, s3://, etc)."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="gs://my-bucket/chromadb",
        )

        # Create test record
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="remote-test",
                content="test content",
                chunks=[
                    {
                        "chunk_id": "chunk-1",
                        "chunk_text": "chunk 1",
                        "chunk_index": 0,
                        "document_id": "remote-test",
                        "document_title": "test",
                        "embedding": [0.1] * 768,
                    }
                ],
            ),
        )

        # Mock ChromaDB and storage utilities
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.upsert = AsyncMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("buttermilk.processors.unified_processors.chromadb") as mock_chromadb:
            with patch("buttermilk.processors.unified_processors.bm") as mock_bm:
                mock_chromadb.PersistentClient.return_value = mock_client

                # Mock session_info for cache path generation
                mock_session = MagicMock()
                mock_session.generate_cache_key.return_value = "test_cache_key"
                mock_session.get_chromadb_cache_dir.return_value = MagicMock(
                    __truediv__=lambda self, x: MagicMock(
                        mkdir=MagicMock(), exists=MagicMock(return_value=True)
                    )
                )
                mock_bm.session_info = mock_session

                # Process record
                outputs = []
                async for output in processor.process(context):
                    outputs.append(output)

                # Verify processing succeeded
                assert len(outputs) == 1


class TestChromaDBProcessorFinalization:
    """Test finalization and sync logic."""

    @pytest.mark.anyio
    async def test_chromadb_processor_finalize_processing_method_exists(self):
        """Verify finalize_processing method exists and can be called."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="/tmp/chromadb_test",
        )

        # Finalize should not raise
        result = await processor.finalize_processing()

        # Should return True on success
        assert result is True

    @pytest.mark.skip(reason="BM singleton cannot be mocked in tests due to buttermilk lazy loading")
    @pytest.mark.anyio
    async def test_chromadb_processor_finalize_syncs_to_remote(self):
        """Verify finalize_processing syncs to remote storage."""
        # Create processor directly as Pydantic model
        processor = ChromaDBProcessor(
            collection_name="test_collection",
            persist_directory="gs://my-bucket/chromadb",
        )

        # Mock upload utility
        with patch(
            "buttermilk.processors.unified_processors.upload_chromadb_cache"
        ) as mock_upload:
            mock_upload.return_value = AsyncMock()

            # Mock session_info for cache path
            with patch("buttermilk.processors.unified_processors.bm") as mock_bm:
                mock_session = MagicMock()
                mock_session.generate_cache_key.return_value = "test_cache_key"
                mock_cache_path = MagicMock()
                mock_cache_path.exists.return_value = True
                mock_session.get_chromadb_cache_dir.return_value = MagicMock(
                    __truediv__=lambda self, x: mock_cache_path
                )
                mock_bm.session_info = mock_session

                # Initialize remote path by processing a record first
                processor._original_remote_path = "gs://my-bucket/chromadb"

                # Finalize
                result = await processor.finalize_processing()

                # Verify sync was called
                assert result is True
                # Upload should have been called with local cache and remote path
                mock_upload.assert_called_once()
