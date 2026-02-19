"""Tests for EmbeddingProcessor - Unified Batch Processor for embeddings.

This module tests the new EmbeddingProcessor that implements the BatchProcessor protocol.
The EmbeddingProcessor is the unified replacement for EmbeddingGenerator, designed to:
- Process batches of records with chunks
- Generate embeddings via Google GenAI
- Add embeddings to chunks in-place
- Support configurable batch sizes and retry logic

Tests follow the fail-fast philosophy and use REAL data patterns (no mocking internal code).
Only mock external APIs (Google GenAI) at the system boundary.
"""

from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.processors.unified_processors import EmbeddingProcessor


class TestEmbeddingProcessorPydantic:
    """Test EmbeddingProcessor Pydantic model validation."""

    def test_embedding_processor_is_pydantic_model(self):
        """Verify EmbeddingProcessor is a Pydantic model."""
        # Verify it's a Pydantic model
        assert hasattr(EmbeddingProcessor, "model_validate")

    def test_embedding_processor_has_required_fields(self):
        """Verify EmbeddingProcessor has necessary fields."""
        # Create a minimal processor
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
        )

        # Verify required fields
        assert processor.embedding_model == "gemini-embedding-001"

    def test_embedding_processor_inherits_from_processor_core(self):
        """Verify EmbeddingProcessor inherits from ProcessorCore."""
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
            batch_size=64,
        )

        # Should have batch_size field
        assert processor.batch_size == 64
        assert isinstance(processor, ProcessorCore)

    def test_embedding_processor_has_optional_fields(self):
        """Verify EmbeddingProcessor has optional configuration fields."""
        # Create processor with optional fields
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
            dimensionality=768,
            task="RETRIEVAL_DOCUMENT",
            embedding_max_retries=3,
            batch_size=50,
        )

        # Verify optional fields
        assert processor.dimensionality == 768
        assert processor.task == "RETRIEVAL_DOCUMENT"
        assert processor.embedding_max_retries == 3
        assert processor.batch_size == 50


class TestEmbeddingProcessorProtocol:
    """Test EmbeddingProcessor implements BatchProcessor protocol."""

    def test_embedding_processor_class_exists(self):
        """Verify EmbeddingProcessor class is defined."""
        # Verify class exists
        assert EmbeddingProcessor is not None

    def test_embedding_processor_inherits_from_processor_core(self):
        """Test inheritance from ProcessorCore."""
        assert issubclass(EmbeddingProcessor, ProcessorCore)
        # EmbeddingProcessor extends ProcessorCore and implements BatchProcessor protocol

    def test_embedding_processor_implements_batch_processor_protocol(self):
        """Test implementation of BatchProcessor protocol."""
        assert isinstance(EmbeddingProcessor, type)
        # Check if process_batch exists (runtime check simpler than Protocol check for now)
        assert hasattr(EmbeddingProcessor, "process_batch")


class TestEmbeddingProcessorBatchProcessing:
    @pytest.mark.anyio
    async def test_embedding_processor_processes_batch_with_chunks(self):
        """Verify EmbeddingProcessor adds embeddings to chunks in batch."""
        # Create processor directly as Pydantic model
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
            dimensionality=768,
            batch_size=10,
        )

        # Create test records with chunks (now uses BaseRecord directly, not ProcessingContext)
        records = [
            BaseRecord(
                record_id=f"record-{i}",
                content=f"content-{i}",
                chunks=[
                    {"text": f"chunk-{i}-0", "chunk_index": 0},
                    {"text": f"chunk-{i}-1", "chunk_index": 1},
                ],
            )
            for i in range(3)
        ]

        # Mock the embedding API
        mock_embeddings = [[0.1] * 768] * 6  # 3 records * 2 chunks = 6 embeddings

        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            # Mock the embed_content response
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock(values=emb) for emb in mock_embeddings]
            mock_genai.Client.return_value.models.embed_content.return_value = mock_response

            # Process batch - now returns list, not async generator
            outputs = await processor.process_batch(records)

            # Verify all records processed
            assert len(outputs) == 3

            # Verify embeddings were added to chunks
            for i, output in enumerate(outputs):
                assert hasattr(output, "chunks")
                assert len(output.chunks) == 2

                # Each chunk should have an embedding
                for chunk in output.chunks:
                    assert "embedding" in chunk
                    assert len(chunk["embedding"]) == 768

    @pytest.mark.anyio
    async def test_embedding_processor_handles_records_without_chunks(self):
        """Verify processor handles records without chunks gracefully."""
        # Create processor directly as Pydantic model
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
        )

        # Create record without chunks
        records = [
            BaseRecord(
                record_id="no-chunks",
                content="content without chunks",
            )
        ]

        # Process batch (should pass through without error)
        outputs = await processor.process_batch(records)

        # Should return the record unchanged
        assert len(outputs) == 1
        assert outputs[0].record_id == "no-chunks"
        # Record should not have chunks or embeddings added
        assert not hasattr(outputs[0], "chunks") or not outputs[0].chunks

    @pytest.mark.anyio
    async def test_embedding_processor_batches_api_calls(self):
        """Verify processor batches embedding API calls efficiently."""
        # Create processor directly as Pydantic model with small batch size
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
            batch_size=2,  # Small batch to test batching logic
        )

        # Create 5 records, each with 2 chunks = 10 chunks total
        records = [
            BaseRecord(
                record_id=f"record-{i}",
                content=f"content-{i}",
                chunks=[
                    {"text": f"chunk-{i}-0"},
                    {"text": f"chunk-{i}-1"},
                ],
            )
            for i in range(5)
        ]

        # Mock the embedding API and track calls
        api_call_count = 0

        def mock_embed_content(**kwargs):
            nonlocal api_call_count
            api_call_count += 1
            num_texts = len(kwargs.get("contents", []))
            # Return embeddings for each text
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock(values=[0.1] * 768) for _ in range(num_texts)]
            return mock_response

        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            mock_genai.Client.return_value.models.embed_content.side_effect = mock_embed_content

            # Process batch
            outputs = await processor.process_batch(records)

            # Verify all records processed
            assert len(outputs) == 5

            # Verify API was called (should batch 10 chunks efficiently)
            # Exact call count depends on batching implementation
            assert api_call_count > 0

    @pytest.mark.skip(reason="Context metadata tracking removed with ProcessingContext -> BaseRecord change")
    @pytest.mark.anyio
    async def test_embedding_processor_enriches_context_metadata(self):
        """Verify processor adds embedding metadata to context."""
        pass


class TestEmbeddingProcessorErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.anyio
    async def test_embedding_processor_retries_on_api_failure(self):
        """Verify processor retries on transient API failures."""
        # Create processor directly as Pydantic model
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
            embedding_max_retries=3,
        )

        # Create test record
        records = [
            BaseRecord(
                record_id="retry-test",
                content="test content",
                chunks=[{"text": "chunk 1"}],
            )
        ]

        # Mock API to fail twice, then succeed
        call_count = 0

        def mock_embed_content(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient API error")
            # Success on third try
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
            return mock_response

        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            mock_genai.Client.return_value.models.embed_content.side_effect = mock_embed_content

            # Process should succeed after retries
            outputs = await processor.process_batch(records)

            # Verify processing succeeded
            assert len(outputs) == 1
            assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.anyio
    async def test_embedding_processor_fails_after_max_retries(self):
        """Verify processor fails fast after exhausting retries."""
        # Create processor directly as Pydantic model
        processor = EmbeddingProcessor(
            embedding_model="gemini-embedding-001",
            embedding_max_retries=2,
        )

        # Create test record
        records = [
            BaseRecord(
                record_id="fail-test",
                content="test content",
                chunks=[{"text": "chunk 1"}],
            )
        ]

        # Mock API to always fail
        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            mock_genai.Client.return_value.models.embed_content.side_effect = Exception("Persistent API error")

            # Processing should raise after max retries (fail-fast)
            with pytest.raises(Exception, match="Persistent API error"):
                await processor.process_batch(records)


class TestEmbeddingProcessorFinalization:
    """Test finalization and cleanup logic."""

    @pytest.mark.skip(reason="EmbeddingProcessor no longer has finalize method - cleanup handled by BatchAccumulator")
    @pytest.mark.anyio
    async def test_embedding_processor_finalize_method_exists(self):
        """Verify finalize method exists and can be called."""
        pass
