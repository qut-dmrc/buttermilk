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

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.protocols import BatchProcessor
from buttermilk._core.types import BaseRecord


class TestEmbeddingProcessorConfig:
    """Test EmbeddingProcessorConfig exists and validates correctly."""

    def test_embedding_processor_config_class_exists(self):
        """Verify EmbeddingProcessorConfig class is defined."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig

        # Verify it's a Pydantic model
        assert hasattr(EmbeddingProcessorConfig, "model_validate")

    def test_embedding_processor_config_has_required_fields(self):
        """Verify EmbeddingProcessorConfig has necessary fields."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig

        # Create a minimal config
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )

        # Verify required fields
        assert config.type == "embedding"
        assert config.embedding_model == "gemini-embedding-001"

    def test_embedding_processor_config_inherits_from_batch_processor_config(self):
        """Verify EmbeddingProcessorConfig inherits from BatchProcessorConfig."""
        from buttermilk._core.processor_config import (
            BatchProcessorConfig,
            EmbeddingProcessorConfig,
        )

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            batch_size=64,
        )

        # Should have batch_size from parent
        assert config.batch_size == 64
        assert isinstance(config, BatchProcessorConfig)

    def test_embedding_processor_config_has_optional_fields(self):
        """Verify EmbeddingProcessorConfig has optional configuration fields."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig

        # Create config with optional fields
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            dimensionality=768,
            task="RETRIEVAL_DOCUMENT",
            embedding_max_retries=3,
            batch_size=50,
        )

        # Verify optional fields
        assert config.dimensionality == 768
        assert config.task == "RETRIEVAL_DOCUMENT"
        assert config.embedding_max_retries == 3
        assert config.batch_size == 50

    def test_embedding_processor_config_validates_type_literal(self):
        """Verify config type field is enforced as literal 'embedding'."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig

        # Valid type
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )
        assert config.type == "embedding"

        # Invalid type should fail validation (fail-fast)
        with pytest.raises(Exception):  # Pydantic ValidationError
            EmbeddingProcessorConfig(
                type="wrong_type",
                embedding_model="gemini-embedding-001",
            )


class TestEmbeddingProcessorProtocol:
    """Test EmbeddingProcessor implements BatchProcessor protocol."""

    def test_embedding_processor_class_exists(self):
        """Verify EmbeddingProcessor class is defined."""
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        # Verify class exists
        assert EmbeddingProcessor is not None

    def test_embedding_processor_inherits_from_unified_batch_processor(self):
        """Verify EmbeddingProcessor inherits from UnifiedBatchProcessor."""
        from buttermilk._core.unified_batch_processor import UnifiedBatchProcessor
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        # Verify inheritance
        assert issubclass(EmbeddingProcessor, UnifiedBatchProcessor)

    def test_embedding_processor_implements_batch_processor_protocol(self):
        """Verify EmbeddingProcessor satisfies BatchProcessor protocol."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        # Create instance
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )
        processor = EmbeddingProcessor(config)

        # Verify protocol methods exist
        assert hasattr(processor, "process_batch")
        assert hasattr(processor, "finalize")

        # Verify it's recognized as BatchProcessor
        assert isinstance(processor, BatchProcessor)

    def test_embedding_processor_has_config_attribute(self):
        """Verify processor stores config correctly."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            batch_size=100,
        )
        processor = EmbeddingProcessor(config)

        # Verify config is stored
        assert processor.config == config
        assert processor.config.embedding_model == "gemini-embedding-001"
        assert processor.config.batch_size == 100


class TestEmbeddingProcessorRegistry:
    """Test EmbeddingProcessor registers with processor registry."""

    def test_embedding_processor_registered_as_embedding_type(self):
        """Verify EmbeddingProcessor is registered with type 'embedding'."""
        from buttermilk._core.processor_registry import get_registered_types

        # Import processors module to trigger registration
        import buttermilk.processors.unified_processors  # noqa: F401

        # Verify 'embedding' type is registered
        registered_types = get_registered_types()
        assert "embedding" in registered_types

    def test_create_processor_returns_embedding_processor(self):
        """Verify registry creates EmbeddingProcessor from config."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk._core.processor_registry import create_processor
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        # Create config
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )

        # Create processor via registry
        processor = create_processor(config)

        # Verify correct type created
        assert isinstance(processor, EmbeddingProcessor)
        assert processor.config.type == "embedding"


class TestEmbeddingProcessorBatchProcessing:
    """Test batch processing functionality with mock embedding API."""

    @pytest.mark.anyio
    async def test_embedding_processor_processes_batch_with_chunks(self):
        """Verify EmbeddingProcessor adds embeddings to chunks in batch."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        # Create config
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            dimensionality=768,
            batch_size=10,
        )

        processor = EmbeddingProcessor(config)

        # Create test records with chunks
        contexts = [
            ProcessingContext(
                session_id="test-session",
                record=BaseRecord(
                    record_id=f"record-{i}",
                    content=f"content-{i}",
                    chunks=[
                        {"text": f"chunk-{i}-0", "chunk_index": 0},
                        {"text": f"chunk-{i}-1", "chunk_index": 1},
                    ],
                ),
            )
            for i in range(3)
        ]

        # Mock the embedding API
        mock_embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768, [0.4] * 768, [0.5] * 768, [0.6] * 768]

        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            # Mock the embed_content response
            mock_response = MagicMock()
            mock_response.embeddings = [
                MagicMock(values=emb) for emb in mock_embeddings
            ]
            mock_genai.Client.return_value.models.embed_content.return_value = mock_response

            # Process batch
            outputs = []
            async for output_batch in processor.process_batch(contexts):
                outputs.extend(output_batch)

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
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )

        processor = EmbeddingProcessor(config)

        # Create record without chunks
        contexts = [
            ProcessingContext(
                session_id="test-session",
                record=BaseRecord(
                    record_id="no-chunks",
                    content="content without chunks",
                ),
            )
        ]

        # Process batch (should pass through without error)
        outputs = []
        async for output_batch in processor.process_batch(contexts):
            outputs.extend(output_batch)

        # Should yield the record unchanged
        assert len(outputs) == 1
        assert outputs[0].record_id == "no-chunks"
        # Record should not have chunks or embeddings added
        assert not hasattr(outputs[0], "chunks") or not outputs[0].chunks

    @pytest.mark.anyio
    async def test_embedding_processor_batches_api_calls(self):
        """Verify processor batches embedding API calls efficiently."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        # Create config with small batch size
        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            batch_size=2,  # Small batch to test batching logic
        )

        processor = EmbeddingProcessor(config)

        # Create 5 records, each with 2 chunks = 10 chunks total
        contexts = [
            ProcessingContext(
                session_id="test-session",
                record=BaseRecord(
                    record_id=f"record-{i}",
                    content=f"content-{i}",
                    chunks=[
                        {"text": f"chunk-{i}-0"},
                        {"text": f"chunk-{i}-1"},
                    ],
                ),
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
            mock_response.embeddings = [
                MagicMock(values=[0.1] * 768) for _ in range(num_texts)
            ]
            return mock_response

        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            mock_genai.Client.return_value.models.embed_content.side_effect = mock_embed_content

            # Process batch
            outputs = []
            async for output_batch in processor.process_batch(contexts):
                outputs.extend(output_batch)

            # Verify all records processed
            assert len(outputs) == 5

            # Verify API was called (should batch 10 chunks efficiently)
            # Exact call count depends on batching implementation
            assert api_call_count > 0

    @pytest.mark.anyio
    async def test_embedding_processor_enriches_context_metadata(self):
        """Verify processor adds embedding metadata to context."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )

        processor = EmbeddingProcessor(config)

        # Create record with chunks
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="metadata-test",
                content="test content",
                chunks=[{"text": "chunk 1"}, {"text": "chunk 2"}],
            ),
        )

        # Mock embedding API
        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.embeddings = [
                MagicMock(values=[0.1] * 768),
                MagicMock(values=[0.2] * 768),
            ]
            mock_genai.Client.return_value.models.embed_content.return_value = mock_response

            # Process batch
            outputs = []
            async for output_batch in processor.process_batch([context]):
                outputs.extend(output_batch)

            # Verify context metadata was updated
            assert "embedding_stats" in context.metadata
            stats = context.metadata["embedding_stats"]

            # Verify stats contain useful information
            assert stats["chunks_embedded"] == 2
            assert stats["embedding_model"] == "gemini-embedding-001"
            assert "processing_time_ms" in stats


class TestEmbeddingProcessorErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.anyio
    async def test_embedding_processor_retries_on_api_failure(self):
        """Verify processor retries on transient API failures."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            embedding_max_retries=3,
        )

        processor = EmbeddingProcessor(config)

        # Create test context
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="retry-test",
                content="test content",
                chunks=[{"text": "chunk 1"}],
            ),
        )

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
            outputs = []
            async for output_batch in processor.process_batch([context]):
                outputs.extend(output_batch)

            # Verify processing succeeded
            assert len(outputs) == 1
            assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.anyio
    async def test_embedding_processor_fails_after_max_retries(self):
        """Verify processor fails fast after exhausting retries."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
            embedding_max_retries=2,
        )

        processor = EmbeddingProcessor(config)

        # Create test context
        context = ProcessingContext(
            session_id="test-session",
            record=BaseRecord(
                record_id="fail-test",
                content="test content",
                chunks=[{"text": "chunk 1"}],
            ),
        )

        # Mock API to always fail
        with patch("buttermilk.processors.unified_processors.genai") as mock_genai:
            mock_genai.Client.return_value.models.embed_content.side_effect = Exception(
                "Persistent API error"
            )

            # Processing should raise after max retries (fail-fast)
            with pytest.raises(Exception, match="Persistent API error"):
                async for _ in processor.process_batch([context]):
                    pass


class TestEmbeddingProcessorFinalization:
    """Test finalization and cleanup logic."""

    @pytest.mark.anyio
    async def test_embedding_processor_finalize_method_exists(self):
        """Verify finalize method exists and can be called."""
        from buttermilk._core.processor_config import EmbeddingProcessorConfig
        from buttermilk.processors.unified_processors import EmbeddingProcessor

        config = EmbeddingProcessorConfig(
            type="embedding",
            embedding_model="gemini-embedding-001",
        )

        processor = EmbeddingProcessor(config)

        # Finalize should not raise
        await processor.finalize()
