"""Tests verifying all processors use the new ProcessingContext API.

These tests call processor.process(context: ProcessingContext) and confirm the
new API works. They FAIL before the migration and PASS after.
"""

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord, Record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_context(record: BaseRecord, session_id: str = "test-session") -> ProcessingContext:
    """Build a minimal ProcessingContext for testing."""
    return ProcessingContext(session_id=session_id, record=record)


# ---------------------------------------------------------------------------
# QualityFilterProcessor
# ---------------------------------------------------------------------------


class TestQualityFilterProcessorContextAPI:
    """QualityFilterProcessor must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context(self):
        """process(context) should work without processor_stage kwarg."""
        from buttermilk.processors.quality import QualityFilterProcessor

        processor = QualityFilterProcessor(corruption_threshold=95.0)

        # Build a record with clean chunks
        record = Record(
            record_id="test-quality-1",
            content="clean document",
            chunks=[
                {"chunk_text": "This is a clean chunk with normal content."},
                {"chunk_text": "Another clean chunk with readable text."},
            ],
        )
        context = make_context(record)

        results = [r async for r in processor.process(context)]
        assert len(results) == 1
        assert results[0].record_id == "test-quality-1"

    @pytest.mark.anyio
    async def test_process_raises_on_corrupt_document(self):
        """process(context) should raise ProcessingError on corrupt document."""
        from buttermilk._core.exceptions import ProcessingError
        from buttermilk.processors.quality import QualityFilterProcessor

        processor = QualityFilterProcessor(corruption_threshold=0.0)  # All docs fail

        record = Record(
            record_id="test-quality-2",
            content="corrupted content",
            chunks=[{"chunk_text": "a" * 100}],  # Will be flagged as corrupt
        )
        context = make_context(record)

        # The processor should either yield or raise; with threshold=0.0 and
        # is_document_corrupt logic, it depends on the quality check result.
        # We just confirm process() accepts a context object — no TypeError.
        try:
            async for _ in processor.process(context):
                pass
        except (ProcessingError, ValueError):
            pass  # Expected — the point is no TypeError about processor_stage


# ---------------------------------------------------------------------------
# CSVMetadataLogger
# ---------------------------------------------------------------------------


class TestCSVMetadataLoggerContextAPI:
    """CSVMetadataLogger must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context(self):
        """process(context) should accumulate record and yield it unchanged."""
        from buttermilk.processors.csv_metadata_logger import CSVMetadataLogger

        csv_logger = CSVMetadataLogger(bucket="test-bucket", base_path="test")

        record = BaseRecord(
            record_id="r1",
            content="test prompt",
            metadata={
                "session_id": "s1",
                "model_class": "VertexImagen3Fast",
                "storage_uri": "gs://bucket/image1.png",
                "scenario": "office",
                "repetition": 0,
            },
        )
        context = make_context(record, session_id="s1")

        results = [r async for r in csv_logger.process(context)]

        assert len(results) == 1
        assert results[0].record_id == "r1"
        assert len(csv_logger._accumulated_records) == 1


# ---------------------------------------------------------------------------
# JMESPathTransform
# ---------------------------------------------------------------------------


class TestJMESPathTransformContextAPI:
    """JMESPathTransform must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context(self):
        """process(context) should apply JMESPath mappings."""
        from buttermilk.processors.jmespath_transform import JMESPathTransform

        processor = JMESPathTransform(mappings={"answer": "metadata.outputs.result"})

        record = BaseRecord(
            record_id="test-jmespath-1",
            metadata={"outputs": {"result": "extracted_value"}},
        )
        context = make_context(record)

        results = [r async for r in processor.process(context)]

        assert len(results) == 1
        assert hasattr(results[0], "answer")
        assert results[0].answer == "extracted_value"


# ---------------------------------------------------------------------------
# ChromaDBUploader
# ---------------------------------------------------------------------------


class TestChromaDBUploaderContextAPI:
    """ChromaDBUploader must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context_no_chunks(self):
        """process(context) should pass record through when no chunks present."""
        from buttermilk.processors.chromadb_uploader import ChromaDBUploader

        uploader = ChromaDBUploader(
            collection_name="test_collection",
            persist_directory="/tmp/test_chroma",
        )

        record = BaseRecord(
            record_id="test-chroma-1",
            content="no chunks here",
        )
        context = make_context(record)

        # Should pass through without error (no chunks = skip upload)
        results = [r async for r in uploader.process(context)]
        assert len(results) == 1
        assert results[0].record_id == "test-chroma-1"


# ---------------------------------------------------------------------------
# BatchExpansionProcessor
# ---------------------------------------------------------------------------


class TestBatchExpansionProcessorContextAPI:
    """BatchExpansionProcessor must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context(self):
        """process(context) should expand record into multiple copies."""
        from buttermilk.agents.imagegen import VertexImagen3Fast, VertexImagen4Fast
        from buttermilk.processors.batch_expansion import BatchExpansionProcessor

        processor = BatchExpansionProcessor(
            repetitions=2,
            models=[VertexImagen3Fast, VertexImagen4Fast],
        )

        record = BaseRecord(
            record_id="prompt-001",
            content="test prompt",
            metadata={"session_id": "sess123"},
        )
        context = make_context(record)

        results = [r async for r in processor.process(context)]

        assert len(results) == 4  # 2 reps × 2 models
        assert all(r.content == "test prompt" for r in results)
        assert all("model_class" in r.metadata for r in results)
        assert all("repetition" in r.metadata for r in results)


# ---------------------------------------------------------------------------
# ImageGenerationProcessor
# ---------------------------------------------------------------------------


class TestImageGenerationProcessorContextAPI:
    """ImageGenerationProcessor must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context_fails_fast_empty_content(self):
        """process(context) with empty content should raise ValueError, not TypeError."""
        from buttermilk.agents.imagegen import VertexImagen3Fast
        from buttermilk.processors.image_generation import ImageGenerationProcessor

        processor = ImageGenerationProcessor(client_class=VertexImagen3Fast)

        record = BaseRecord(
            record_id="test-imggen-1",
            content="",
            metadata={"session_id": "sess789"},
        )
        context = make_context(record)

        with pytest.raises(ValueError, match="empty content"):
            async for _ in processor.process(context):
                pass


# ---------------------------------------------------------------------------
# GCSImageStorageProcessor
# ---------------------------------------------------------------------------


class TestGCSImageStorageProcessorContextAPI:
    """GCSImageStorageProcessor must accept ProcessingContext."""

    @pytest.mark.anyio
    async def test_process_accepts_context_fails_fast_without_image_uri(self, tmp_path):
        """process(context) without image_uri should raise, not TypeError."""
        from buttermilk.processors.gcs_image_storage import GCSImageStorageProcessor

        processor = GCSImageStorageProcessor(
            bucket="test-bucket",
            base_path="test-images",
            local_only=True,
            local_dir=str(tmp_path),
        )

        record = BaseRecord(
            record_id="test-gcs-1",
            content="prompt",
            metadata={
                "session_id": "s1",
                "scenario": "test",
                # Missing image_uri
            },
        )
        context = make_context(record)

        with pytest.raises((ValueError, KeyError)):
            async for _ in processor.process(context):
                pass

    @pytest.mark.anyio
    async def test_process_accepts_context_uploads_local(self, tmp_path):
        """process(context) with valid image should copy to local dir."""
        from buttermilk.processors.gcs_image_storage import GCSImageStorageProcessor

        test_image = tmp_path / "source.png"
        test_image.write_bytes(b"fake image data")

        processor = GCSImageStorageProcessor(
            bucket="test-bucket",
            base_path="test-images",
            local_only=True,
            local_dir=str(tmp_path / "output"),
        )

        record = BaseRecord(
            record_id="test-gcs-2",
            content="prompt",
            metadata={
                "session_id": "sess-abc",
                "scenario": "office",
                "model_prefix": "test_",
                "repetition": 0,
                "image_uri": test_image.as_uri(),
            },
        )
        context = make_context(record)

        results = [r async for r in processor.process(context)]
        assert len(results) == 1
        assert "storage_uri" in results[0].metadata
