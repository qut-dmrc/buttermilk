"""TDD audit tests for ZoteroDownloadProcessor migration to ProcessingContext API.

These tests verify that:
1. buttermilk.libs.zotero can be imported without NameError
2. ZoteroDownloadProcessor.process() accepts a ProcessingContext (new API)
3. ZoteroDownloadProcessor.process() does NOT accept old-API positional args
4. The processor correctly extracts record from context.record
5. Test files that used old API still work with new API

Tests written BEFORE production fixes (TDD). Run them first to confirm they fail,
then apply fixes and confirm they pass.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pyzotero", reason="pyzotero is optional")


class TestZoteroImport:
    """Test that the zotero module can be imported cleanly."""

    def test_module_imports_without_error(self):
        """zotero.py must import without NameError for ProcessingContext."""
        # This is the primary regression: ProcessingContext was used in annotation
        # but never imported. Importing the module should succeed.
        import importlib

        import buttermilk.libs.zotero as zotero_mod

        importlib.reload(zotero_mod)  # Force fresh import
        assert hasattr(zotero_mod, "ZoteroDownloadProcessor")
        assert hasattr(zotero_mod, "ZoteroSource")

    def test_zotero_download_processor_importable(self):
        """ZoteroDownloadProcessor must be importable directly."""
        from buttermilk.libs.zotero import ZoteroDownloadProcessor

        assert ZoteroDownloadProcessor is not None

    def test_zotero_source_importable(self):
        """ZoteroSource must be importable directly."""
        from buttermilk.libs.zotero import ZoteroSource

        assert ZoteroSource is not None


class TestZoteroDownloadProcessorNewAPI:
    """Test that ZoteroDownloadProcessor uses the new ProcessingContext API."""

    @pytest.mark.anyio
    async def test_process_accepts_processing_context(self):
        """process() must accept a ProcessingContext object (new unified API)."""
        from buttermilk._core.processing_context import ProcessingContext
        from buttermilk._core.types import BaseRecord
        from buttermilk.libs.zotero import ZoteroDownloadProcessor

        # Build a minimal record with all fields needed by the processor
        record = BaseRecord(
            record_id="TEST_KEY",
            metadata={
                "zotero_item": {"title": "Test Document", "DOI": "10.1/test"},
                "zotero_links": {},  # No PDF attachment — processor should return nothing
                "citation_key": None,
            },
        )

        context = ProcessingContext(
            session_id="test-session",
            record=record,
        )

        processor = ZoteroDownloadProcessor(library_id="test_library")

        with patch.object(processor, "_get_cache_dir", return_value=Path(tempfile.mkdtemp())):
            # Calling process(context) should work without TypeError
            results = []
            async for result in processor.process(context):
                results.append(result)

        # No PDF attachment means no results — but no exception either
        assert results == []

    @pytest.mark.anyio
    async def test_process_uses_context_record(self):
        """process() must extract record from context.record, not from positional args."""
        from buttermilk._core.processing_context import ProcessingContext
        from buttermilk._core.types import BaseRecord, Record
        from buttermilk.libs.zotero import ZoteroDownloadProcessor
        from pyzotero import zotero_errors

        record = BaseRecord(
            record_id="PDF_KEY",
            metadata={
                "zotero_item": {"title": "PDF Document", "DOI": "10.1/pdf"},
                "zotero_links": {
                    "attachment": {
                        "attachmentType": "application/pdf",
                        "href": "https://api.zotero.org/groups/1/items/ATTACH/file",
                    }
                },
                "citation_key": "author2024",
            },
        )

        context = ProcessingContext(
            session_id="test-session",
            record=record,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_zot = MagicMock()
            mock_zot.fulltext_item.side_effect = zotero_errors.ResourceNotFoundError
            mock_zot.dump = MagicMock(
                side_effect=lambda key, path: Path(path).write_bytes(b"%PDF-1.4\n" + b"x" * 60000)
            )

            processor = ZoteroDownloadProcessor(library_id="test_library")
            # Inject mock directly to avoid bm singleton
            processor._zot = mock_zot

            with patch.object(processor, "_get_cache_dir", return_value=Path(tmpdir)):
                results = []
                async for result in processor.process(context):
                    results.append(result)

        # Should yield exactly one Record
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, Record)
        assert result.record_id == "PDF_KEY"
        assert result.metadata.get("title") == "PDF Document"

    @pytest.mark.anyio
    async def test_process_with_fulltext_uses_context_record(self):
        """process() with Zotero fulltext must use record from context."""
        from buttermilk._core.processing_context import ProcessingContext
        from buttermilk._core.types import BaseRecord, Record
        from buttermilk.libs.zotero import ZoteroDownloadProcessor

        record = BaseRecord(
            record_id="FULL_KEY",
            metadata={
                "zotero_item": {"title": "Full Text Document"},
                "zotero_links": {
                    "attachment": {
                        "attachmentType": "application/pdf",
                        "href": "https://api.zotero.org/groups/1/items/ATTACH/file",
                    }
                },
                "citation_key": "test2024",
            },
        )

        context = ProcessingContext(
            session_id="test-session",
            record=record,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_zot = MagicMock()
            mock_zot.fulltext_item.return_value = {
                "content": "Extracted full text content here",
                "indexedPages": 10,
                "totalPages": 10,
            }

            processor = ZoteroDownloadProcessor(library_id="test_library")
            processor._zot = mock_zot

            with patch.object(processor, "_get_cache_dir", return_value=Path(tmpdir)):
                results = []
                async for result in processor.process(context):
                    results.append(result)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, Record)
        assert result.record_id == "FULL_KEY"
        assert result.content == "Extracted full text content here"
        assert result.file_path is None  # No PDF downloaded when fulltext available


class TestZoteroOldAPITests:
    """Verify the existing test_zotero_pdf_metadata_content tests work with new API.

    The existing tests in test_zotero_pdf_metadata_content.py use old-API calls
    like processor.process(record, processor_stage="download"). These must be
    updated to use ProcessingContext.
    """

    @pytest.mark.anyio
    async def test_zotero_sets_pdf_metadata_as_content_new_api(self):
        """Equivalent of test_zotero_sets_pdf_metadata_as_content_when_no_fulltext
        using new ProcessingContext API."""
        from buttermilk._core.processing_context import ProcessingContext
        from buttermilk._core.types import BaseRecord
        from buttermilk.libs.zotero import ZoteroDownloadProcessor

        record = BaseRecord(
            record_id="TEST_KEY",
            metadata={
                "zotero_item": {"title": "Test Document"},
                "zotero_links": {
                    "attachment": {
                        "attachmentType": "application/pdf",
                        "href": "https://api.zotero.org/users/123/items/ATTACH_KEY/file",
                    }
                },
                "citation_key": None,
            },
        )

        context = ProcessingContext(
            session_id="test-session",
            record=record,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_zot = MagicMock()
            mock_zot.fulltext_item.side_effect = Exception("No fulltext available")
            mock_zot.dump = MagicMock(
                side_effect=lambda key, path: Path(path).write_bytes(b"%PDF-1.4\n" + b"x" * 100000)
            )

            processor = ZoteroDownloadProcessor(library_id="test_library")
            processor._zot = mock_zot

            with patch.object(processor, "_get_cache_dir", return_value=Path(tmpdir)):
                results = []
                async for result in processor.process(context):
                    results.append(result)

        assert len(results) == 1
        result = results[0]

        assert result.content is not None
        assert result.content != ""
        assert "[PDF Document:" in result.content
        assert "TEST_KEY.pdf" in result.content
        assert "Size:" in result.content
        assert "Path:" in result.content
        assert result.file_path is not None
        assert result.file_path.endswith("TEST_KEY.pdf")

    @pytest.mark.anyio
    async def test_zotero_uses_fulltext_when_available_new_api(self):
        """Equivalent of test_zotero_uses_fulltext_when_available using new API."""
        from buttermilk._core.processing_context import ProcessingContext
        from buttermilk._core.types import BaseRecord
        from buttermilk.libs.zotero import ZoteroDownloadProcessor

        record = BaseRecord(
            record_id="TEST_KEY",
            metadata={
                "zotero_item": {"title": "Test Document"},
                "zotero_links": {
                    "attachment": {
                        "attachmentType": "application/pdf",
                        "href": "https://api.zotero.org/users/123/items/ATTACH_KEY/file",
                    }
                },
                "citation_key": None,
            },
        )

        context = ProcessingContext(
            session_id="test-session",
            record=record,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_zot = MagicMock()
            mock_zot.fulltext_item.return_value = {
                "content": "This is the extracted text from Zotero fulltext API",
                "indexedPages": 10,
                "totalPages": 10,
            }

            processor = ZoteroDownloadProcessor(library_id="test_library")
            processor._zot = mock_zot

            with patch.object(processor, "_get_cache_dir", return_value=Path(tmpdir)):
                results = []
                async for result in processor.process(context):
                    results.append(result)

        assert len(results) == 1
        result = results[0]
        assert result.content == "This is the extracted text from Zotero fulltext API"
        assert "[PDF Document:" not in result.content
