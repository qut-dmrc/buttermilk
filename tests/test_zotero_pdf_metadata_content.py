"""Test that ZoteroDownloadProcessor sets meaningful PDF metadata as content.

This test verifies the architectural change where ZoteroDownloadProcessor
sets PDF metadata as content when Zotero fulltext API doesn't provide text,
allowing PDFToTextProcessor to handle actual text extraction.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.anyio
async def test_zotero_sets_pdf_metadata_as_content_when_no_fulltext():
    """Test that meaningful PDF metadata is set as content when fulltext not available.

    This ensures:
    1. Record contract is satisfied (content is not None/empty)
    2. PDF metadata provides meaningful information
    3. Content indicates that text extraction is needed
    """
    from buttermilk._core.types import Record
    from buttermilk.libs.zotero import ZoteroDownloadProcessor

    # Create a test record
    test_record = Record(
        record_id="TEST_KEY",
        content="[Placeholder from fetch]",
        metadata={
            "title": "Test Document",
            "zotero_item": {"key": "TEST_KEY", "data": {"title": "Test Document"}},
            "zotero_links": {
                "attachment": {
                    "attachmentType": "application/pdf",
                    "href": "https://api.zotero.org/users/123/items/ATTACH_KEY/file",
                }
            },
        },
    )

    # Mock the Zotero API client at module level
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("buttermilk.libs.zotero.Zotero") as MockZotero:
            mock_zot = MagicMock()
            mock_zot.fulltext_item.side_effect = Exception("No fulltext available")
            mock_zot.dump = MagicMock(
                side_effect=lambda key, path: Path(path).write_bytes(
                    b"%PDF-1.4\n" + b"x" * 100000
                )
            )
            MockZotero.return_value = mock_zot

            downloader = ZoteroDownloadProcessor(
                library_id="test_library", save_dir=tmpdir
            )

            # Process the record
            results = []
            async for result in downloader.process(
                test_record, processor_stage="download"
            ):
                results.append(result)

            # Verify we got a result
            assert len(results) == 1
            result = results[0]

            # Verify content is not None/empty
            assert result.content is not None
            assert result.content != ""

            # Verify content contains meaningful PDF metadata
            assert "[PDF Document:" in result.content
            assert "TEST_KEY.pdf" in result.content
            assert "Size:" in result.content
            assert "bytes" in result.content
            assert "Path:" in result.content

            # Verify file_path is set correctly
            assert result.file_path.endswith("TEST_KEY.pdf")

            print(f"\n✅ Content set correctly: {result.content}")


@pytest.mark.anyio
async def test_zotero_uses_fulltext_when_available():
    """Test that Zotero fulltext API content is used when available.

    This ensures backward compatibility - if Zotero provides fulltext,
    we use it instead of PDF metadata.
    """
    from buttermilk._core.types import Record
    from buttermilk.libs.zotero import ZoteroDownloadProcessor

    test_record = Record(
        record_id="TEST_KEY",
        content="[Placeholder from fetch]",
        metadata={
            "title": "Test Document",
            "zotero_item": {"key": "TEST_KEY", "data": {"title": "Test Document"}},
            "zotero_links": {
                "attachment": {
                    "attachmentType": "application/pdf",
                    "href": "https://api.zotero.org/users/123/items/ATTACH_KEY/file",
                }
            },
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("buttermilk.libs.zotero.Zotero") as MockZotero:
            mock_zot = MagicMock()
            mock_zot.fulltext_item.return_value = {
                "content": "This is the extracted text from Zotero fulltext API",
                "indexedPages": 10,
                "totalPages": 10,
            }
            MockZotero.return_value = mock_zot

            downloader = ZoteroDownloadProcessor(
                library_id="test_library", save_dir=tmpdir
            )

            # Process the record
            results = []
            async for result in downloader.process(
                test_record, processor_stage="download"
            ):
                results.append(result)

            assert len(results) == 1
            result = results[0]

            # Verify Zotero fulltext was used
            assert (
                result.content == "This is the extracted text from Zotero fulltext API"
            )
            assert "[PDF Document:" not in result.content

            print(f"\n✅ Zotero fulltext used: {result.content[:50]}...")


@pytest.mark.anyio
async def test_pdftotext_processor_replaces_metadata_content():
    """Test that PDFToTextProcessor correctly replaces PDF metadata with extracted text.

    This verifies the full workflow:
    1. ZoteroDownloadProcessor sets PDF metadata as content
    2. PDFToTextProcessor replaces it with actual extracted text
    """
    from buttermilk._core.types import Record
    from buttermilk.processors.bash import PDFToTextProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test PDF with some text
        pdf_path = Path(tmpdir) / "test.pdf"

        # For this test, we'll create a simple text file that pdftotext would read
        # (In reality, pdftotext would extract from PDF, but for testing we simulate)
        pdf_path.write_bytes(b"%PDF-1.4\n" + b"Test content" * 10000)

        # Create a record with PDF metadata as content (as set by ZoteroDownloadProcessor)
        record = Record(
            record_id="TEST_DOC",
            content=f"[PDF Document: test.pdf, Size: {pdf_path.stat().st_size:,} bytes, Path: {pdf_path}]",
            file_path=str(pdf_path),
            metadata={"title": "Test Document"},
        )

        # Verify initial content is PDF metadata
        assert "[PDF Document:" in record.content

        # Mock pdftotext command to return extracted text
        processor = PDFToTextProcessor()

        with patch("asyncio.create_subprocess_shell") as mock_subprocess:
            # Create an async mock process
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(
                return_value=(b"Extracted text from PDF", b"")
            )
            mock_process.returncode = 0

            # Make create_subprocess_shell return an awaitable that yields the mock process
            async def async_return_process():
                return mock_process

            mock_subprocess.return_value = async_return_process()

            # Process with PDFToTextProcessor
            results = []
            async for result in processor.process(record, processor_stage="pdftotext"):
                results.append(result)

            assert len(results) == 1
            result = results[0]

            # Verify content was replaced with extracted text
            assert result.content == "Extracted text from PDF"
            assert "[PDF Document:" not in result.content

            print(f"\n✅ PDF metadata replaced with extracted text: {result.content}")
