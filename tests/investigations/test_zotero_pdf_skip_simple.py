"""Simple test for Zotero PDF skip logic without BM dependency."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.types import Record


@pytest.mark.anyio
async def test_no_pdf_download_when_fulltext_exists():
    """Test that PDF is NOT downloaded when Zotero provides fulltext."""

    test_record = Record(
        record_id="TEST_KEY",
        content="[Placeholder from fetch]",
        metadata={
            "title": "Test Document with Fulltext",
            "zotero_item": {
                "key": "TEST_KEY",
                "title": "Test Document with Fulltext",
                "itemType": "journalArticle",
                "DOI": "10.1234/test",
            },
            "zotero_links": {
                "attachment": {
                    "attachmentType": "application/pdf",
                    "href": "https://api.zotero.org/users/123/items/ATTACH_KEY/file",
                }
            },
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock the BM singleton that ZoteroDownloadProcessor needs
        with patch("buttermilk._core.dmrc.get_bm") as mock_get_bm:
            # Create mock BM instance
            mock_bm = MagicMock()
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(exist_ok=True)
            mock_bm.session_info.get_cache_subdir.return_value = cache_dir
            mock_get_bm.return_value = mock_bm

            with patch("buttermilk.libs.zotero.zotero.Zotero") as MockZotero:
                mock_zot = MagicMock()

                # Zotero provides fulltext with 100% indexed pages
                mock_zot.fulltext_item.return_value = {
                    "content": "This is the complete text extracted by Zotero API",
                    "indexedPages": 10,
                    "totalPages": 10,
                }

                # Mock dump to track if it's called (it shouldn't be!)
                mock_zot.dump = MagicMock()

                MockZotero.return_value = mock_zot

                # Import after patching to use mocked BM
                from buttermilk.libs.zotero import ZoteroDownloadProcessor

<<<<<<< HEAD
                downloader = ZoteroDownloadProcessor(library_id="test_library", save_dir=tmpdir)

                # Process the record
                results = []
                async for result in downloader.process(test_record, processor_stage="download"):
=======
                downloader = ZoteroDownloadProcessor(
                    library_id="test_library", save_dir=tmpdir
                )

                # Process the record
                results = []
                async for result in downloader.process(
                    test_record, processor_stage="download"
                ):
>>>>>>> origin/stable
                    results.append(result)

                assert len(results) == 1
                result = results[0]

                # CRITICAL ASSERTIONS:
                # 1. Zotero fulltext was used
<<<<<<< HEAD
                assert result.content == "This is the complete text extracted by Zotero API"
=======
                assert (
                    result.content
                    == "This is the complete text extracted by Zotero API"
                )
>>>>>>> origin/stable

                # 2. PDF was NOT downloaded (dump not called)
                mock_zot.dump.assert_not_called()

                # 3. file_path should be None
<<<<<<< HEAD
                assert result.file_path is None, f"file_path should be None, got: {result.file_path}"
=======
                assert result.file_path is None, (
                    f"file_path should be None, got: {result.file_path}"
                )
>>>>>>> origin/stable

                # 4. No PDF metadata in content
                assert "[PDF Document:" not in result.content

                print("\n✅ SUCCESS: No PDF download when fulltext exists:")
                print(f"   - Content: {result.content[:50]}...")
                print(f"   - file_path: {result.file_path}")
                print(f"   - PDF download skipped: {not mock_zot.dump.called}")


@pytest.mark.anyio
async def test_pdf_downloads_when_no_fulltext():
    """Test that PDF IS downloaded when no fulltext available."""

    test_record = Record(
        record_id="TEST_KEY2",
        content="[Placeholder from fetch]",
        metadata={
            "title": "Test Document without Fulltext",
            "zotero_item": {
                "key": "TEST_KEY2",
                "title": "Test Document without Fulltext",
                "itemType": "journalArticle",
                "DOI": "10.1234/test2",
            },
            "zotero_links": {
                "attachment": {
                    "attachmentType": "application/pdf",
                    "href": "https://api.zotero.org/users/123/items/ATTACH_KEY2/file",
                }
            },
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock the BM singleton
        with patch("buttermilk._core.dmrc.get_bm") as mock_get_bm:
            mock_bm = MagicMock()
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(exist_ok=True)
            mock_bm.session_info.get_cache_subdir.return_value = cache_dir
            mock_get_bm.return_value = mock_bm

            with patch("buttermilk.libs.zotero.zotero.Zotero") as MockZotero:
                mock_zot = MagicMock()

                # No fulltext available from Zotero
                mock_zot.fulltext_item.side_effect = Exception("No fulltext available")

                # Mock dump to create a PDF file
<<<<<<< HEAD
                mock_zot.dump = MagicMock(side_effect=lambda key, path: Path(path).write_bytes(b"%PDF-1.4\n" + b"x" * 100000))
=======
                mock_zot.dump = MagicMock(
                    side_effect=lambda key, path: Path(path).write_bytes(
                        b"%PDF-1.4\n" + b"x" * 100000
                    )
                )
>>>>>>> origin/stable

                MockZotero.return_value = mock_zot

                from buttermilk.libs.zotero import ZoteroDownloadProcessor

<<<<<<< HEAD
                downloader = ZoteroDownloadProcessor(library_id="test_library", save_dir=tmpdir)

                # Process the record
                results = []
                async for result in downloader.process(test_record, processor_stage="download"):
=======
                downloader = ZoteroDownloadProcessor(
                    library_id="test_library", save_dir=tmpdir
                )

                # Process the record
                results = []
                async for result in downloader.process(
                    test_record, processor_stage="download"
                ):
>>>>>>> origin/stable
                    results.append(result)

                assert len(results) == 1
                result = results[0]

                # CRITICAL ASSERTIONS:
                # 1. PDF was downloaded
                mock_zot.dump.assert_called_once()

                # 2. file_path is set
                assert result.file_path is not None
                assert result.file_path.endswith("TEST_KEY2.pdf")

                # 3. Content has PDF metadata placeholder
                assert "[PDF Document:" in result.content
                assert "TEST_KEY2.pdf" in result.content

                print("\n✅ SUCCESS: PDF downloaded when no fulltext:")
                print(f"   - Content: {result.content[:80]}...")
                print(f"   - file_path: {result.file_path}")
                print(f"   - PDF downloaded: {mock_zot.dump.called}")
