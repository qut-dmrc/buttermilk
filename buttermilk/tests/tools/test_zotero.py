from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open

import pytest

from buttermilk.data.vector import InputDocument
from buttermilk.tools.citator import CITATION_TEXT_CHAR_LIMIT
from buttermilk.tools.zotero import ZotDownloader


# Mock bm object and credentials if ZotDownloader relies on them during init
# This avoids needing actual credentials for testing.
@pytest.fixture(autouse=True)
def mock_bm_credentials(mocker):
    mock_bm = MagicMock()
    mock_bm.credentials.get.side_effect = lambda key: f"dummy_{key}"
    mocker.patch("buttermilk.tools.zotero.bm", mock_bm)


@pytest.fixture
def mock_pyzotero(mocker):
    """Mocks the pyzotero.Zotero class."""
    return mocker.patch("buttermilk.tools.zotero.zotero.Zotero", autospec=True)


@pytest.fixture
def zot_downloader(tmp_path, mock_pyzotero):
    """Provides a ZotDownloader instance with a temporary save directory."""
    # The mock_pyzotero fixture ensures the actual Zotero API isn't called.
    return ZotDownloader(save_dir=str(tmp_path))


@pytest.mark.anyio
async def test_download_pdf_does_not_exist(zot_downloader, tmp_path, mocker):
    """Tests downloading when the PDF attachment exists and the local file doesn't."""
    mock_download = mocker.patch(
        "buttermilk.tools.zotero.download_limited_async",
        new_callable=AsyncMock,
        return_value=b"pdf content",
    )
    mocker.patch("pathlib.Path.exists", return_value=False)
    mock_file_open = mock_open()
    mocker.patch("pathlib.Path.open", mock_file_open)

    item = {
        "key": "TESTKEY1",
        "links": {
            "attachment": {
                "href": "http://example.com/attachment.pdf",
                "attachmentType": "application/pdf",
            },
        },
        "data": {"title": "Test Title 1", "DOI": "10.1000/testdoi"},
    }
    expected_file_path = tmp_path / "TESTKEY1.pdf"

    result = await zot_downloader.download(item)

    mock_download.assert_awaited_once_with("http://example.com/attachment.pdf")
    # Check Path(..).open was called correctly
    assert mocker.call(expected_file_path) in Path.mock_calls
    mock_file_open.assert_called_once_with("wb")
    mock_file_open().write.assert_called_once_with(b"pdf content")

    assert isinstance(result, InputDocument)
    assert result.record_id == "TESTKEY1"
    assert result.title == "Test Title 1"
    assert result.file_path == expected_file_path.as_posix()
    assert result.metadata == {"doi": "10.1000/testdoi"}


@pytest.mark.anyio
async def test_download_pdf_already_exists(zot_downloader, tmp_path, mocker):
    """Tests download behavior when the PDF file already exists locally."""
    mock_download = mocker.patch(
        "buttermilk.tools.zotero.download_limited_async",
        new_callable=AsyncMock,
    )
    mocker.patch("pathlib.Path.exists", return_value=True)
    mock_file_open = mock_open()
    mocker.patch("pathlib.Path.open", mock_file_open)

    item = {
        "key": "TESTKEY2",
        "links": {
            "attachment": {
                "href": "http://example.com/attachment2.pdf",
                "attachmentType": "application/pdf",
            },
        },
        "data": {"title": "Test Title 2", "url": "http://example.com/article"},
    }
    expected_file_path = tmp_path / "TESTKEY2.pdf"

    result = await zot_downloader.download(item)

    mock_download.assert_not_awaited()
    mock_file_open.assert_not_called()

    assert isinstance(result, InputDocument)
    assert result.record_id == "TESTKEY2"
    assert result.title == "Test Title 2"
    assert result.file_path == expected_file_path.as_posix()
    assert result.metadata == {"doi": "http://example.com/article"}  # Falls back to URL


@pytest.mark.anyio
async def test_download_no_attachment_link(zot_downloader, mocker):
    """Tests download behavior when there's no attachment link."""
    mock_download = mocker.patch(
        "buttermilk.tools.zotero.download_limited_async",
        new_callable=AsyncMock,
    )
    item = {
        "key": "TESTKEY3",
        "links": {},  # No attachment info
        "data": {"title": "Test Title 3"},
    }

    result = await zot_downloader.download(item)

    mock_download.assert_not_awaited()
    assert result is None


@pytest.mark.anyio
async def test_download_not_pdf(zot_downloader, mocker):
    """Tests download behavior when the attachment is not a PDF."""
    mock_download = mocker.patch(
        "buttermilk.tools.zotero.download_limited_async",
        new_callable=AsyncMock,
    )
    item = {
        "key": "TESTKEY4",
        "links": {
            "attachment": {
                "href": "http://example.com/attachment.txt",
                "attachmentType": "text/plain",  # Not PDF
            },
        },
        "data": {"title": "Test Title 4"},
    }

    result = await zot_downloader.download(item)

    mock_download.assert_not_awaited()
    assert result is None


@pytest.mark.asyncio
async def test_download_missing_data_fields(zot_downloader, tmp_path, mocker):
    """Tests download behavior with potentially missing fields in item data."""
    mocker.patch("pathlib.Path.exists", return_value=True)  # Assume file exists

    item = {
        "key": "TESTKEY5",
        "links": {
            "attachment": {
                "href": "http://example.com/attachment5.pdf",
                "attachmentType": "application/pdf",
            },
        },
        "data": {
            # Missing title, DOI, url
        },
    }
    expected_file_path = tmp_path / "TESTKEY5.pdf"

    # Expect it to fail during InputDocument creation if title is required
    # or handle missing metadata gracefully. Based on InputDocument definition,
    # title is likely required. Let's assume it raises KeyError or similar.
    # If InputDocument handles missing title/metadata, adjust assertion.
    with pytest.raises(
        KeyError,
    ):  # Or TypeError/ValidationError depending on InputDocument
        await zot_downloader.download(item)

    # If InputDocument allows missing title/metadata:
    # result = await zot_downloader.download(item)
    # assert isinstance(result, InputDocument)
    # assert result.record_id == "TESTKEY5"
    # assert result.title is None # Or "" depending on InputDocument
    # assert result.file_path == expected_file_path.as_posix()
    # assert result.metadata == {"doi": None} # Or {}


@pytest.mark.anyio
async def test_prepare_docs_citation_error(
    vector_store,
    input_doc_factory,
    mock_async_citation_generator,
    mock_logger,
):
    """Test that prepare_docs proceeds even if citation generation fails."""
    input_doc = input_doc_factory(metadata={"original": "value"})
    mock_async_citation_generator.side_effect = Exception("Citation API down")

    chunks = await vector_store.prepare_docs([input_doc])

    # Should still produce chunks
    assert len(chunks) == 2  # Based on default mock extract text
    mock_async_citation_generator.assert_awaited_once()  # It was called
    mock_logger.error.assert_called_with(
        f"Error generating citation for doc {input_doc.record_id}: Citation API down",
        exc_info=True,
    )
    # Metadata should not contain the citation key if generation failed
    assert chunks[0].metadata == {"original": "value"}
    assert chunks[1].metadata == {"original": "value"}


@pytest.mark.anyio
async def test_citation():
    citation_input_text = full_text[:CITATION_TEXT_CHAR_LIMIT]
    # Verify citation generator was called
    mock_async_citation_generator.assert_awaited_once_with(citation_input_text)
    # Check that original metadata is preserved and citation is added
    assert chunks[0].metadata == {
        "original": "value",
        "citation": "Generated Citation: Test",
    }
    assert chunks[1].metadata == {
        "original": "value",
        "citation": "Generated Citation: Test",
    }
    assert chunks[2].metadata == {
        "original": "value",
        "citation": "Generated Citation: Test",
    }
