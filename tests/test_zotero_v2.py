"""Tests for the refactored Zotero source and processor.

These tests verify:
1. ZoteroSource yields BaseRecord objects with IDs and metadata
2. ZoteroDownloadProcessor converts IDs to full Record objects with content
3. Filtering works correctly
4. Incremental sync state is tracked
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from buttermilk._core.types import BaseRecord, Record
from buttermilk.libs.zotero import (
    ZoteroSource,
    ZoteroDownloadProcessor,
    VectorStoreExistenceFilter,
)


@pytest.fixture
def mock_zotero_api():
    """Mock Zotero API responses."""
    mock_api = Mock()

    # Mock items() response
    mock_api.items.return_value = [
        {
            "key": "ITEM001",
            "version": 100,
            "data": {
                "itemType": "journalArticle",
                "title": "Test Article 1",
                "DOI": "10.1234/test1",
            },
            "links": {
                "attachment": {
                    "href": "/groups/123/items/ATT001",
                    "attachmentType": "application/pdf",
                }
            },
        },
        {
            "key": "ITEM002",
            "version": 101,
            "data": {
                "itemType": "book",
                "title": "Test Book",
                "url": "https://example.com/book",
            },
            "links": {},
        },
    ]

    # Mock fulltext_item() response
    mock_api.fulltext_item.return_value = {
        "content": "This is the full text from Zotero API",
        "indexedPages": 10,
        "totalPages": 10,
    }

    return mock_api


@pytest.fixture
def temp_save_dir(tmp_path):
    """Create temporary directory for test files."""
    save_dir = tmp_path / "zotero_test"
    save_dir.mkdir()
    return save_dir


class TestZoteroSource:
    """Test ZoteroSource class."""

    @pytest.mark.asyncio
    async def test_fetch_items_yields_base_records(self, mock_zotero_api, temp_save_dir):
        """Test that ZoteroSource yields BaseRecord objects with correct structure."""
        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        # Patch the zot property to return our mock
        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_zotero_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Should yield 2 records
        assert len(records) == 2

        # Check first record structure
        assert isinstance(records[0], BaseRecord)
        assert records[0].record_id == "ITEM001"
        assert "zotero_item" in records[0].metadata
        assert records[0].metadata["zotero_item"]["title"] == "Test Article 1"
        assert records[0].metadata["zotero_version"] == 100

    @pytest.mark.asyncio
    async def test_filter_applied_correctly(self, mock_zotero_api, temp_save_dir):
        """Test that RecordFilter is applied correctly."""

        # Create a filter that only includes records with "Article" in title
        class TitleFilter:
            async def should_include(self, record: BaseRecord) -> bool:
                title = record.metadata.get("zotero_item", {}).get("title", "")
                return "Article" in title

        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
            filter=TitleFilter(),
        )

        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_zotero_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Should only yield 1 record (the one with "Article" in title)
        assert len(records) == 1
        assert records[0].record_id == "ITEM001"

    @pytest.mark.asyncio
    async def test_sync_state_saved(self, mock_zotero_api, temp_save_dir):
        """Test that sync state is saved after fetching items."""
        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_zotero_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Check that state file was created
        state_file = temp_save_dir / ".zotero_sync_state.json"
        assert state_file.exists()

        # Check state content
        with state_file.open("r") as f:
            state = json.load(f)

        assert state["last_version"] == 101  # Highest version from mock data
        assert state["last_sync_timestamp"] is not None


class TestVectorStoreExistenceFilter:
    """Test VectorStoreExistenceFilter."""

    @pytest.mark.asyncio
    async def test_filters_existing_records(self):
        """Test that filter excludes records that exist in vector store."""
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.check_document_exists.return_value = True  # Record exists

        filter_instance = VectorStoreExistenceFilter(mock_vector_store)

        record = BaseRecord(record_id="TEST001", metadata={})

        # Should return False (don't include) because record exists
        should_include = await filter_instance.should_include(record)
        assert should_include is False
        mock_vector_store.check_document_exists.assert_called_once_with("TEST001")

    @pytest.mark.asyncio
    async def test_includes_new_records(self):
        """Test that filter includes records that don't exist in vector store."""
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.check_document_exists.return_value = False  # Record doesn't exist

        filter_instance = VectorStoreExistenceFilter(mock_vector_store)

        record = BaseRecord(record_id="TEST002", metadata={})

        # Should return True (include) because record doesn't exist
        should_include = await filter_instance.should_include(record)
        assert should_include is True


class TestZoteroDownloadProcessor:
    """Test ZoteroDownloadProcessor."""

    @pytest.mark.asyncio
    async def test_process_downloads_full_text(self, mock_zotero_api, temp_save_dir):
        """Test that processor downloads full text successfully."""
        processor = ZoteroDownloadProcessor(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        # Create input record (what ZoteroSource would yield)
        input_record = BaseRecord(
            record_id="ITEM001",
            metadata={
                "zotero_item": {
                    "title": "Test Article",
                    "DOI": "10.1234/test",
                },
                "zotero_links": {
                    "attachment": {
                        "href": "/groups/123/items/ATT001",
                        "attachmentType": "application/pdf",
                    }
                },
            },
        )

        with patch.object(ZoteroDownloadProcessor, "zot", new_callable=lambda: property(lambda self: mock_zotero_api)):
            results = []
            async for record in processor.process(input_record, processor_stage="test"):
                results.append(record)

        # Should yield exactly one Record
        assert len(results) == 1
        result = results[0]

        # Verify it's a full Record with content
        assert isinstance(result, Record)
        assert result.record_id == "ITEM001"
        assert result.content == "This is the full text from Zotero API"
        assert result.metadata["title"] == "Test Article"

    @pytest.mark.asyncio
    async def test_process_raises_on_no_attachment(self, mock_zotero_api, temp_save_dir):
        """Test that processor raises error when no PDF attachment exists."""
        processor = ZoteroDownloadProcessor(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        # Create input record with no attachment
        input_record = BaseRecord(
            record_id="ITEM002",
            metadata={
                "zotero_item": {"title": "No PDF Item"},
                "zotero_links": {},  # No attachment
            },
        )

        with patch.object(ZoteroDownloadProcessor, "zot", new_callable=lambda: property(lambda self: mock_zotero_api)):
            with pytest.raises(Exception, match="No PDF attachment found"):
                async for _ in processor.process(input_record, processor_stage="test"):
                    pass

    @pytest.mark.asyncio
    async def test_process_uses_cache(self, mock_zotero_api, temp_save_dir):
        """Test that processor uses cached content when available."""
        processor = ZoteroDownloadProcessor(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        # Create cached file
        cache_file = temp_save_dir / "ITEM001.json"
        cache_data = {
            "key": "ITEM001",
            "content": "Cached content",
            "data": {"title": "Cached Article"},
        }
        with cache_file.open("w") as f:
            json.dump(cache_data, f)

        input_record = BaseRecord(
            record_id="ITEM001",
            metadata={
                "zotero_item": {"title": "Cached Article"},
                "zotero_links": {
                    "attachment": {
                        "href": "/groups/123/items/ATT001",
                        "attachmentType": "application/pdf",
                    }
                },
            },
        )

        # Mock should NOT be called if cache works
        with patch.object(ZoteroDownloadProcessor, "zot", new_callable=lambda: property(lambda self: mock_zotero_api)):
            results = []
            async for record in processor.process(input_record, processor_stage="test"):
                results.append(record)

        # Should use cached content
        assert len(results) == 1
        assert results[0].content == "Cached content"

        # Zotero API should not have been called
        mock_zotero_api.fulltext_item.assert_not_called()
