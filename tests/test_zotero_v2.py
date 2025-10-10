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

    # Mock last_modified_version() - library version (different from item versions)
    mock_api.last_modified_version.return_value = 200

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
    async def test_library_version_saved_not_max_item_version(self, temp_save_dir):
        """Test that library version is saved to state, not max item version.

        This test verifies the fix for issue #91:
        - WRONG: Tracking max(item.version) across all items
        - RIGHT: Using zot.last_modified_version() for library version
        """
        # Create mock API with items that have different versions
        mock_api = Mock()
        mock_api.items.return_value = [
            {
                "key": "ITEM001",
                "version": 51050,  # Item version
                "data": {"itemType": "journalArticle", "title": "Article 1"},
                "links": {},
            },
            {
                "key": "ITEM002",
                "version": 51055,  # Item version
                "data": {"itemType": "book", "title": "Book 1"},
                "links": {},
            },
            {
                "key": "ITEM003",
                "version": 51061,  # Max item version
                "data": {"itemType": "journalArticle", "title": "Article 2"},
                "links": {},
            },
        ]
        # Mock library version (different from max item version)
        mock_api.last_modified_version.return_value = 51100  # Library version is higher

        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Verify we got records
        assert len(records) == 3

        # Check state file
        state_file = temp_save_dir / ".zotero_sync_state.json"
        assert state_file.exists()

        with state_file.open("r") as f:
            state = json.load(f)

        # EXPECTED: State should contain library version (51100), not max item version (51061)
        # This will FAIL with current implementation
        assert state["last_version"] == 51100, (
            f"Expected library version 51100, "
            f"but got {state['last_version']} (max item version is 51061)"
        )

    @pytest.mark.asyncio
    async def test_items_returned_in_date_order(self, temp_save_dir):
        """Test that items are returned sorted by dateModified in ascending order.

        Verifies that API is called with correct sort parameters:
        - sort: "dateModified"
        - direction: "asc"
        """
        mock_api = Mock()
        mock_api.items.return_value = [
            {
                "key": "ITEM001",
                "version": 100,
                "data": {
                    "itemType": "journalArticle",
                    "title": "Oldest",
                    "dateModified": "2020-01-01T00:00:00Z",
                },
                "links": {},
            },
            {
                "key": "ITEM002",
                "version": 101,
                "data": {
                    "itemType": "book",
                    "title": "Middle",
                    "dateModified": "2021-01-01T00:00:00Z",
                },
                "links": {},
            },
            {
                "key": "ITEM003",
                "version": 102,
                "data": {
                    "itemType": "journalArticle",
                    "title": "Newest",
                    "dateModified": "2022-01-01T00:00:00Z",
                },
                "links": {},
            },
        ]
        mock_api.last_modified_version.return_value = 102

        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Verify API was called with correct sort parameters
        mock_api.items.assert_called()
        call_kwargs = mock_api.items.call_args[1]
        assert call_kwargs["sort"] == "dateModified"
        assert call_kwargs["direction"] == "asc"

        # Verify records are in date order
        dates = [r.metadata["zotero_item"]["dateModified"] for r in records]
        assert dates == sorted(dates), "Items not in ascending date order"

    @pytest.mark.asyncio
    async def test_incremental_sync_uses_since_parameter(self, temp_save_dir):
        """Test that incremental sync uses 'since' parameter with library version.

        Verifies that:
        1. When sync state exists, API is called with since=<last_version>
        2. The since parameter uses library version, not item version
        """
        mock_api = Mock()

        # First call (with since parameter) returns newer items
        mock_api.items.return_value = [
            {
                "key": "ITEM_NEW",
                "version": 51100,
                "data": {"itemType": "journalArticle", "title": "New Article"},
                "links": {},
            }
        ]
        mock_api.last_modified_version.return_value = 51100

        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        # Simulate previous sync at library version 51000
        source._save_sync_state(51000, "2025-01-01T00:00:00Z")

        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Verify API was called with since parameter
        mock_api.items.assert_called()
        call_kwargs = mock_api.items.call_args[1]
        assert "since" in call_kwargs
        assert call_kwargs["since"] == 51000  # Should use library version from state

    @pytest.mark.asyncio
    async def test_no_items_returned_when_library_unchanged(self, temp_save_dir):
        """Test that no items are returned when library hasn't changed since last sync.

        When since=<current_library_version>, API should return empty list.
        """
        mock_api = Mock()

        # Mock returns empty list (no changes)
        mock_api.items.return_value = []
        mock_api.last_modified_version.return_value = 51100

        source = ZoteroSource(
            library_id="123",
            save_dir=str(temp_save_dir),
        )

        # Simulate sync at current library version (nothing should have changed)
        source._save_sync_state(51100, "2025-01-01T00:00:00Z")

        with patch.object(ZoteroSource, "zot", new_callable=lambda: property(lambda self: mock_api)):
            records = []
            async for record in source.fetch_items():
                records.append(record)

        # Should get 0 records
        assert len(records) == 0, "Expected no records when library unchanged"

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

        # Should save library version (200), not max item version (101)
        assert state["last_version"] == 200  # Library version from mock
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
