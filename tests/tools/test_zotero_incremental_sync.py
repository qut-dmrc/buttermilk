"""Unit tests for Zotero incremental sync functionality.

Tests version tracking and incremental fetching of Zotero items.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from buttermilk._core.types import Record
from buttermilk.libs.zotero import ZotDownloader


class TestZoteroIncrementalSync:
    """Test suite for Zotero incremental sync functionality."""

    @pytest.fixture
    def mock_get_bm(self, real_bm):
        """Mock the get_bm function to provide test credentials."""
        with patch("buttermilk.libs.zotero.get_bm") as mock:
            real_bm.credentials.get.return_value = "test_api_key"
            mock.return_value = real_bm
            yield mock

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return tmp_path

    @pytest.fixture
    def zot_downloader(self, temp_dir, mock_get_bm):
        """Create a ZotDownloader instance for testing."""
        with patch("buttermilk.libs.zotero.zotero.Zotero") as mock_zotero_class:
            mock_zot_instance = MagicMock()
            mock_zotero_class.return_value = mock_zot_instance
            
            downloader = ZotDownloader(
                save_dir=str(temp_dir),
                library="test_library"
            )
            # Make sure the mock is properly set
            downloader._zot = mock_zot_instance
            return downloader

    def test_version_state_file_path(self, zot_downloader, temp_dir):
        """Test that version state file path is correctly constructed."""
        expected_path = Path(temp_dir) / ".zotero_sync_state.json"
        assert zot_downloader._get_state_file_path() == expected_path

    def test_load_version_state_no_file(self, zot_downloader):
        """Test loading version state when file doesn't exist."""
        state = zot_downloader._load_version_state()
        assert state == {"last_version": None, "last_sync_timestamp": None}

    def test_load_version_state_with_file(self, zot_downloader, temp_dir):
        """Test loading version state from existing file."""
        state_file = Path(temp_dir) / ".zotero_sync_state.json"
        test_state = {
            "last_version": 12345,
            "last_sync_timestamp": "2024-01-15T10:30:00Z"
        }
        state_file.write_text(json.dumps(test_state))
        
        loaded_state = zot_downloader._load_version_state()
        assert loaded_state == test_state

    def test_save_version_state(self, zot_downloader, temp_dir):
        """Test saving version state to file."""
        test_version = 67890
        test_timestamp = "2024-01-16T14:45:00Z"
        
        zot_downloader._save_version_state(test_version, test_timestamp)
        
        state_file = Path(temp_dir) / ".zotero_sync_state.json"
        assert state_file.exists()
        
        saved_state = json.loads(state_file.read_text())
        assert saved_state["last_version"] == test_version
        assert saved_state["last_sync_timestamp"] == test_timestamp

    @pytest.mark.anyio
    async def test_get_all_records_first_run(self, zot_downloader, mock_get_bm):
        """Test get_all_records on first run (no previous version)."""
        # Mock the Zotero API
        mock_zot = zot_downloader._zot
        
        # Mock initial items response
        mock_items = [
            {"key": "ITEM1", "data": {"title": "Test Item 1", "version": 100}},
            {"key": "ITEM2", "data": {"title": "Test Item 2", "version": 101}}
        ]
        mock_zot.items.return_value = mock_items
        mock_zot.links = {"next": None}
        
        # Mock the last-modified-version header
        mock_zot._extract_links.return_value = {}
        mock_zot.request.headers = {"last-modified-version": "101"}
        
        # Mock download_record to return Records
        async def mock_download(self, item):
            return Record(
                record_id=item["key"],
                content="Test content",
                file_path=f"/test/{item['key']}.pdf",
                metadata={"title": item["data"]["title"]}
            )
        
        # Mock the download_record method properly
        with patch.object(ZotDownloader, "download_record", new=mock_download):
            records = []
            async for record in zot_downloader.get_all_records():
                records.append(record)
        
        # Should fetch all items on first run
        mock_zot.items.assert_called_once_with(
            itemType="-attachment",
            limit=100
        )
        
        # Should have processed both items
        assert len(records) == 2
        assert records[0].record_id == "ITEM1"
        assert records[1].record_id == "ITEM2"

    @pytest.mark.anyio
    async def test_get_all_records_incremental(self, zot_downloader, temp_dir, mock_get_bm):
        """Test get_all_records with existing version (incremental sync)."""
        # Set up previous sync state
        state_file = Path(temp_dir) / ".zotero_sync_state.json"
        previous_state = {
            "last_version": 95,
            "last_sync_timestamp": "2024-01-14T10:00:00Z"
        }
        state_file.write_text(json.dumps(previous_state))
        
        # Mock the Zotero API
        mock_zot = zot_downloader._zot
        
        # Mock items response - only items with version > 95
        mock_items = [
            {"key": "ITEM3", "data": {"title": "Updated Item", "version": 102}}
        ]
        mock_zot.items.return_value = mock_items
        mock_zot.links = {"next": None}
        mock_zot._extract_links.return_value = {}
        mock_zot.request.headers = {"last-modified-version": "102"}
        
        # Mock download_record
        async def mock_download(self, item):
            return Record(
                record_id=item["key"],
                content="Updated content",
                file_path=f"/test/{item['key']}.pdf",
                metadata={"title": item["data"]["title"]}
            )
        
        # Mock the download_record method properly
        with patch.object(ZotDownloader, "download_record", new=mock_download):
            records = []
            async for record in zot_downloader.get_all_records():
                records.append(record)
        
        # Should use 'since' parameter for incremental sync
        mock_zot.items.assert_called_once_with(
            itemType="-attachment",
            limit=100,
            since=95,
            sort="dateModified",
            direction="asc"
        )
        
        # Should have processed only the new/updated item
        assert len(records) == 1
        assert records[0].record_id == "ITEM3"

    @pytest.mark.anyio
    async def test_version_state_updated_after_sync(self, zot_downloader, temp_dir, mock_get_bm):
        """Test that version state is updated after successful sync."""
        # Mock the Zotero API
        mock_zot = zot_downloader._zot
        mock_items = [{"key": "ITEM1", "data": {"title": "Test", "version": 105}}]
        mock_zot.items.return_value = mock_items
        mock_zot.links = {"next": None}
        mock_zot._extract_links.return_value = {}
        
        # Mock the response headers properly - need to set up the request mock
        mock_request = Mock()
        mock_request.headers = {"last-modified-version": "105"}
        mock_zot.request = mock_request
        
        # For pagination
        mock_response = Mock()
        mock_response.headers = {"last-modified-version": "105"}
        mock_zot._retrieve_data.return_value = mock_response
        mock_response.json.return_value = []
        
        # Mock download_record
        async def mock_download(self, item):
            return Record(
                record_id=item["key"],
                content="Test",
                file_path=f"/test/{item['key']}.pdf",
                metadata={}
            )
        
        # Mock the download_record method properly
        with patch.object(ZotDownloader, "download_record", new=mock_download):
            records = []
            async for record in zot_downloader.get_all_records():
                records.append(record)
        
        # Check that state file was updated
        state_file = Path(temp_dir) / ".zotero_sync_state.json"
        assert state_file.exists()
        
        saved_state = json.loads(state_file.read_text())
        assert saved_state["last_version"] == 105
        assert saved_state["last_sync_timestamp"] is not None

    @pytest.mark.anyio
    async def test_reset_sync_state(self, zot_downloader, temp_dir):
        """Test resetting sync state for full re-sync."""
        # Create existing state
        state_file = Path(temp_dir) / ".zotero_sync_state.json"
        state_file.write_text(json.dumps({
            "last_version": 200,
            "last_sync_timestamp": "2024-01-10T10:00:00Z"
        }))
        
        # Reset state
        zot_downloader.reset_sync_state()
        
        # State file should be deleted
        assert not state_file.exists()
        
        # Loading state should return defaults
        state = zot_downloader._load_version_state()
        assert state["last_version"] is None
        assert state["last_sync_timestamp"] is None

    @pytest.mark.anyio
    async def test_get_all_records_with_force_full_sync(self, zot_downloader, temp_dir, mock_get_bm):
        """Test that force_full_sync parameter bypasses incremental sync."""
        # Set up existing state
        state_file = Path(temp_dir) / ".zotero_sync_state.json"
        state_file.write_text(json.dumps({
            "last_version": 150,
            "last_sync_timestamp": "2024-01-10T10:00:00Z"
        }))
        
        # Mock Zotero API
        mock_zot = zot_downloader._zot
        mock_zot.items.return_value = []
        mock_zot.links = {"next": None}
        
        # Run with force_full_sync
        records = []
        async for record in zot_downloader.get_all_records(force_full_sync=True):
            records.append(record)
        
        # Should NOT use 'since' parameter
        mock_zot.items.assert_called_once_with(
            itemType="-attachment",
            limit=100
        )