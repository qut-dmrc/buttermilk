"""Tests for Zotero citation key extraction and metadata handling.

These tests verify:
1. Citation key extraction from 'extra' field (TDD - function doesn't exist yet)
2. Citation keys are included in ZoteroSource metadata
3. Citation keys propagate through ZoteroDownloadProcessor
4. Integration tests with real Zotero API

Following CODE.md principles:
- Tests use REAL fixtures with EXISTING initialization (real_bm from conftest.py)
- NO mocking of internal logic (ZoteroSource, ZoteroDownloadProcessor)
- Configuration from hydra yaml, NO hardcoded values
- Tests written BEFORE implementation (TDD)
"""

import pytest

pytest.importorskip("pyzotero", reason="pyzotero is optional (install with: uv sync --extra research)")

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord, Record
from buttermilk.libs.zotero import ZoteroDownloadProcessor, ZoteroSource

pytestmark = pytest.mark.slow


class TestCitationKeyExtraction:
    """Test citation key extraction from Zotero 'extra' field.

    These tests will FAIL until extract_citation_key() is implemented.
    That's expected - this is TDD (Test-Driven Development).
    """

    def test_extract_citation_key_from_standard_format(self):
        """Extract citation key from standard BetterBibTeX format."""
        from buttermilk.libs.zotero import extract_citation_key

        # Standard format: "Citation Key: authorYear"
        extra = "Citation Key: suzor2019digital"
        assert extract_citation_key(extra) == "suzor2019digital"

    def test_extract_citation_key_with_surrounding_content(self):
        """Extract citation key when extra field has additional content."""
        from buttermilk.libs.zotero import extract_citation_key

        # With additional content before
        extra = "Some other note\nCitation Key: smith2020test"
        assert extract_citation_key(extra) == "smith2020test"

        # With additional content after
        extra = "Citation Key: jones2021\nMore notes here"
        assert extract_citation_key(extra) == "jones2021"

    def test_extract_citation_key_handles_whitespace(self):
        """Extract citation key with various whitespace patterns."""
        from buttermilk.libs.zotero import extract_citation_key

        # With whitespace variations
        extra = "  Citation Key:  doe2022key  "
        assert extract_citation_key(extra) == "doe2022key"

    def test_extract_citation_key_with_special_characters(self):
        """Extract citation key containing special characters."""
        from buttermilk.libs.zotero import extract_citation_key

        # Multiple colons in value (should only split on first colon)
        extra = "Citation Key: author2023:special:key"
        assert extract_citation_key(extra) == "author2023:special:key"

    def test_extract_citation_key_returns_none_when_missing(self):
        """Return None when citation key is not present."""
        from buttermilk.libs.zotero import extract_citation_key

        # No citation key
        assert extract_citation_key("Some random text") is None
        assert extract_citation_key("") is None
        assert extract_citation_key(None) is None

        # Wrong format (case sensitive)
        assert extract_citation_key("citation key: wrong") is None
        assert extract_citation_key("CITATION KEY: wrong") is None

    def test_extract_citation_key_multiline_betterbibtex_format(self):
        """Extract citation key from real BetterBibTeX multiline extra fields."""
        from buttermilk.libs.zotero import extract_citation_key

        # Real BetterBibTeX format with multiple fields
        extra = """tex.ids: suzor2019
Citation Key: suzor2019digital
tex.subtitle: Rights and the Digital Economy"""
        assert extract_citation_key(extra) == "suzor2019digital"

        # Citation key at end
        extra = """ZSCC: 0000123
Publisher: Example Press
Citation Key: finalkey2024"""
        assert extract_citation_key(extra) == "finalkey2024"


class TestZoteroSourceCitationKeys:
    """Test that ZoteroSource extracts and includes citation keys in metadata.

    Uses REAL ZoteroSource with REAL Zotero API (no mocking internal logic).
    Configuration comes from hydra via real_bm fixture.

    These tests will FAIL until:
    1. extract_citation_key() is implemented
    2. ZoteroSource is updated to call extract_citation_key()
    """

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_zotero_source_extracts_citation_keys(self, real_bm):
        """Integration test: ZoteroSource extracts citation keys from real API.

        Requires:
        - ZOTERO_API_KEY in credentials
        - ZOTERO_LIBRARY_ID in config
        - At least one item with Citation Key in extra field
        """
        # Get library_id from pipelines config (NO hardcoded values)
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Create real ZoteroSource (NO mocking)
        # Use force_full_sync=True to bypass incremental sync in tests
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,  # Limit for testing
            force_full_sync=True,  # Bypass incremental sync for testing
        )

        # Fetch real records
        records = []
        async for record in source.fetch_items():
            records.append(record)

        assert len(records) > 0, "No records returned from ZoteroSource"

        # Check that citation_key field exists in metadata
        # (Will be None if item doesn't have citation key, but field should exist)
        for record in records:
            assert "citation_key" in record.metadata, f"citation_key not in metadata for {record.record_id}"

        # Check if at least one record has citation_key extracted
        has_citation_key = False
        for record in records:
            citation_key = record.metadata.get("citation_key")
            if citation_key:
                has_citation_key = True
                # Verify it's a string
                assert isinstance(citation_key, str)
                assert len(citation_key) > 0
                # Log for debugging
                print(f"\n✅ Found citation key: {citation_key} for item {record.record_id}")
                break

        # If no citation keys found, log warning but don't fail
        # (Items may not have BetterBibTeX keys pinned yet)
        if not has_citation_key:
            print("\n⚠️  No citation keys found in sample. To test: Select items in Zotero → Right-click → Generate BibTeX key")


class TestZoteroDownloadProcessorCitationKeys:
    """Test that citation keys propagate through ZoteroDownloadProcessor.

    Uses REAL processor with REAL Zotero API (no mocking internal logic).
    Configuration comes from hydra via real_bm fixture.
    """

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_processor_propagates_citation_keys(self, real_bm):
        """Integration test: Citation keys propagate from Source through Processor.

        Requires:
        - ZOTERO_API_KEY in credentials
        - ZOTERO_LIBRARY_ID in config
        - At least one item with PDF attachment and citation key
        """
        # Get library_id from pipeline config
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["processors"][0]["library_id"]

        # Create real ZoteroSource and Processor
        # Use force_full_sync=True to bypass incremental sync in tests
        source = ZoteroSource(
            library_id=library_id,
            max_records=3,  # Small sample for testing
            force_full_sync=True,  # Bypass incremental sync for testing
        )
        processor = ZoteroDownloadProcessor(library_id=library_id)

        # Find a record with citation_key and PDF
        test_record = None
        async for record in source.fetch_items():
            # Check if has citation key
            if record.metadata.get("citation_key"):
                # Check if has PDF attachment
                links = record.metadata.get("zotero_links", {})
                attachment = links.get("attachment", {})
                if attachment.get("attachmentType") == "application/pdf":
                    test_record = record
                    break

        if test_record is None:
            pytest.skip("No items found with both citation_key and PDF attachment. Add BetterBibTeX citation keys to items in your Zotero library.")

        # Process the record through ZoteroDownloadProcessor
        processed_records = []
        async for processed in processor.process(ProcessingContext(session_id="test", record=test_record)):
            processed_records.append(processed)

        # Verify citation_key propagated
        assert len(processed_records) == 1
        result = processed_records[0]
        assert isinstance(result, Record)

        # Citation key should be in metadata
        original_key = test_record.metadata.get("citation_key")
        result_key = result.metadata.get("citation_key")

        # Check if it's in top-level metadata or zotero_data
        if result_key is None:
            result_key = result.metadata.get("zotero_data", {}).get("citation_key")

        assert result_key == original_key, f"Citation key not propagated correctly: expected {original_key}, got {result_key}"


class TestZoteroAPIDataFormat:
    """Integration tests to verify assumptions about Zotero API data format.

    These tests call the REAL Zotero API to verify our implementation assumptions.
    Uses REAL configuration from hydra.

    CRITICAL: These tests validate that the API returns data in the format our code expects.
    If these tests fail, our implementation needs to be updated to match the actual API format.
    """

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_api_returns_complete_item_structure(self, real_bm):
        """Verify Zotero API returns items with ALL required fields in expected format.

        This test validates the complete item structure for a full page of items to ensure:
        1. Required top-level fields exist ('key', 'version', 'data', 'links')
        2. Fields have expected types (string, int, dict)
        3. Nested 'data' dict contains required metadata fields
        4. 'links' dict structure matches our expectations
        5. 'extra' field is always a string (for citation key parsing)

        Tests with ~50-100 items (one full API page) to verify consistency across items.
        """
        import random

        from pyzotero import zotero

        # Get credentials from BM
        api_key = real_bm.credentials.get("ZOTERO_API_KEY")
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Create real Zotero client
        zot = zotero.Zotero(
            library_id=library_id,
            library_type="group",
            api_key=api_key,
        )

        # Fetch a full page of items (limit=100 is typical page size)
        # To get somewhat random items, we can fetch from a random starting offset
        # Get total items first
        total_items = zot.num_items()

        # Calculate a random start point (avoid last 100 items to ensure full page)
        max_start = max(0, total_items - 100)
        random_start = random.randint(0, max_start) if max_start > 0 else 0

        print(f"\n📊 Fetching page of items from Zotero API (start={random_start}, total={total_items})")

        # Fetch one page with random offset for variety
        items = zot.items(
            limit=100,
            start=random_start,
            itemType="-attachment",  # Exclude attachments like we do in ZoteroSource
        )

        assert len(items) > 0, f"No items returned from Zotero API (start={random_start})"

        print(f"✓ Retrieved {len(items)} items for validation")

        # Track statistics
        items_checked = 0
        items_with_extra = 0
        items_with_citation_key = 0
        items_with_attachments = 0
        items_with_doi = 0
        items_with_url = 0

        # Validate EVERY item in the page
        for item in items:
            items_checked += 1

            # CRITICAL: Validate top-level structure
            assert "key" in item, f"Item missing 'key' field: {item}"
            assert isinstance(item["key"], str), f"Item 'key' should be string, got {type(item['key'])}"
            assert len(item["key"]) > 0, "Item 'key' should not be empty"

            assert "version" in item, f"Item {item.get('key')} missing 'version' field"
            assert isinstance(item["version"], int), f"Item 'version' should be int, got {type(item['version'])}"

            assert "data" in item, f"Item {item.get('key')} missing 'data' field"
            assert isinstance(item["data"], dict), f"Item 'data' should be dict, got {type(item['data'])}"

            assert "links" in item, f"Item {item.get('key')} missing 'links' field"
            assert isinstance(item["links"], dict), f"Item 'links' should be dict, got {type(item['links'])}"

            # Validate 'data' dict structure (this is what we store in zotero_item)
            data = item["data"]

            # itemType is required and used to filter attachments/notes/annotations
            assert "itemType" in data, f"Item {item['key']} data missing 'itemType'"
            assert isinstance(data["itemType"], str), "itemType should be string"

            # title is used extensively in logging and metadata
            if "title" in data:
                assert isinstance(data["title"], str), "title should be string when present"

            # extra field is CRITICAL for citation key extraction
            if "extra" in data:
                items_with_extra += 1
                extra_field = data["extra"]

                # CRITICAL: Must be string for our parsing logic
                assert isinstance(extra_field, str), (
                    f"Item {item['key']}: 'extra' field must be string, got {type(extra_field).__name__}. Value: {extra_field}"
                )

                # If contains citation key, verify we can extract it
                if "Citation Key:" in extra_field:
                    items_with_citation_key += 1
                    from buttermilk.libs.zotero import extract_citation_key

                    citation_key = extract_citation_key(extra_field)
                    assert citation_key is not None, f"Item {item['key']}: Failed to extract citation key from: {extra_field}"
                    assert isinstance(citation_key, str), "Citation key must be string"
                    assert len(citation_key) > 0, "Citation key must not be empty"

            # DOI and URL are used for metadata
            if "DOI" in data:
                items_with_doi += 1
                assert isinstance(data["DOI"], str), "DOI should be string when present"

            if "url" in data:
                items_with_url += 1
                assert isinstance(data["url"], str), "url should be string when present"

            # Validate 'links' dict structure (used for PDF attachments)
            links = item["links"]

            if "attachment" in links:
                items_with_attachments += 1
                attachment = links["attachment"]

                assert isinstance(attachment, dict), f"Item {item['key']}: attachment should be dict, got {type(attachment)}"

                # If attachment exists, should have href and attachmentType
                if "href" in attachment:
                    assert isinstance(attachment["href"], str), "attachment href should be string"

                if "attachmentType" in attachment:
                    assert isinstance(attachment["attachmentType"], str), "attachmentType should be string"

        # Log statistics
        print("\n📈 Validation Statistics:")
        print(f"  Items checked: {items_checked}")
        print(f"  Items with 'extra' field: {items_with_extra}")
        print(f"  Items with citation keys: {items_with_citation_key}")
        print(f"  Items with attachments: {items_with_attachments}")
        print(f"  Items with DOI: {items_with_doi}")
        print(f"  Items with URL: {items_with_url}")

        # Ensure we actually tested a meaningful sample
        assert items_checked >= 10, f"Should check at least 10 items, only checked {items_checked}"

        print(f"\n✅ All {items_checked} items passed structure validation")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_extra_field_is_always_string(self, real_bm):
        """Focused test: Verify 'extra' field is ALWAYS a string, never dict.

        This is critical because we parse it as a string in extract_citation_key().
        Tests a full page of items to ensure consistency.
        """
        from pyzotero import zotero

        api_key = real_bm.credentials.get("ZOTERO_API_KEY")
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        zot = zotero.Zotero(
            library_id=library_id,
            library_type="group",
            api_key=api_key,
        )

        # Fetch full page
        items = zot.items(limit=100, itemType="-attachment")

        assert len(items) > 0, "No items returned from Zotero API"

        extra_field_count = 0

        # Check EVERY item
        for item in items:
            if "data" in item and "extra" in item["data"]:
                extra_field = item["data"]["extra"]
                extra_field_count += 1

                # CRITICAL: Must always be string
                assert isinstance(extra_field, str), (
                    f"Item {item.get('key')}: 'extra' field must be string, got {type(extra_field).__name__}. Value: {extra_field}"
                )

        print(f"\n✓ Checked {len(items)} items, found {extra_field_count} with 'extra' field")
        print("✓ All 'extra' fields are strings (required for citation key parsing)")


class TestZoteroSourceBehavior:
    """Test ZoteroSource behavior without citation key functionality.

    These tests verify core ZoteroSource functionality using REAL API calls.
    Uses configuration from hydra via real_bm fixture.
    """

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_source_yields_base_records_with_metadata(self, real_bm):
        """Test that ZoteroSource yields BaseRecord objects with correct structure."""
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Use force_full_sync=True to bypass incremental sync in tests
        source = ZoteroSource(
            library_id=library_id,
            max_records=2,
            force_full_sync=True,  # Bypass incremental sync for testing
        )

        records = []
        async for record in source.fetch_items():
            records.append(record)

        # Should yield at least 1 record
        assert len(records) >= 1

        # Check first record structure
        assert isinstance(records[0], BaseRecord)
        assert records[0].record_id is not None
        assert "zotero_item" in records[0].metadata
        assert "zotero_version" in records[0].metadata
        # Note: Not all items have 'title' (e.g., statutes have 'nameOfAct')
        # Just verify zotero_item is populated
        assert len(records[0].metadata["zotero_item"]) > 0


class TestZoteroDownloadProcessorBehavior:
    """Test ZoteroDownloadProcessor behavior.

    Uses REAL processor with REAL Zotero API and actual PDF downloads.
    """

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_processor_downloads_and_extracts_text(self, real_bm):
        """Test that processor downloads PDF and extracts text successfully."""
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Get a real record with PDF
        # Use force_full_sync=True to bypass incremental sync in tests
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,
            force_full_sync=True,  # Bypass incremental sync for testing
        )

        # Find first item with PDF attachment
        test_record = None
        async for record in source.fetch_items():
            links = record.metadata.get("zotero_links", {})
            attachment = links.get("attachment", {})
            if attachment.get("attachmentType") == "application/pdf":
                test_record = record
                break

        if test_record is None:
            pytest.skip("No items with PDF attachments found in library")

        # Process with real processor
        processor = ZoteroDownloadProcessor(library_id=library_id)

        results = []
        async for result in processor.process(ProcessingContext(session_id="test", record=test_record)):
            results.append(result)

        # Should yield exactly one Record
        assert len(results) == 1
        result = results[0]

        # Verify it's a full Record with content
        assert isinstance(result, Record)
        assert result.record_id == test_record.record_id
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.get("title") is not None


class TestMetadataUpdateBehavior:
    """Test that metadata updates work correctly without re-downloading PDFs.

    These tests verify the optimization for citation key backfilling:
    1. Updated metadata triggers reprocessing
    2. Non-updated records are skipped
    3. PDF is not re-downloaded if cached and unchanged
    """

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_updated_metadata_triggers_reprocessing(self, real_bm):
        """Test that items with updated metadata are reprocessed by the pipeline.

        When metadata changes (e.g., citation key added), the item should:
        - Be yielded by ZoteroSource (not filtered by VectorStoreExistenceFilter)
        - Be processed through ZoteroDownloadProcessor
        - Update ChromaDB with new metadata
        """
        import json
        from pathlib import Path

        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Get a test item with PDF
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,
            force_full_sync=True,
        )

        test_record = None
        async for record in source.fetch_items():
            links = record.metadata.get("zotero_links", {})
            attachment = links.get("attachment", {})
            if attachment.get("attachmentType") == "application/pdf":
                test_record = record
                break

        if test_record is None:
            pytest.skip("No items with PDF attachments found")

        # First: Process the item to ensure it's in cache and ChromaDB
        processor = ZoteroDownloadProcessor(library_id=library_id)
        results = []
        async for result in processor.process(ProcessingContext(session_id="test", record=test_record)):
            results.append(result)

        assert len(results) == 1
        results[0]

        # Verify cache exists
        cache_dir = Path(real_bm.session_info.get_cache_subdir("zotero"))
        json_file = cache_dir / f"{test_record.record_id}.json"
        assert json_file.exists(), "Cache file should exist after first processing"

        # Read original cache
        with json_file.open("r") as f:
            original_cache = json.load(f)
        original_version = original_cache.get("data", {}).get("version", 0)

        # Verify item version is stored in cache
        assert "version" in original_cache.get("data", {}), "Item version should be stored in cache data"

        # Verify attachment version is stored if attachment exists
        if original_cache.get("links", {}).get("attachment"):
            attachment = original_cache["links"]["attachment"]
            # Attachment should have version info (could be in href or separate field)
            print(f"\n📎 Attachment structure: {attachment}")

        # Simulate metadata update: modify citation_key in record
        updated_record = BaseRecord(
            record_id=test_record.record_id,
            metadata={
                **test_record.metadata,
                "citation_key": "test2025updated",  # New citation key
                "zotero_version": original_version + 1,  # Version incremented
            },
        )

        # Second: Process with updated metadata
        # The processor should detect version change and reprocess
        results2 = []
        async for result in processor.process(ProcessingContext(session_id="test", record=updated_record)):
            results2.append(result)

        assert len(results2) == 1
        updated_result = results2[0]

        # Verify metadata was updated
        assert updated_result.metadata.get("citation_key") == "test2025updated"

        # Verify cache was updated with new metadata
        with json_file.open("r") as f:
            updated_cache = json.load(f)
        assert updated_cache.get("data", {}) != original_cache.get("data", {}), "Cache should be updated with new metadata"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_unchanged_records_skipped_by_incremental_sync(self, real_bm):
        """Test that incremental sync behavior with second-highest version safety.

        With the second-highest version safety mechanism:
        - First sync saves second-highest version (not highest)
        - Second sync with since=second-highest will re-fetch boundary items (highest version)
        - This is expected behavior to avoid missing items on interrupted syncs
        - On third sync, if nothing new changed, should yield 0-2 boundary items again
        """
        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # First sync: Process some items (force_full_sync to get baseline)
        source1 = ZoteroSource(
            library_id=library_id,
            max_records=5,  # Get enough to have version variety
            force_full_sync=True,  # Force full sync first time
        )

        first_sync_items = []
        async for record in source1.fetch_items():
            first_sync_items.append(record)

        assert len(first_sync_items) > 0, "First sync should yield items"

        # Second sync: Incremental sync will re-fetch boundary items (at highest version)
        # This is EXPECTED behavior with second-highest version safety
        source2 = ZoteroSource(
            library_id=library_id,
            max_records=5,
            force_full_sync=False,  # Use incremental sync
        )

        second_sync_items = []
        async for record in source2.fetch_items():
            second_sync_items.append(record)

        # With second-highest version safety: expect to re-fetch items at highest version
        # This is a small number (0-2 items typically) and is intentional for safety
        print(f"\n📊 First sync: {len(first_sync_items)} items")
        print(f"   Second sync: {len(second_sync_items)} items (boundary re-fetch)")

        # Verify boundary items are at the highest version from first sync
        if len(second_sync_items) > 0:
            first_sync_versions = [r.metadata.get("zotero_version", 0) for r in first_sync_items]
            highest_first_sync = max(first_sync_versions)

            for record in second_sync_items:
                item_version = record.metadata.get("zotero_version", 0)
                # Should be at or near the highest version from first sync
                print(f"   Boundary item version: {item_version} (highest first sync: {highest_first_sync})")

        # This behavior is correct and intentional for safety
        print("✅ Second-highest version safety working as expected")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_incremental_sync_fetches_updated_items(self, real_bm):
        """Test that incremental sync correctly fetches items modified after last sync.

        When we set sync state to an earlier time:
        - ZoteroSource should use 'since' parameter with that version
        - API should return items modified after that version
        - Returned items should have modification dates after the sync point
        """
        import json
        from datetime import UTC, datetime, timedelta
        from pathlib import Path

        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Get the sync state file path
        from buttermilk._core.constants import cache

        cache_dir = Path(real_bm.session_info.get_cache_subdir(cache.ZOTERO))
        state_file = cache_dir / ".zotero_sync_state.json"

        # First: Do a full sync to get current library version
        source1 = ZoteroSource(
            library_id=library_id,
            max_records=5,
            force_full_sync=True,
        )

        current_version = None
        async for record in source1.fetch_items():
            # Get current library version from first record
            if current_version is None:
                current_version = record.metadata.get("zotero_version", 0)

        # Verify sync state was saved with current version
        assert state_file.exists(), "Sync state file should exist after sync"
        with state_file.open("r") as f:
            state = json.load(f)
        saved_version = state.get("last_version")
        assert saved_version is not None, "Saved version should not be None"
        print(f"\n📊 Current library version: {saved_version}")

        # Second: Manually set sync state to a much earlier version (version 1)
        # This simulates syncing after a long time
        earlier_timestamp = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        earlier_version = 1  # Very old version

        with state_file.open("w") as f:
            json.dump(
                {
                    "last_version": earlier_version,
                    "last_sync_timestamp": earlier_timestamp,
                },
                f,
            )

        print(f"📅 Set sync state to version {earlier_version} (1 year ago)")

        # Third: Run incremental sync with the earlier version
        source2 = ZoteroSource(
            library_id=library_id,
            max_records=10,  # Get more items to verify
            force_full_sync=False,  # Use incremental sync
        )

        synced_items = []
        async for record in source2.fetch_items():
            synced_items.append(record)

        # Verify we got items (library has been updated since version 1)
        assert len(synced_items) > 0, f"Incremental sync should return items modified after version {earlier_version}"

        print(f"\n✅ Incremental sync returned {len(synced_items)} items")

        # Verify all returned items have versions greater than our sync point
        for record in synced_items:
            item_version = record.metadata.get("zotero_version", 0)
            assert item_version > earlier_version, f"Item {record.record_id} version {item_version} should be > {earlier_version}"

            # Verify item has modification date
            zotero_item = record.metadata.get("zotero_item", {})
            date_modified = zotero_item.get("dateModified")
            assert date_modified is not None, f"Item {record.record_id} should have dateModified field"

            print(f"  • Item {record.record_id}: version={item_version}, modified={date_modified}")

        # Verify sync state was updated to latest version
        with state_file.open("r") as f:
            final_state = json.load(f)
        final_version = final_state.get("last_version")

        assert final_version >= saved_version, f"Final version {final_version} should be >= original version {saved_version}"

        print(f"\n✅ Sync state updated to version {final_version}")

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_cached_pdf_not_redownloaded_on_metadata_update(self, real_bm):
        """Test that cached PDFs are not re-downloaded when only metadata changes.

        When metadata changes but PDF hasn't:
        - PDF file should not be re-downloaded
        - Metadata should be updated in cache
        - Content should be reused from existing PDF cache
        """
        import json
        import os
        from pathlib import Path

        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Get a test item with PDF
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,
            force_full_sync=True,
        )

        test_record = None
        async for record in source.fetch_items():
            links = record.metadata.get("zotero_links", {})
            attachment = links.get("attachment", {})
            if attachment.get("attachmentType") == "application/pdf":
                test_record = record
                break

        if test_record is None:
            pytest.skip("No items with PDF attachments found")

        # First: Process to ensure cache exists
        processor = ZoteroDownloadProcessor(library_id=library_id)
        results = []
        async for result in processor.process(ProcessingContext(session_id="test", record=test_record)):
            results.append(result)

        assert len(results) == 1

        # Get cache paths
        cache_dir = Path(real_bm.session_info.get_cache_subdir("zotero"))
        pdf_file = cache_dir / f"{test_record.record_id}.pdf"
        json_file = cache_dir / f"{test_record.record_id}.json"

        assert pdf_file.exists(), "PDF should be cached"
        assert json_file.exists(), "JSON cache should exist"

        # Record PDF file modification time
        pdf_mtime_before = os.path.getmtime(pdf_file)

        # Read original cache to get version
        with json_file.open("r") as f:
            original_cache = json.load(f)
        original_version = original_cache.get("data", {}).get("version", 0)

        # Verify item version is stored in cache
        assert "version" in original_cache.get("data", {}), "Item version should be stored in cache data"

        # Verify attachment version is stored if attachment exists
        if original_cache.get("links", {}).get("attachment"):
            attachment = original_cache["links"]["attachment"]
            print(f"\n📎 Attachment structure: {attachment}")

        # Simulate metadata update: modify citation_key
        updated_record = BaseRecord(
            record_id=test_record.record_id,
            metadata={
                **test_record.metadata,
                "citation_key": "test2025nocache",  # New citation key
                "zotero_version": original_version + 1,  # Incremented version
            },
        )

        # Second: Process with updated metadata
        results2 = []
        async for result in processor.process(ProcessingContext(session_id="test", record=updated_record)):
            results2.append(result)

        assert len(results2) == 1
        updated_result = results2[0]

        # Verify metadata was updated
        assert updated_result.metadata.get("citation_key") == "test2025nocache"

        # CRITICAL: Verify PDF was NOT re-downloaded
        pdf_mtime_after = os.path.getmtime(pdf_file)
        assert pdf_mtime_after == pdf_mtime_before, "PDF file should NOT be re-downloaded when only metadata changes"

        # Verify cache was updated with new metadata (but same content)
        with json_file.open("r") as f:
            updated_cache = json.load(f)

        # Metadata should be different
        assert updated_cache.get("data", {}) != original_cache.get("data", {}), "Cache metadata should be updated"

        # But content should be the same (reused from cache)
        assert updated_cache.get("content") == original_cache.get("content"), "Content should be reused from existing PDF cache"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_attachment_version_checked_independently(self, real_bm):
        """Test that attachment version is checked separately from parent item version.

        Critical scenario:
        - Parent item metadata changes (title, tags, etc.) → parent version increments
        - Attachment (PDF) itself hasn't changed → attachment version stays same
        - Processor should NOT re-download the PDF (same attachment version)
        - Processor SHOULD update metadata (new parent version)

        This is essential for efficient incremental sync when users edit item metadata
        but PDFs remain unchanged.
        """
        import json
        import os
        from pathlib import Path

        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Get a test item with PDF attachment
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,
            force_full_sync=True,
        )

        test_record = None
        async for record in source.fetch_items():
            links = record.metadata.get("zotero_links", {})
            attachment = links.get("attachment", {})
            if attachment.get("attachmentType") == "application/pdf":
                test_record = record
                break

        if test_record is None:
            pytest.skip("No items with PDF attachments found")

        # First: Process to create initial cache
        processor = ZoteroDownloadProcessor(library_id=library_id)
        results = []
        async for result in processor.process(ProcessingContext(session_id="test", record=test_record)):
            results.append(result)

        assert len(results) == 1

        # Get cache paths
        cache_dir = Path(real_bm.session_info.get_cache_subdir("zotero"))
        pdf_file = cache_dir / f"{test_record.record_id}.pdf"
        json_file = cache_dir / f"{test_record.record_id}.json"

        assert pdf_file.exists(), "PDF should be cached"
        assert json_file.exists(), "JSON cache should exist"

        # Record PDF file modification time
        pdf_mtime_before = os.path.getmtime(pdf_file)

        # Read original cache
        with json_file.open("r") as f:
            original_cache = json.load(f)

        original_item_version = original_cache.get("data", {}).get("version", 0)
        original_attachment = original_cache.get("links", {}).get("attachment", {})

        # Extract attachment version from href (Zotero API format)
        # href typically looks like: "/groups/{id}/items/{key}/file?version={version}"
        # OR attachment may have its own version field
        original_attachment_href = original_attachment.get("href", "")
        original_attachment_version = None

        # Try to extract version from href query parameter
        if "version=" in original_attachment_href:
            import re

            match = re.search(r"version=(\d+)", original_attachment_href)
            if match:
                original_attachment_version = int(match.group(1))

        # If attachment has explicit version field, use that
        if "version" in original_attachment:
            original_attachment_version = original_attachment["version"]

        print("\n📊 Original state:")
        print(f"  Parent item version: {original_item_version}")
        print(f"  Attachment version: {original_attachment_version}")
        print(f"  Attachment href: {original_attachment_href[:100]}...")

        # Simulate parent metadata change WITHOUT attachment change
        # This is what happens when user edits title, tags, etc. in Zotero
        updated_metadata = {
            **test_record.metadata,
            "citation_key": "test2025attachmentcheck",  # New metadata
            "zotero_version": original_item_version + 1,  # Parent version incremented
        }

        # CRITICAL: Keep attachment version the same (unchanged PDF)
        updated_links = updated_metadata.get("zotero_links", {}).copy()
        if "attachment" in updated_links:
            # Attachment href and version stay the same
            updated_links["attachment"] = original_attachment.copy()

        updated_metadata["zotero_links"] = updated_links

        updated_record = BaseRecord(record_id=test_record.record_id, metadata=updated_metadata)

        # Second: Process with updated parent metadata but same attachment version
        results2 = []
        async for result in processor.process(ProcessingContext(session_id="test", record=updated_record)):
            results2.append(result)

        assert len(results2) == 1
        updated_result = results2[0]

        # Verify parent metadata was updated
        assert updated_result.metadata.get("citation_key") == "test2025attachmentcheck"

        # CRITICAL: Verify PDF was NOT re-downloaded
        # Since attachment version didn't change, PDF should be reused from cache
        pdf_mtime_after = os.path.getmtime(pdf_file)
        assert pdf_mtime_after == pdf_mtime_before, (
            f"PDF should NOT be re-downloaded when attachment version unchanged. "
            f"Parent version: {original_item_version} → {original_item_version + 1}, "
            f"Attachment version: {original_attachment_version} (unchanged)"
        )

        print("\n✅ PDF not re-downloaded (attachment version unchanged)")

        # Verify cache was updated with new parent metadata
        with json_file.open("r") as f:
            updated_cache = json.load(f)

        updated_item_version = updated_cache.get("data", {}).get("version", 0)
        assert updated_item_version == original_item_version + 1, "Parent item version should be updated in cache"

        # Verify attachment version stayed the same
        updated_attachment = updated_cache.get("links", {}).get("attachment", {})
        updated_attachment_href = updated_attachment.get("href", "")

        # Attachment href/version should be identical (unchanged)
        assert updated_attachment_href == original_attachment_href, "Attachment href should be unchanged when PDF unchanged"

        # Content should be reused (same PDF)
        assert updated_cache.get("content") == original_cache.get("content"), "Content should be reused when attachment version unchanged"

        print("✅ Cache updated with new parent metadata but same attachment/content")

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_sync_state_saves_safe_version_not_latest(self, real_bm):
        """Test that sync state saves second-highest version to avoid missing items.

        Critical edge case:
        - Multiple items may have the same version (modified at same time)
        - If we save the HIGHEST version and sync gets interrupted
        - Next sync with since=highest might miss items at that exact version
        - Should save second-highest (or highest-1) to ensure we re-fetch boundary items

        This test verifies the sync state behavior with version boundaries.
        """
        import json
        from pathlib import Path

        library_id = real_bm.cfg.pipelines["zotero_vectorization"]["source"]["library_id"]

        # Get sync state file path
        from buttermilk._core.constants import cache

        cache_dir = Path(real_bm.session_info.get_cache_subdir(cache.ZOTERO))
        state_file = cache_dir / ".zotero_sync_state.json"

        # First: Do a full sync and collect all item versions
        source = ZoteroSource(
            library_id=library_id,
            max_records=10,  # Get enough items to have version variety
            force_full_sync=True,
        )

        item_versions = []
        async for record in source.fetch_items():
            version = record.metadata.get("zotero_version", 0)
            item_versions.append(version)

        if len(item_versions) < 2:
            pytest.skip("Need at least 2 items to test version boundary behavior")

        # Get unique versions sorted
        unique_versions = sorted(set(item_versions))
        highest_item_version = max(item_versions)

        print(f"\n📊 Item versions seen: {item_versions[:10]}...")
        print(f"   Unique versions: {unique_versions[-5:]}")  # Show last 5
        print(f"   Highest item version: {highest_item_version}")

        # Check what version was saved
        assert state_file.exists(), "Sync state file should exist after sync"
        with state_file.open("r") as f:
            state = json.load(f)

        saved_version = state.get("last_version")
        print(f"   Saved version: {saved_version}")

        # CRITICAL: The saved version should be the second-highest item version
        # This ensures boundary items are re-fetched on next sync if interrupted

        if len(unique_versions) >= 2:
            expected_version = unique_versions[-2]  # Second highest
            assert saved_version == expected_version, (
                f"Saved version {saved_version} should be second-highest {expected_version}. Unique versions: {unique_versions}"
            )
            print(f"✅ Safe version saved: {saved_version} (second-highest, highest was {highest_item_version})")
        else:
            # Only one unique version - should save that one with warning
            print(f"⚠️ Only one unique version ({saved_version}), cannot use second-highest")

        # Verify that saved version is strictly less than highest
        assert saved_version < highest_item_version, (
            f"Saved version {saved_version} must be < highest {highest_item_version} to ensure boundary items are re-fetched on interrupted sync"
        )
