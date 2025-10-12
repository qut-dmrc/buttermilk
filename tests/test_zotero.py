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

from buttermilk._core.types import BaseRecord, Record
from buttermilk.libs.zotero import ZoteroSource, ZoteroDownloadProcessor


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

    @pytest.mark.integration
    async def test_zotero_source_extracts_citation_keys(self, real_bm):
        """Integration test: ZoteroSource extracts citation keys from real API.

        Requires:
        - ZOTERO_API_KEY in credentials
        - ZOTERO_LIBRARY_ID in config
        - At least one item with Citation Key in extra field
        """
        # Get library_id from hydra config (NO hardcoded values)
        library_id = real_bm.cfg.zotero.library

        # Create real ZoteroSource (NO mocking)
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,  # Limit for testing
        )

        # Fetch real records
        records = []
        async for record in source.fetch_items():
            records.append(record)

        assert len(records) > 0, "No records returned from ZoteroSource"

        # Check that citation_key field exists in metadata
        # (Will be None if item doesn't have citation key, but field should exist)
        for record in records:
            assert "citation_key" in record.metadata, (
                f"citation_key not in metadata for {record.record_id}"
            )

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
            print(
                "\n⚠️  No citation keys found in sample. "
                "To test: Select items in Zotero → Right-click → Generate BibTeX key"
            )


class TestZoteroDownloadProcessorCitationKeys:
    """Test that citation keys propagate through ZoteroDownloadProcessor.

    Uses REAL processor with REAL Zotero API (no mocking internal logic).
    Configuration comes from hydra via real_bm fixture.
    """

    @pytest.mark.integration
    async def test_processor_propagates_citation_keys(self, real_bm):
        """Integration test: Citation keys propagate from Source through Processor.

        Requires:
        - ZOTERO_API_KEY in credentials
        - ZOTERO_LIBRARY_ID in config
        - At least one item with PDF attachment and citation key
        """
        # Get library_id from hydra config
        library_id = real_bm.cfg.zotero.library

        # Create real ZoteroSource and Processor
        source = ZoteroSource(
            library_id=library_id,
            max_records=3,  # Small sample for testing
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
            pytest.skip(
                "No items found with both citation_key and PDF attachment. "
                "Add BetterBibTeX citation keys to items in your Zotero library."
            )

        # Process the record through ZoteroDownloadProcessor
        processed_records = []
        async for processed in processor.process(
            test_record,
            processor_stage="test"
        ):
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

        assert result_key == original_key, (
            f"Citation key not propagated correctly: "
            f"expected {original_key}, got {result_key}"
        )


class TestZoteroAPIDataFormat:
    """Integration tests to verify assumptions about Zotero API data format.

    These tests call the REAL Zotero API to verify our implementation assumptions.
    Uses REAL configuration from hydra.
    """

    @pytest.mark.integration
    async def test_extra_field_is_string_not_dict(self, real_bm):
        """Verify Zotero API returns 'extra' field as string, not dict.

        This test ensures our string parsing logic in extract_citation_key()
        is appropriate for the actual API response format.
        """
        from pyzotero import zotero

        # Get credentials from BM (NO direct os.getenv)
        api_key = real_bm.credentials.get("ZOTERO_API_KEY")
        library_id = real_bm.cfg.zotero.library

        # Create real Zotero client
        zot = zotero.Zotero(
            library_id=library_id,
            library_type="group",
            api_key=api_key,
        )

        # Fetch a small sample
        items = zot.items(limit=5)

        assert len(items) > 0, "No items returned from Zotero API"

        # Check format of 'extra' field
        for item in items:
            if "data" in item and "extra" in item["data"]:
                extra_field = item["data"]["extra"]

                # CRITICAL: Verify 'extra' is a string
                assert isinstance(extra_field, str), (
                    f"Expected 'extra' field to be string, got {type(extra_field).__name__}. "
                    f"Item key: {item.get('key')}, extra value: {extra_field}"
                )

                # If it contains "Citation Key:", verify we can parse it
                if "Citation Key:" in extra_field:
                    from buttermilk.libs.zotero import extract_citation_key
                    citation_key = extract_citation_key(extra_field)
                    assert citation_key is not None, (
                        f"Failed to extract citation key from: {extra_field}"
                    )

                # Log for inspection
                print(f"\n✓ Item {item.get('key')} extra field format:")
                print(f"  Type: {type(extra_field).__name__}")
                print(f"  Value preview: {extra_field[:100] if len(extra_field) > 100 else extra_field}")
                break  # Only need to check one item


class TestZoteroSourceBehavior:
    """Test ZoteroSource behavior without citation key functionality.

    These tests verify core ZoteroSource functionality using REAL API calls.
    Uses configuration from hydra via real_bm fixture.
    """

    @pytest.mark.integration
    async def test_source_yields_base_records_with_metadata(self, real_bm):
        """Test that ZoteroSource yields BaseRecord objects with correct structure."""
        library_id = real_bm.cfg.zotero.library

        source = ZoteroSource(
            library_id=library_id,
            max_records=2,
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
        assert records[0].metadata["zotero_item"].get("title") is not None


class TestZoteroDownloadProcessorBehavior:
    """Test ZoteroDownloadProcessor behavior.

    Uses REAL processor with REAL Zotero API and actual PDF downloads.
    """

    @pytest.mark.integration
    async def test_processor_downloads_and_extracts_text(self, real_bm):
        """Test that processor downloads PDF and extracts text successfully."""
        library_id = real_bm.cfg.zotero.library

        # Get a real record with PDF
        source = ZoteroSource(
            library_id=library_id,
            max_records=5,
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
        async for result in processor.process(test_record, processor_stage="test"):
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
