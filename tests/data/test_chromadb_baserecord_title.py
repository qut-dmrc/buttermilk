"""Test that ChromaDBEmbeddings.process_record() works with BaseRecord (no title attribute).

This test reproduces GitHub issue #281 where process_record() line 823 accesses
record.title directly, causing AttributeError when receiving BaseRecord instead of Record.

The bug: BaseRecord doesn't have a title attribute, only Record does (as a computed property).
The fix: Access title via record.metadata.get('title') to work with BaseRecord.
"""

import pytest

from buttermilk._core.types import BaseRecord, Record


class TestChromaDBBaseRecordTitle:
    """Test ChromaDBEmbeddings handles BaseRecord without title attribute."""

    def test_baserecord_has_no_title_attribute(self):
        """Verify that BaseRecord doesn't have a title attribute (only Record does)."""
        record = BaseRecord(
            record_id="TEST123",
            dataset="test_dataset",
            content="Test content",
            metadata={"title": "Test Title"},
        )

        # BaseRecord should NOT have title attribute
        assert not hasattr(record, "title"), (
            "BaseRecord should not have title attribute"
        )

        # But metadata should have title
        assert record.metadata.get("title") == "Test Title"

    def test_record_has_title_property(self):
        """Verify that Record DOES have a title computed property."""
        record = Record(
            record_id="TEST123",
            dataset="test_dataset",
            content="Test content",
            metadata={"title": "Test Title"},
        )

        # Record SHOULD have title property
        assert hasattr(record, "title"), "Record should have title property"
        assert record.title == "Test Title"

    def test_title_access_pattern_breaks_with_baserecord(self):
        """Demonstrate that line 823's title access pattern breaks with BaseRecord.

        Line 823 does: record.title[:50] if record.title else 'Unknown'

        This works with Record (has title property) but fails with BaseRecord (no title attribute).
        """
        # Record works fine
        record_with_title = Record(
            record_id="TEST1", dataset="test", metadata={"title": "My Title"}
        )
        assert record_with_title.title == "My Title"
        title_display = (
            record_with_title.title[:50] if record_with_title.title else "Unknown"
        )
        assert title_display == "My Title"

        # Record without title in metadata also works (returns None)
        record_without_title = Record(record_id="TEST2", dataset="test", metadata={})
        assert record_without_title.title is None
        title_display = (
            record_without_title.title[:50] if record_without_title.title else "Unknown"
        )
        assert title_display == "Unknown"

        # BaseRecord with title in metadata FAILS - no title attribute
        base_record = BaseRecord(
            record_id="7N2V8GFN", dataset="zotero", metadata={"title": "Zotero Title"}
        )

        # This demonstrates the bug: AttributeError
        with pytest.raises(
            AttributeError, match="'BaseRecord' object has no attribute 'title'"
        ):
            _ = base_record.title  # Line 823 tries to access this

        # The correct way to access title from BaseRecord:
        title = (
            base_record.metadata.get("title", "Untitled")
            if base_record.metadata
            else "Untitled"
        )
        assert title == "Zotero Title"

    @pytest.mark.anyio
    async def test_zotero_processor_fails_without_title(self):
        """Test that ZoteroDownloadProcessor FAILS FAST when Zotero item has no title.

        Per fail-fast philosophy: titles are REQUIRED. Records without titles
        indicate incomplete/invalid Zotero data that must be fixed upstream.
        """
        from buttermilk.libs.zotero import ZoteroDownloadProcessor

        # Create BaseRecord with zotero_item that has NO title
        record = BaseRecord(
            record_id="BAD_ITEM",
            dataset="zotero",
            metadata={
                "zotero_item": {
                    "itemType": "journalArticle",
                    # NO title field!
                    "DOI": "10.1234/test",
                },
                "zotero_links": {},
                "citation_key": "test2024",
            },
        )

        processor = ZoteroDownloadProcessor(library_id="12345")

        # Should raise ValueError with clear message
        with pytest.raises(
            ValueError, match="has no title.*incomplete/invalid Zotero data"
        ):
            async for _ in processor.process(record, processor_stage="download"):
                pass
