"""Tests for Zotero citation key extraction.

Following CODE.md principles:
- Configuration from hydra yaml, NO hardcoded values
- Tests written BEFORE implementation (TDD)

Note: integration tests against a real Zotero library (ZoteroSource,
ZoteroDownloadProcessor, sync/metadata-update behavior) used to live here but
depended on the `zotero_vectorization` pipeline, which now lives in the
separate zotmcp repo, not buttermilk. Removed along with that pipeline.
"""

import pytest

pytest.importorskip("pyzotero", reason="pyzotero is optional (install with: uv sync --extra research)")

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
