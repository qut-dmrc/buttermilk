"""Tests for PDF size validation in Zotero downloader.

Ensures that PDFs smaller than 50KB are rejected as invalid downloads
and are not cached.

These are unit tests that directly test the size validation logic
without requiring full BM initialization.
"""

from pathlib import Path

import pytest

from buttermilk._core.exceptions import ProcessingError


def validate_pdf_size(pdf_path: Path, min_size_kb: int = 50) -> None:
    """Validate that a PDF meets minimum size requirements.

    This function will be implemented in buttermilk/libs/zotero.py

    Args:
        pdf_path: Path to the PDF file
        min_size_kb: Minimum size in kilobytes (default 50KB)

    Raises:
        ProcessingError: If PDF is smaller than minimum size
    """
    # This is a placeholder - the real implementation will be in zotero.py
    # For now, import it to make the test fail correctly
    from buttermilk.libs.zotero import validate_pdf_size as real_validate

    return real_validate(pdf_path, min_size_kb)


class TestPDFSizeValidation:
    """Test PDF size validation logic."""

    def test_pdf_smaller_than_50kb_raises_processing_error(self, tmp_path):
        """PDFs smaller than 50KB should raise ProcessingError.

        This prevents caching corrupt or placeholder PDFs.
        """
        # Create a tiny PDF (< 50KB)
        tiny_pdf = tmp_path / "tiny.pdf"
        tiny_pdf.write_bytes(b"%PDF-1.4\nfake content")  # Only ~20 bytes

        # Should raise ProcessingError
        with pytest.raises(ProcessingError) as exc_info:
            validate_pdf_size(tiny_pdf)

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert "50" in error_msg or "size" in error_msg.lower()
        assert "tiny.pdf" in error_msg or str(tiny_pdf) in error_msg

    def test_pdf_exactly_50kb_is_accepted(self, tmp_path):
        """PDFs exactly 50KB should be accepted (boundary test)."""
        # Create a PDF exactly 50KB
        valid_pdf = tmp_path / "valid.pdf"
        valid_pdf.write_bytes(b"x" * (50 * 1024))

        # Should NOT raise - should validate successfully
        validate_pdf_size(valid_pdf)  # No assertion - just shouldn't raise

    def test_pdf_larger_than_50kb_is_accepted(self, tmp_path):
        """PDFs larger than 50KB should be accepted."""
        # Create a PDF larger than 50KB
        large_pdf = tmp_path / "large.pdf"
        large_pdf.write_bytes(b"x" * (100 * 1024))  # 100KB

        # Should NOT raise
        validate_pdf_size(large_pdf)

    def test_nonexistent_pdf_raises_appropriate_error(self, tmp_path):
        """Nonexistent PDF should raise an appropriate error."""
        nonexistent = tmp_path / "does_not_exist.pdf"

        # Should raise an error (either FileNotFoundError or ProcessingError)
        with pytest.raises((FileNotFoundError, ProcessingError)):
            validate_pdf_size(nonexistent)

    def test_custom_minimum_size_threshold(self, tmp_path):
        """Test that custom minimum size thresholds work."""
        # Create a 30KB PDF
        pdf_30kb = tmp_path / "30kb.pdf"
        pdf_30kb.write_bytes(b"x" * (30 * 1024))

        # Should fail with 50KB threshold (default)
        with pytest.raises(ProcessingError):
            validate_pdf_size(pdf_30kb, min_size_kb=50)

        # Should pass with 20KB threshold
        validate_pdf_size(pdf_30kb, min_size_kb=20)
