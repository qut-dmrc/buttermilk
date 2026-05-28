"""Test PDF extraction error handling.

This test reproduces the PDFObjRef error encountered with Zotero item L5BK5MEM.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from buttermilk._core.exceptions import ProcessingError
from buttermilk.utils.utils import get_pdf_text


class TestPDFExtractionErrors:
    """Test error handling in PDF text extraction."""

    def test_pdf_objref_error_is_caught_and_reraised_as_processing_error(self):
        """Test that PDFObjRef iteration errors are caught and re-raised as ProcessingError."""
        # Mock the pdfminer extract_text to raise the error we saw in production
        with patch("buttermilk.utils.utils.extract_text") as mock_extract:
            # Simulate the exact error from the logs
            mock_extract.side_effect = TypeError("'PDFObjRef' object is not iterable")

            # Call get_pdf_text with a dummy file path
            test_pdf = "/tmp/test_pdf_objref.pdf"

            # Should raise ProcessingError with the original error message
            with pytest.raises(ProcessingError) as exc_info:
                get_pdf_text(test_pdf)

            # Verify the error message contains both the path and the original error
            assert test_pdf in str(exc_info.value)
            assert "'PDFObjRef' object is not iterable" in str(exc_info.value)

    def test_pdf_extraction_with_malformed_pdf(self, tmp_path):
        """Test PDF extraction with a malformed PDF file."""
        # Create a malformed PDF (just random bytes)
        malformed_pdf = tmp_path / "malformed.pdf"
        malformed_pdf.write_bytes(b"%PDF-1.4\n" + b"\x00\x01\x02\x03" * 100)

        # Should raise ProcessingError
        with pytest.raises(ProcessingError) as exc_info:
            get_pdf_text(str(malformed_pdf))

        # Error message should mention the file path
        assert str(malformed_pdf) in str(exc_info.value)

    def test_pdf_extraction_with_empty_file(self, tmp_path):
        """Test PDF extraction with an empty file."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")

        # Should raise ProcessingError
        with pytest.raises(ProcessingError):
            get_pdf_text(str(empty_pdf))

    def test_pdf_extraction_error_includes_full_context(self):
        """Test that PDF extraction errors include full context for debugging."""
        with patch("buttermilk.utils.utils.extract_text") as mock_extract:
            # Create a mock exception with args
            error = TypeError("'PDFObjRef' object is not iterable")
            error.args = ("'PDFObjRef' object is not iterable",)
            mock_extract.side_effect = error

            test_pdf = "/test/path.pdf"

            with pytest.raises(ProcessingError) as exc_info:
                get_pdf_text(test_pdf)

            error_msg = str(exc_info.value)
            # Should include file path
            assert test_pdf in error_msg
            # Should include error message
            assert "PDFObjRef" in error_msg
            # Should include error args for debugging
            assert "e.args=" in error_msg


class TestPDFExtractionWithRealFile:
    """Test PDF extraction with actual PDF file if L5BK5MEM.pdf is available."""

    @pytest.fixture
    def l5bk5mem_pdf(self):
        """Return path to the problematic PDF if it exists."""
        pdf_path = Path.home() / ".cache/buttermilk/zotero/L5BK5MEM.pdf"
        if pdf_path.exists():
            return pdf_path
        pytest.skip("L5BK5MEM.pdf not found in cache")

    def test_l5bk5mem_pdf_extraction_succeeds(self, l5bk5mem_pdf):
        """Test that L5BK5MEM.pdf can be extracted successfully.

        This PDF previously triggered a PDFObjRef error, but the extraction
        code has since been fixed to handle it.
        """
        text = get_pdf_text(str(l5bk5mem_pdf))
        assert text is not None
        assert len(text) > 0
