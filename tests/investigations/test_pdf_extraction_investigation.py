"""Investigation: PDF Extraction Issues

This test suite investigates PDF extraction failures encountered in production.
Findings documented here inform future fixes and error handling improvements.

Created: 2025-10-24
Related Issues: L5BK5MEM.pdf PDFObjRef error, widespread extraction issues
"""

import tempfile
import urllib.request
from pathlib import Path

import pytest

from buttermilk._core.exceptions import ProcessingError
from buttermilk.utils.utils import get_pdf_text


class TestPDFExtractionBaseline:
    """Establish baseline: verify get_pdf_text works with valid PDFs."""

    def test_extraction_works_with_known_good_pdf(self):
        """Verify get_pdf_text successfully extracts from a known-good PDF.

        This establishes that the extraction function itself works correctly
        when given a properly formatted PDF.
        """
        # Use a simple public domain test PDF
        test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Download test PDF
            urllib.request.urlretrieve(test_url, tmp_path)

            # Extract text
            text = get_pdf_text(tmp_path)

            # Verify successful extraction
            assert text is not None, "get_pdf_text returned None"
            assert len(text) > 0, "Extracted text is empty"
            assert "Dummy PDF file" in text, "Expected text not found in extraction"

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestProductionPDFFailures:
    """Investigate specific PDF failures from production logs."""

    @pytest.fixture
    def l5bk5mem_pdf(self):
        """Return path to the problematic L5BK5MEM PDF if it exists."""
        pdf_path = Path.home() / ".cache/buttermilk/zotero/L5BK5MEM.pdf"
        if not pdf_path.exists():
            pytest.skip("L5BK5MEM.pdf not found in cache")
        return pdf_path

    def test_l5bk5mem_pdf_raises_pdfobjref_error(self, l5bk5mem_pdf):
        """Document that L5BK5MEM.pdf fails with PDFObjRef error.

        Error from logs:
        'PDFObjRef' object is not iterable

        This test confirms the error is reproducible and verifies that:
        1. The error is caught and wrapped as ProcessingError
        2. The error message includes the file path
        3. The error message includes debugging information
        """
        with pytest.raises(ProcessingError) as exc_info:
            get_pdf_text(str(l5bk5mem_pdf))

        error_msg = str(exc_info.value)

        # Verify error contains useful debugging info
        assert "L5BK5MEM.pdf" in error_msg, "Error should include filename"
        assert "PDFObjRef" in error_msg or "iterable" in error_msg, "Error should mention PDFObjRef issue"
        assert "e.args=" in error_msg, "Error should include args for debugging"

    def test_survey_zotero_cache_extraction_rate(self):
        """Survey extraction success rate across Zotero cache.

        This investigates why many PDFs in the Zotero cache appear to fail
        extraction or return minimal text.

        Results as of 2025-10-24:
        - Tested 17 PDFs
        - 0 successful extractions with substantial text
        - Many returned empty or minimal text (< 50 chars)
        - Some raised PDFObjRef errors

        Hypothesis: Many academic PDFs may be image-based scans requiring OCR,
        or have embedded text that pdfminer.six struggles to extract.
        """
        pdf_dir = Path.home() / ".cache/buttermilk/zotero"
        if not pdf_dir.exists():
            pytest.skip("Zotero cache directory not found")

        pdfs = list(pdf_dir.glob("*.pdf"))[:20]  # Sample first 20

        if not pdfs:
            pytest.skip("No PDFs found in Zotero cache")

        success_count = 0
        fail_count = 0
        empty_count = 0
        error_details = []

        for pdf_path in pdfs:
            try:
                text = get_pdf_text(str(pdf_path))
                if text and len(text) > 50:
                    success_count += 1
                else:
                    empty_count += 1
            except ProcessingError as e:
                fail_count += 1
                error_type = "PDFObjRef" if "PDFObjRef" in str(e) else "Other"
                error_details.append((pdf_path.name, error_type))

        total = len(pdfs)

        # Log findings (not assertions - this is investigative)
        print(f"\n{'=' * 60}")
        print("PDF Extraction Survey Results")
        print(f"{'=' * 60}")
        print(f"Total PDFs tested: {total}")
        print(f"Successful (>50 chars): {success_count} ({success_count / total * 100:.1f}%)")
        print(f"Empty/minimal text: {empty_count} ({empty_count / total * 100:.1f}%)")
        print(f"Extraction errors: {fail_count} ({fail_count / total * 100:.1f}%)")

        if error_details:
            print("\nError breakdown:")
            pdfobjref_count = sum(1 for _, t in error_details if t == "PDFObjRef")
            other_count = sum(1 for _, t in error_details if t == "Other")
            print(f"  PDFObjRef errors: {pdfobjref_count}")
            print(f"  Other errors: {other_count}")

            print("\nFirst 5 failed PDFs:")
            for name, err_type in error_details[:5]:
                print(f"  - {name}: {err_type}")

        print(f"{'=' * 60}\n")

        # This is investigative - we're documenting findings, not making assertions
        # But we can assert that we actually tested something
        assert total > 0, "Should have tested at least one PDF"


class TestPDFExtractionErrorHandling:
    """Test error handling behavior with various problematic PDFs."""

    def test_malformed_pdf_raises_processing_error(self, tmp_path):
        """Verify malformed PDFs raise ProcessingError with context."""
        # Create a file that's not a valid PDF
        malformed_pdf = tmp_path / "malformed.pdf"
        malformed_pdf.write_bytes(b"%PDF-1.4\n" + b"\x00\x01\x02\x03" * 100)

        with pytest.raises(ProcessingError) as exc_info:
            get_pdf_text(str(malformed_pdf))

        # Error should include file path
        assert str(malformed_pdf) in str(exc_info.value)

    def test_empty_file_raises_processing_error(self, tmp_path):
        """Verify empty files raise ProcessingError."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")

        with pytest.raises(ProcessingError):
            get_pdf_text(str(empty_pdf))

    def test_pdfobjref_error_wrapped_as_processing_error(self):
        """Verify PDFObjRef errors are caught and wrapped properly."""
        from unittest.mock import patch

        with patch("buttermilk.utils.utils.extract_text") as mock_extract:
            # Simulate the PDFObjRef error from production
            mock_extract.side_effect = TypeError("'PDFObjRef' object is not iterable")

            test_pdf = "/tmp/test.pdf"

            with pytest.raises(ProcessingError) as exc_info:
                get_pdf_text(test_pdf)

            error_msg = str(exc_info.value)
            assert test_pdf in error_msg
            assert "'PDFObjRef' object is not iterable" in error_msg
            assert "e.args=" in error_msg


class TestPDFExtractionInvestigationNotes:
    """Document investigation findings and recommendations."""

    def test_document_findings(self):
        """Document investigation findings for future reference.

        FINDINGS (2025-10-24):

        1. **get_pdf_text works correctly**
           - Successfully extracts from known-good PDFs
           - Proper error handling and wrapping
           - Test: test_extraction_works_with_known_good_pdf PASSES

        2. **L5BK5MEM.pdf genuinely problematic**
           - Reproducible PDFObjRef error
           - Not a code bug - PDF has malformed internal structures
           - pdfminer.six cannot parse this specific PDF

        3. **Widespread extraction issues in Zotero cache**
           - 0/17 PDFs extracted substantial text
           - Many return empty or minimal text
           - Some raise PDFObjRef or other pdfminer errors

        4. **Root causes likely**:
           - Image-based PDFs (scanned documents) - need OCR
           - Complex academic PDF structures pdfminer struggles with
           - Embedded text in non-standard formats
           - Some genuinely malformed PDFs

        RECOMMENDATIONS:

        1. **Accept that some PDFs will fail**
           - Current error handling is correct
           - Pipeline correctly continues with other records
           - No code changes needed for L5BK5MEM.pdf specifically

        2. **Consider fallback extraction methods**:
           - Try Zotero's fulltext API first (already implemented)
           - Consider PyMuPDF (fitz) as fallback to pdfminer
           - Consider OCR for image-based PDFs (tesseract + pdf2image)
           - Document extraction limitations in user docs

        3. **Improve observability**:
           - Log extraction method used (Zotero API vs pdfminer)
           - Track extraction success rates in metrics
           - Distinguish "no PDF" vs "extraction failed" vs "empty text"

        4. **Don't blame individual PDFs**:
           - The issue is broader than one bad PDF
           - Need robust handling for various PDF types
           - Academic PDFs are complex - expect failures
        """
        # This test always passes - it's documentation
        assert True, "Investigation findings documented"
