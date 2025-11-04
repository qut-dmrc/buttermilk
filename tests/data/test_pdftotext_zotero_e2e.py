"""E2E test for pdftotext processor in Zotero pipeline.

This TRUE end-to-end test:
- Uses REAL Zotero API
- Downloads REAL PDFs
- Extracts text with REAL pdftotext
- Verifies the complete workflow

NO MOCKS of internal code.
"""

from pathlib import Path

import pytest


@pytest.mark.anyio
async def test_pdftotext_extracts_from_real_zotero_pdf(real_bm):
    """E2E test: Download PDF from Zotero and extract text with pdftotext.

    This test verifies:
    1. ZoteroDownloadProcessor downloads a real PDF
    2. PDFToTextProcessor extracts text using pdftotext
    3. Text extraction works with real academic PDFs
    4. Pipeline continues to downstream processors
    """
    import os

    from buttermilk.libs.zotero import ZoteroDownloadProcessor, ZoteroSource
    from buttermilk.processors.bash import PDFToTextProcessor

    # Get Zotero library ID from environment
    library_id = os.environ.get("ZOTERO_LIBRARY_ID")
    if not library_id:
        pytest.skip("ZOTERO_LIBRARY_ID not set")

    # Step 1: Fetch multiple items from Zotero (some may not have PDFs/titles)
    source = ZoteroSource(
        library_id=library_id,
        max_records=10,  # Try up to 10 to find one with a PDF
        force_full_sync=True,
    )

    records = [record async for record in source.fetch_items()]

    if not records:
        pytest.skip("No records returned from Zotero API")

    # Step 2: Try to download PDF from each record until we find one
    downloader = ZoteroDownloadProcessor(library_id=library_id)

    pdf_record = None
    for base_record in records:
        try:
            downloaded_records = [
                rec
                async for rec in downloader.process(
                    base_record,
                    processor_stage="download",
                )
            ]

            if downloaded_records:
                pdf_record = downloaded_records[0]
                break  # Found a valid PDF, stop looking
        except (ValueError, Exception) as e:
            # Skip records without titles or PDFs
            print(f"Skipping {base_record.record_id}: {e}")
            continue

    if not pdf_record:
        pytest.skip("No valid PDF found in first 10 Zotero items")

    # Verify PDF was downloaded
    assert pdf_record.file_path is not None
    pdf_path = Path(pdf_record.file_path)
    assert pdf_path.exists(), f"PDF not downloaded: {pdf_path}"
    assert pdf_path.suffix == ".pdf"

    # Get the content from pdfminer (baseline)
    pdfminer_content = pdf_record.content
    pdfminer_length = len(pdfminer_content) if pdfminer_content else 0

    # Step 3: Extract text with pdftotext
    pdftotext_processor = PDFToTextProcessor()

    pdftotext_records = [
        rec
        async for rec in pdftotext_processor.process(
            pdf_record,
            processor_stage="pdftotext",
        )
    ]

    assert len(pdftotext_records) == 1
    result = pdftotext_records[0]

    # Verify pdftotext extraction
    assert result.content is not None, "pdftotext returned no content"
    assert len(result.content) > 0, "pdftotext returned empty content"

    # Log comparison for manual verification
    pdftotext_length = len(result.content)
    print(f"\n{'='*60}")
    print("PDF Extraction Comparison")
    print(f"{'='*60}")
    print(f"Record ID: {result.record_id}")
    print(f"PDF: {pdf_path.name}")
    print(f"PDF size: {pdf_path.stat().st_size / 1024:.1f} KB")
    print(f"\npdfminer length: {pdfminer_length:,} chars")
    print(f"pdftotext length: {pdftotext_length:,} chars")
    print(f"Difference: {pdftotext_length - pdfminer_length:+,} chars")
    print("\nFirst 200 chars of pdftotext output:")
    print(result.content[:200])
    print(f"{'='*60}\n")

    # Basic sanity checks
    assert pdftotext_length > 100, "Extracted text seems too short"

    # If pdfminer also got content, compare (rough heuristic)
    if pdfminer_length > 100:
        # Both should extract similar amounts (within 50% difference)
        ratio = pdftotext_length / pdfminer_length
        assert 0.5 < ratio < 2.0, f"Extraction lengths differ significantly: {ratio:.2f}x"


@pytest.mark.anyio
async def test_pdftotext_handles_problematic_pdfs_gracefully(real_bm):
    """E2E test: Verify pdftotext handles PDFs that failed with pdfminer.

    This tests the small PDFs we identified earlier that caused issues.
    """
    from buttermilk._core.exceptions import ProcessingError
    from buttermilk._core.types import Record
    from buttermilk.processors.bash import PDFToTextProcessor

    # Use one of the tiny PDFs we found earlier
    tiny_pdfs = [
        "RE7UHJGI.pdf",  # 234 bytes - should fail size validation
        "6QBYSNHY.pdf",  # 7.3KB - should fail size validation
    ]

    cache_dir = Path.home() / ".cache/buttermilk/zotero"

    for pdf_name in tiny_pdfs:
        pdf_path = cache_dir / pdf_name
        if not pdf_path.exists():
            continue  # Skip if not in cache

        record = Record(
            record_id=pdf_name.replace(".pdf", ""),
            content="placeholder",
            file_path=str(pdf_path),
            metadata={"title": f"Test {pdf_name}"},
        )

        processor = PDFToTextProcessor()

        # Small/corrupt PDFs should either:
        # 1. Raise ProcessingError (preferred)
        # 2. Return minimal/empty text (acceptable)
        try:
            results = [r async for r in processor.process(record, processor_stage="test")]

            if results:
                result = results[0]
                # If it returns content, it should be minimal or indicate an error
                print(f"\n{pdf_name}: Returned {len(result.content)} chars")

                # Very small PDFs typically can't be extracted
                # pdftotext might return error text or empty string
                assert len(result.content) < 1000, "Unexpected: large content from tiny PDF"

        except ProcessingError as e:
            # Expected for corrupt PDFs
            print(f"\n{pdf_name}: ProcessingError (expected) - {str(e)[:100]}")
            assert "pdftotext" in str(e).lower() or "syntax error" in str(e).lower()


@pytest.mark.anyio
async def test_full_pipeline_with_pdftotext(real_bm):
    """E2E test: Run complete vectorization pipeline with pdftotext.

    This tests the full pipeline as configured in testing.yaml:
    1. Zotero source
    2. Download processor
    3. PDFToText processor (NEW!)
    4. Semantic splitter
    5. Embedding generator
    6. ChromaDB storage

    This is the ultimate integration test.
    """

    # Check if pdftotext is available
    import subprocess

    from hydra.utils import instantiate

    try:
        subprocess.run(["pdftotext", "-v"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("pdftotext not installed")

    # Get the pipeline config from real_bm
    pipeline_config = real_bm.conf.pipeline

    # Run pipeline with max 2 records (quick test)
    pipeline = instantiate(
        pipeline_config,
        max_records=2,
    )

    # Execute pipeline
    results = await pipeline.run()

    # Verify results
    assert results is not None
    print(f"\n{'='*60}")
    print("Pipeline E2E Test Results")
    print(f"{'='*60}")
    print(f"Records processed: {results.get('records_processed', 'unknown')}")
    print(f"Status: {results.get('status', 'unknown')}")
    if "errors" in results and results["errors"]:
        print(f"Errors: {results['errors']}")
    print(f"{'='*60}\n")

    # Basic assertions
    assert results.get("status") in ["completed", "success"], f"Pipeline failed: {results}"
    assert results.get("records_processed", 0) > 0, "No records were processed"
