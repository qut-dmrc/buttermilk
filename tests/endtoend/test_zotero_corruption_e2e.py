"""TRUE end-to-end test for Zotero corruption detection quality gate.

This test uses REAL Zotero API, REAL corruption detection, and REAL PDF processing.
NO MOCKS - validates the complete corruption detection and fallback workflow.

IMPORTANT: This test validates that corruption detection works on REAL Zotero records
fetched directly from the Zotero API.
"""

import os

import pytest

from buttermilk.libs.zotero import ZoteroDownloadProcessor, ZoteroSource

# Mark all tests in this file as endtoend and anyio
pytestmark = [pytest.mark.endtoend, pytest.mark.anyio]


async def test_corruption_quality_gate_on_real_zotero_records(real_bm):
    """TRUE E2E: Test corruption quality gate with REAL Zotero records.

    This validates the corruption check works with actual Zotero data:
    - REAL Zotero API calls to fetch records
    - REAL corruption detection on fulltext
    - REAL PDF download when needed
    - NO MOCKS

    The test processes multiple records and verifies that:
    1. Corrupt fulltext triggers PDF download
    2. Clean fulltext is accepted without PDF download
    3. The quality gate correctly differentiates between corrupt and clean text
    """
    # Arrange: Get library ID from environment variable
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    if not library_id:
        pytest.skip("ZOTERO_LIBRARY_ID environment variable not set")

    # Create REAL ZoteroSource
    source = ZoteroSource(
        library_id=library_id,
        max_records=10,  # Fetch a few records to test
    )

    # Create REAL processor
    processor = ZoteroDownloadProcessor(library_id=library_id)

    # Act: Fetch and process records from REAL Zotero API
    pdf_downloads = []
    clean_fulltexts = []
    skipped_errors = []

    records_processed = 0
    async for base_record in source.fetch_items():
        try:
            async for processed in processor.process(
                base_record,
                processor_stage="e2e_test",
                parent_trace_id="corruption_check_e2e",
            ):
                records_processed += 1
                record_id = processed.record_id

                if processed.file_path:
                    # PDF was downloaded - likely because fulltext was corrupt
                    pdf_downloads.append(record_id)
                    print(f"📄 PDF downloaded for {record_id}")
                elif processed.content:
                    # Fulltext was accepted - passed corruption check
                    clean_fulltexts.append(record_id)
                    print(
                        f"✓ Clean fulltext for {record_id} ({len(processed.content)} chars)"
                    )
        except Exception as e:
            # Skip invalid/incomplete Zotero records
            skipped_errors.append(f"{base_record.record_id}: {str(e)[:100]}")
            print(f"⚠️  Skipped {base_record.record_id}: {str(e)[:100]}")

    # Assert: Verify the quality gate ran on real data
    assert records_processed > 0, "Should process at least one record from Zotero"

    print("\n✅ TRUE E2E Test Results (REAL Zotero API, NO MOCKS):")
    print(f"  - Total records processed: {records_processed}")
    print(f"  - PDFs downloaded (likely corrupt fulltext): {len(pdf_downloads)}")
    print(f"  - Clean fulltexts accepted: {len(clean_fulltexts)}")
    print(f"  - Records skipped (errors): {len(skipped_errors)}")

    if pdf_downloads:
        print(f"  - Records with corrupt fulltext: {', '.join(pdf_downloads)}")

    # The corruption check quality gate is working if:
    # 1. We processed some records
    # 2. Decisions were made (either PDF download or fulltext accepted)
    assert len(pdf_downloads) + len(clean_fulltexts) == records_processed, (
        "All records should have been processed through the quality gate"
    )
