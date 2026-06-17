"""Clean separation of Zotero source and processing logic.

This module provides:
- ZoteroSource: Yields BaseRecord objects with Zotero item IDs and metadata
- ZoteroDownloadProcessor: Downloads PDFs and extracts full text for each item
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr
from pyzotero import zotero, zotero_errors

from buttermilk import bm, logger
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.types import BaseRecord, Record
from buttermilk.storage.base import RecordFilter
from buttermilk.utils.text_quality import detect_text_corruption


def extract_citation_key(extra_field: str | None) -> str | None:
    """Extract BetterBibTeX citation key from Zotero 'extra' field.

    Looks for pattern 'Citation Key: <key>' on its own line in the extra field.

    Args:
        extra_field: String content of Zotero item's 'extra' field, or None

    Returns:
        Citation key string if found, None otherwise

    Examples:
        >>> extract_citation_key("Citation Key: suzor2019digital")
        'suzor2019digital'

        >>> extract_citation_key("Publisher: Routledge\\nCitation Key: abbate2017")
        'abbate2017'

        >>> extract_citation_key("No key here")
        None
    """
    if not extra_field:
        return None

    # Split by newlines and check each line
    for line in extra_field.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Citation Key:"):
            # Extract everything after the colon and strip whitespace
            key = stripped.split(":", 1)[1].strip()
            return key if key else None

    return None


def validate_pdf_size(pdf_path: Path, min_size_kb: int = 50) -> None:
    """Validate that a PDF meets minimum size requirements.

    This prevents caching corrupt, placeholder, or incomplete PDF downloads.
    PDFs smaller than the threshold are typically:
    - Corrupted downloads (< 1KB)
    - Placeholder files from Zotero (< 10KB)
    - Incomplete downloads (variable size)

    Args:
        pdf_path: Path to the PDF file to validate
        min_size_kb: Minimum acceptable size in kilobytes (default: 50KB)

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ProcessingError: If PDF is smaller than minimum size

    Examples:
        >>> validate_pdf_size(Path("document.pdf"))  # 100KB file - passes
        >>> validate_pdf_size(Path("tiny.pdf"))      # 10KB file - raises ProcessingError
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Get file size in bytes
    file_size_bytes = pdf_path.stat().st_size
    file_size_kb = file_size_bytes / 1024

    # Check minimum size
    if file_size_kb < min_size_kb:
        raise ProcessingError(
            f"PDF {pdf_path.name} is too small ({file_size_kb:.1f} KB). "
            f"Minimum size is {min_size_kb} KB. "
            f"This likely indicates a corrupted or incomplete download. "
            f"Skipping this PDF."
        )

    logger.debug(f"PDF size validation passed: {pdf_path.name} ({file_size_kb:.1f} KB)")


class VectorStoreExistenceFilter(RecordFilter):
    """Filter that checks if a record already exists in a vector store."""

    def __init__(self, vector_store: Any):
        """Initialize with a vector store that has check_document_exists() method.

        Args:
            vector_store: Vector store with check_document_exists(record_id: str) -> bool
        """
        self.vector_store = vector_store
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure vector store cache is initialized (lazy, once-only)."""
        if self._initialized:
            return

        # Initialize cache if vector store has ensure_cache_initialized method
        if hasattr(self.vector_store, "ensure_cache_initialized"):
            await self.vector_store.ensure_cache_initialized()

        self._initialized = True

    async def should_include(self, record: BaseRecord) -> bool:
        """Check if record should be included (i.e., does NOT already exist).

        Args:
            record: Record to check

        Returns:
            True if record should be included (doesn't exist), False to skip (exists)
        """
        if not hasattr(record, "record_id"):
            return True

        # Ensure cache is initialized before checking
        await self._ensure_initialized()

        # Synchronous check - vector_store.check_document_exists is sync
        exists = self.vector_store.check_document_exists(record.record_id)
        return not exists


class ZoteroSource(BaseModel):
    """Source that yields BaseRecord objects with Zotero item IDs and metadata.

    This source:
    - Fetches items from Zotero API in strict ascending date order
    - Tracks version state for incremental sync
    - Filters items using the RecordFilter protocol
    - Yields minimal BaseRecord objects (ID + metadata only)

    The heavy lifting (downloads, extraction) is delegated to ZoteroDownloadProcessor.
    """

    model_config = {"arbitrary_types_allowed": True}

    library_id: str = Field(..., description="Zotero library ID")
    filter: Any = Field(default=None, description="Optional RecordFilter implementation")
    force_full_sync: bool = Field(default=False, description="Bypass incremental sync")
    start: int = Field(default=0, description="Item offset to start fetching from (0-based)")
    max_records: int | None = Field(default=None, description="Maximum items to yield")

    _zot: RetryWrapper | None = PrivateAttr(default=None)

    @property
    def zot(self) -> RetryWrapper:
        """Lazily initialize the Zotero client wrapped in RetryWrapper."""
        if self._zot is None:
            from buttermilk._core.constants import cache

            zot_client = zotero.Zotero(
                library_id=self.library_id,
                library_type="group",
                api_key=bm.credentials.get("ZOTERO_API_KEY"),
            )
            self._zot = RetryWrapper(
                client=zot_client,
                max_retries=3,
                min_wait_seconds=5.0,
                max_wait_seconds=60.0,
                jitter_seconds=5.0,
            )
            # Ensure cache directory exists
            bm.session_info.get_cache_subdir(cache.ZOTERO, create=True)
        return self._zot

    def _state_file_path(self) -> Path:
        """Get path to sync state file in centralized cache."""
        from buttermilk._core.constants import cache

        return bm.session_info.get_cache_subdir(cache.ZOTERO) / ".zotero_sync_state.json"

    def _load_sync_state(self) -> dict[str, Any]:
        """Load last sync state.

        Returns:
            dict: {'last_version': int|None, 'last_sync_timestamp': str|None}
        """
        state_file = self._state_file_path()
        if not state_file.exists():
            return {"last_version": None, "last_sync_timestamp": None}

        try:
            with state_file.open("r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading sync state: {e}. Starting fresh.")
            return {"last_version": None, "last_sync_timestamp": None}

    def _save_sync_state(self, version: int, timestamp: str) -> None:
        """Save current sync state.

        Args:
            version: Last processed item version
            timestamp: ISO format timestamp
        """
        state = {"last_version": version, "last_sync_timestamp": timestamp}
        try:
            with self._state_file_path().open("w") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved sync state: version={version}")
        except OSError as e:
            logger.error(f"Failed to save sync state: {e}")

    def __aiter__(self) -> AsyncGenerator[BaseRecord, None]:
        """Enable async iteration."""
        return self.fetch_items()

    async def fetch_items(self) -> AsyncGenerator[BaseRecord, None]:  # noqa: PLR0912 - Complex Zotero API logic, authorized by NS 20251024
        """Fetch Zotero items and yield BaseRecord objects with IDs and metadata.

        This method:
        - Implements incremental sync using version tracking
        - Returns items in strict ascending date order
        - Applies filters before yielding
        - Tracks the highest version processed for next sync

        Yields:
            BaseRecord: Minimal records with record_id and metadata
        """
        # Load sync state
        sync_state = self._load_sync_state()
        last_version = sync_state["last_version"]

        # Prepare API parameters
        api_params = {
            "itemType": "-attachment",  # Exclude attachments
            "limit": 100,
            "sort": "dateModified",
            "direction": "asc",  # Ascending order for incremental sync
        }

        # Add incremental sync parameter
        if not self.force_full_sync and last_version is not None:
            api_params["since"] = last_version
            logger.info(f"🔄 Incremental sync from version {last_version}")
        elif self.start > 0:
            logger.info(f"🔄 Full sync of Zotero library starting from item {self.start}")
        else:
            logger.info("🔄 Full sync of Zotero library")

        # Track stats
        fetched_count = 0
        yielded_count = 0
        filtered_count = 0
        skipped_count = 0  # Track items skipped due to type filtering

        # Track versions for safe incremental sync
        # We'll save the second-highest to avoid missing items at version boundaries
        seen_versions: list[int] = []

        # Manual pagination to maintain async control and avoid blocking
        # Fetch items page by page (100 items per page)
        start = self.start
        page_num = 0

        while True:
            page_num += 1

            # Add pagination parameters
            page_params = {**api_params, "start": start}

            try:
                # Fetch one page of results with retry logic
                # Use RetryWrapper's _execute_with_retry with async wrapper for sync function
                async def _fetch_items() -> list[dict[str, Any]]:
                    """Async wrapper for synchronous zot.items() call."""
                    return await asyncio.get_event_loop().run_in_executor(None, lambda: list(self.zot.client.items(**page_params)))

                items = await self.zot._execute_with_retry(_fetch_items)
                page_size = len(items)

                logger.info(
                    f"📥 Fetching from Zotero API, page {page_num}: {page_size} items (start={start}, fetched={fetched_count})",
                    fetched_count=fetched_count,
                    filtered_count=filtered_count,
                    yielded_count=yielded_count,
                    skipped_count=skipped_count,
                    start=start,
                    page_num=page_num,
                    page_size=page_size,
                )

                # If no items, we're done
                if not items:
                    break

            except Exception as e:
                # RetryWrapper already attempted retries, so if we're here, all retries failed
                logger.error(f"All retry attempts failed for Zotero API (page {page_num}, start={start}): {e}")
                raise  # Re-raise to fail the pipeline (don't silently continue with partial results)

            # Process items from this page
            for item in items:
                fetched_count += 1

                # Stop if we hit max_records
                if self.max_records is not None and yielded_count >= self.max_records:
                    logger.info(f"Reached max_records limit ({self.max_records})")
                    break

                # Skip invalid items
                key = item.get("key")
                if not key:
                    logger.warning(f"Item missing key: {item.get('data', {}).get('title', 'N/A')}")
                    continue

                item_type = item.get("data", {}).get("itemType")
                if item_type in {"attachment", "note", "annotation"}:
                    skipped_count += 1
                    continue

                # Extract citation key from 'extra' field
                zotero_data = item.get("data", {})
                citation_key = extract_citation_key(zotero_data.get("extra"))

                # Track item version for safe sync state
                item_version = item.get("version", 0)
                seen_versions.append(item_version)

                # Create minimal BaseRecord with ID and metadata
                record = BaseRecord(
                    record_id=key,
                    metadata={
                        "zotero_item": zotero_data,
                        "zotero_version": item_version,
                        "zotero_links": item.get("links", {}),
                        "citation_key": citation_key,  # Add citation_key to metadata
                    },
                )

                # Apply filter
                if self.filter and not await self.filter.should_include(record):
                    filtered_count += 1
                    continue

                yielded_count += 1
                yield record

            # Check if we should continue to next page
            if self.max_records is not None and yielded_count >= self.max_records:
                break

            # If we got fewer items than the limit, we're done
            limit = api_params["limit"]
            assert isinstance(limit, int), "limit must be an int"
            if page_size < limit:
                logger.debug(f"Last page received (only {page_size} items)")
                break

            # Move to next page
            start += page_size

        # Save sync state using second-highest version to avoid missing boundary items
        # CRITICAL: If we save the highest version and sync gets interrupted,
        # next sync with since=highest might miss items at that exact version
        timestamp = datetime.now(UTC).isoformat()

        # Calculate safe version to save
        if seen_versions:
            unique_versions = sorted(set(seen_versions), reverse=True)
            if len(unique_versions) >= 2:
                # Use second-highest version for safety
                safe_version = unique_versions[1]
                logger.debug(
                    f"Saving second-highest version {safe_version} (highest was {unique_versions[0]}, {len(unique_versions)} unique versions)"
                )
            else:
                # Only one unique version - use it but log warning
                safe_version = unique_versions[0]
                logger.warning(f"Only one unique version ({safe_version}) seen during sync. Cannot use second-highest for safety.")
        else:
            # No items processed - use library version from API
            safe_version = self.zot.last_modified_version()
            logger.debug(f"No items processed, using library version {safe_version}")

        self._save_sync_state(safe_version, timestamp)

        # Log comprehensive summary
        logger.info(
            f"✅ Sync complete: {fetched_count} fetched from API, "
            f"{skipped_count} skipped (attachments/notes/annotations), "
            f"{filtered_count} filtered (already in vector store), "
            f"{yielded_count} yielded to pipeline (saved version: {safe_version})"
        )


class ZoteroDownloadProcessor(BaseModel):
    """Processor that downloads PDFs and extracts full text from Zotero items.

    This processor:
    - Takes a BaseRecord with Zotero item metadata
    - Downloads full text from Zotero API if available
    - Falls back to PDF download + extraction if needed
    - Yields a full Record with content, or raises an error if download fails
    """

    model_config = {"arbitrary_types_allowed": True}

    library_id: str = Field(..., description="Zotero library ID")

    _zot: RetryWrapper | None = PrivateAttr(default=None)

    @property
    def zot(self) -> RetryWrapper:
        """Lazily initialize the Zotero client wrapped in RetryWrapper."""
        if self._zot is None:
            from buttermilk._core.constants import cache

            zot_client = zotero.Zotero(
                library_id=self.library_id,
                library_type="group",
                api_key=bm.credentials.get("ZOTERO_API_KEY"),
            )
            self._zot = RetryWrapper(
                client=zot_client,
                max_retries=3,
                min_wait_seconds=5.0,
                max_wait_seconds=60.0,
                jitter_seconds=5.0,
            )
            # Ensure cache directory exists
            bm.session_info.get_cache_subdir(cache.ZOTERO, create=True)
        return self._zot

    def _get_cache_dir(self) -> Path:
        """Get the centralized Zotero cache directory."""
        from buttermilk._core.constants import cache

        return bm.session_info.get_cache_subdir(cache.ZOTERO)

    async def process(self, context: ProcessingContext) -> AsyncGenerator[Record, None]:
        """Process a Zotero item: download and extract full text.

        Pipeline caching: The pipeline's RecordCache handles caching of Records.
        This processor is only invoked when no cached Record exists.

        Args:
            context: ProcessingContext with record containing Zotero item metadata

        Yields:
            Record: Full Record with extracted text content

        Raises:
            Exception: If download or extraction fails
        """
        record = context.record
        # Extract Zotero metadata from record
        zotero_item = record.metadata.get("zotero_item", {})
        zotero_links = record.metadata.get("zotero_links", {})
        citation_key = record.metadata.get("citation_key")
        key = record.record_id
        title = zotero_item.get("title")

        # FAIL FAST: Title is required
        if not title:
            raise ValueError(
                f"Zotero item {key} has no title. "
                f"Item type: {zotero_item.get('itemType')}. "
                f"This indicates incomplete/invalid Zotero data that must be fixed upstream."
            )

        doi_or_url = zotero_item.get("DOI") or zotero_item.get("url")

        # Define file paths for Zotero-specific assets (PDFs, metadata)
        # Note: These are NOT cache - they're source assets needed by downstream processors
        cache_dir = self._get_cache_dir()
        pdf_file = cache_dir / f"{key}.pdf"

        # Download full text or PDF
        content = None
        pdf_downloaded = False  # Track if we actually downloaded a PDF
        attachment = zotero_links.get("attachment", {})

        if attachment.get("attachmentType") == "application/pdf" and (pdf_href := attachment.get("href")):
            attachment_key = pdf_href.split("/")[-1]
            have_fulltext = False  # Track if we got fulltext from Zotero API

            # Try full text from Zotero API first
            try:
                logger.debug(f"⬇️  Downloading full text for {key} '{title[:50]}'")
                fulltext = self.zot.fulltext_item(attachment_key)

                # Check if Zotero indexed enough pages (>=50% minimum threshold)
                indexed_pages = fulltext.get("indexedPages", 0)
                total_pages = fulltext.get("totalPages", 0)

                if fulltext and indexed_pages > 0 and total_pages > 0:
                    index_ratio = indexed_pages / total_pages
                    if index_ratio >= 0.5:  # At least 50% of pages indexed
                        content = fulltext["content"]

                        # QUALITY GATE: Check for text corruption
                        corruption_result = detect_text_corruption(content)
                        if corruption_result["is_corrupted"]:
                            logger.warning(
                                f"Zotero fulltext for {key} is corrupt: "
                                f"{corruption_result['corruption_percentage']:.1f}% corruption, "
                                f"CID count: {corruption_result['cid_count']}, "
                                f"newline ratio: {corruption_result['newline_ratio']:.1f}%, "
                                f"avg line length: {corruption_result['avg_line_length']:.1f}. "
                                f"Will download and extract from PDF instead."
                            )
                            have_fulltext = False  # Trigger PDF fallback
                            content = None  # Clear corrupt content
                        else:
                            have_fulltext = True
                            logger.debug(
                                f"Full text retrieved and validated: {indexed_pages}/{total_pages} pages ({index_ratio:.1%} indexed), "
                                f"corruption: {corruption_result['corruption_percentage']:.1f}%"
                            )
                    else:
                        logger.debug(f"Full text incomplete: {indexed_pages}/{total_pages} pages ({index_ratio:.1%} indexed) - will download PDF")
                else:
                    logger.debug(f"Full text metadata invalid (indexed={indexed_pages}, total={total_pages}) - will download PDF")
            except zotero_errors.ResourceNotFoundError:
                logger.debug(f"Full text not available for {key}")
            except Exception as e:
                logger.warning(f"Error fetching full text for {key}: {e}")

            # Fall back to PDF download + extraction
            # CRITICAL FIX: Only download PDF if we DON'T have fulltext
            # This prevents wasting bandwidth and prevents PDFToTextProcessor from overwriting good content
            if not have_fulltext:
                logger.debug(f"No fulltext from Zotero API, will download PDF for {key}")
                if not pdf_file.exists():
                    logger.debug(f"Downloading PDF for {key} to {pdf_file}")
                    # Zotero library is synchronous, don't try to async it
                    self.zot.dump(attachment_key, str(pdf_file))

                # Validate PDF size before attempting extraction
                # This fails fast on corrupt/placeholder downloads (< 50KB)
                try:
                    validate_pdf_size(pdf_file)
                except ProcessingError as e:
                    # PDF too small - delete it and raise error to skip this record
                    logger.error(f"PDF validation failed for {key}: {e}")
                    pdf_file.unlink(missing_ok=True)  # Don't keep invalid PDFs
                    raise  # Re-raise to skip this record

                pdf_downloaded = True  # We have a PDF file
                # Note: Text extraction now handled by PDFToTextProcessor in pipeline
                # This allows using pdftotext instead of pdfminer
                logger.debug(f"PDF downloaded successfully: {key}. Text extraction will be done by downstream processor.")

                # Set meaningful PDF metadata as content to satisfy Record contract
                # This will be replaced by PDFToTextProcessor with actual extracted text
                if not content:
                    pdf_size = pdf_file.stat().st_size
                    content = f"[PDF Document: {pdf_file.name}, Size: {pdf_size:,} bytes, Path: {pdf_file.as_posix()}]"
                    logger.debug(f"Set PDF metadata as content for {key}: {content}")
            else:
                # have_fulltext is True here, which is only set alongside a non-None content (see above).
                assert content is not None
                logger.debug(f"Skipping PDF download for {key} - already have fulltext with {len(content)} chars")

        else:
            # No PDF attachment found - skip this record (don't mark as failed)
            logger.debug(
                f"No PDF attachment found for {key} - skipping",
                key=key,
                title=title[:50] if title else "Unknown",
                doi_or_url=doi_or_url,
            )
            return  # Yield nothing - pipeline will mark as "skipped"

        # Yield full Record
        # Pipeline's RecordCache will handle caching automatically
        logger.debug(f"✅ Download complete: {key} '{title[:50]}'")
        # CRITICAL FIX: Only set file_path if we actually downloaded a PDF
        # This prevents PDFToTextProcessor from running when we already have fulltext
        record_file_path = pdf_file.as_posix() if pdf_downloaded else None

        yield Record(
            record_id=key,
            content=content,
            file_path=record_file_path,  # Only set if PDF was downloaded
            metadata={
                "title": title,
                "doi_or_url": doi_or_url,
                "zotero_data": zotero_item,
                "zotero_links": zotero_links,
                "citation_key": citation_key,
            },
        )


class ZoteroItemSource(BaseModel):
    """Source that yields BaseRecord objects for specific Zotero items by key.

    This source:
    - Reads item keys from a text file (one key per line)
    - Fetches each item individually from Zotero API
    - Yields minimal BaseRecord objects (ID + metadata only)

    Unlike ZoteroSource which iterates through all items, this fetches
    only specific items by key. Useful for reprocessing corrupt documents
    or processing a curated list of items.

    Usage in config:
        source:
          _target_: buttermilk.libs.zotero.ZoteroItemSource
          library_id: ${oc.env:ZOTERO_LIBRARY_ID}
          item_keys_file: corrupt_documents_66pct.txt
    """

    model_config = {"arbitrary_types_allowed": True}

    library_id: str = Field(..., description="Zotero library ID")
    item_keys_file: str = Field(..., description="Path to text file with item keys (one per line)")
    item_keys: list[str] = Field(default_factory=list, description="Loaded item keys")

    _zot: RetryWrapper | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        """Initialize and load item keys from file."""
        super().__init__(**data)

        # Load item keys from file
        keys_file = Path(self.item_keys_file)
        if not keys_file.exists():
            raise FileNotFoundError(f"Item keys file not found: {keys_file}")

        # Read keys, strip whitespace, skip empty lines
        keys_text = keys_file.read_text()
        self.item_keys = [line.strip() for line in keys_text.split("\n") if line.strip()]

        logger.info(f"Loaded {len(self.item_keys)} item keys from {keys_file}")

    @property
    def zot(self) -> RetryWrapper:
        """Lazily initialize the Zotero client wrapped in RetryWrapper."""
        if self._zot is None:
            from buttermilk._core.constants import cache

            zot_client = zotero.Zotero(
                library_id=self.library_id,
                library_type="group",
                api_key=bm.credentials.get("ZOTERO_API_KEY"),
            )
            self._zot = RetryWrapper(
                client=zot_client,
                max_retries=3,
                min_wait_seconds=5.0,
                max_wait_seconds=60.0,
                jitter_seconds=5.0,
            )
            # Ensure cache directory exists
            bm.session_info.get_cache_subdir(cache.ZOTERO, create=True)
        return self._zot

    def __aiter__(self) -> AsyncGenerator[BaseRecord, None]:
        """Enable async iteration."""
        return self.fetch_items()

    async def fetch_items(self) -> AsyncGenerator[BaseRecord, None]:
        """Fetch specific Zotero items by key and yield BaseRecord objects.

        This method:
        - Fetches each item individually using pyzotero's item() method
        - Skips items that don't exist or are invalid types
        - Returns items in the order specified in the keys file

        Yields:
            BaseRecord: Minimal records with record_id and metadata
        """
        yielded_count = 0
        skipped_count = 0

        for key in self.item_keys:
            try:
                # Fetch single item with retry logic
                async def _fetch_item() -> dict[str, Any]:
                    """Async wrapper for synchronous zot.item() call."""
                    return await asyncio.get_event_loop().run_in_executor(None, lambda: self.zot.client.item(key))

                item = await self.zot._execute_with_retry(_fetch_item)

                # Skip invalid items
                if not item:
                    logger.warning(f"Item {key} not found, skipping")
                    skipped_count += 1
                    continue

                item_type = item.get("data", {}).get("itemType")
                if item_type in {"attachment", "note", "annotation"}:
                    logger.debug(f"Skipping {item_type} item: {key}")
                    skipped_count += 1
                    continue

                # Extract citation key from 'extra' field
                zotero_data = item.get("data", {})
                citation_key = extract_citation_key(zotero_data.get("extra"))

                # Track item version
                item_version = item.get("version", 0)

                # Create minimal BaseRecord with ID and metadata
                record = BaseRecord(
                    record_id=key,
                    metadata={
                        "zotero_item": zotero_data,
                        "zotero_version": item_version,
                        "zotero_links": item.get("links", {}),
                        "citation_key": citation_key,
                    },
                )

                yielded_count += 1
                logger.debug(f"Yielded item {key}: {zotero_data.get('title', 'N/A')[:50]}")
                yield record

            except Exception as e:
                logger.error(f"Error fetching item {key}: {e}")
                skipped_count += 1
                continue

        logger.info(f"✅ Fetch complete: {yielded_count} items yielded, {skipped_count} skipped")
