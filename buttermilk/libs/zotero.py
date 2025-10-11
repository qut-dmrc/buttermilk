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
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.types import BaseRecord, Record
from buttermilk.storage.base import RecordFilter
from buttermilk.utils.utils import get_pdf_text


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

    def __aiter__(self):
        """Enable async iteration."""
        return self.fetch_items()

    async def fetch_items(self) -> AsyncGenerator[BaseRecord, None]:
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
        else:
            if self.start > 0:
                logger.info(f"🔄 Full sync of Zotero library starting from item {self.start}")
            else:
                logger.info("🔄 Full sync of Zotero library")

        # Track stats
        fetched_count = 0
        yielded_count = 0
        filtered_count = 0
        skipped_count = 0  # Track items skipped due to type filtering

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
                async def _fetch_items():
                    """Async wrapper for synchronous zot.items() call."""
                    return await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: list(self.zot.client.items(**page_params))
                    )

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

                # Create minimal BaseRecord with ID and metadata
                record = BaseRecord(
                    record_id=key,
                    metadata={
                        "zotero_item": item.get("data", {}),
                        "zotero_version": item.get("version", 0),
                        "zotero_links": item.get("links", {}),
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
            if page_size < api_params["limit"]:
                logger.debug(f"Last page received (only {page_size} items)")
                break

            # Move to next page
            start += page_size

        # Save sync state using library version (not max item version)
        # This is the correct approach per Zotero API docs
        timestamp = datetime.now(UTC).isoformat()
        library_version = self.zot.last_modified_version()
        self._save_sync_state(library_version, timestamp)

        # Log comprehensive summary
        logger.info(
            f"✅ Sync complete: {fetched_count} fetched from API, "
            f"{skipped_count} skipped (attachments/notes/annotations), "
            f"{filtered_count} filtered (already in vector store), "
            f"{yielded_count} yielded to pipeline (library version: {library_version})"
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

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        component_name: str = "ZoteroDownloadProcessor",
        cancellation_token: Any | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Record, None]:
        """Process a Zotero item: download and extract full text.

        Args:
            record: BaseRecord with Zotero item metadata

        Yields:
            Record: Full Record with extracted text content

        Raises:
            Exception: If download or extraction fails
        """
        # Extract Zotero metadata from record
        zotero_item = record.metadata.get("zotero_item", {})
        zotero_links = record.metadata.get("zotero_links", {})
        key = record.record_id
        title = zotero_item.get("title", "Unknown Title")
        doi_or_url = zotero_item.get("DOI") or zotero_item.get("url")

        # Define file paths using centralized cache
        cache_dir = self._get_cache_dir()
        pdf_file = cache_dir / f"{key}.pdf"
        json_file = cache_dir / f"{key}.json"

        # Check cache first
        if json_file.exists():
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    cached_item = json.load(f)
                if cached_item.get("content"):
                    logger.debug(f"✅ Loaded from cache: {key} '{title[:50]}'")
                    # Get links from cache if available
                    cached_links = cached_item.get("links", {})
                    yield Record(
                        record_id=key,
                        content=cached_item["content"],
                        file_path=pdf_file.as_posix(),
                        metadata={
                            "title": title,
                            "doi_or_url": doi_or_url,
                            "uri": json_file.as_posix(),
                            "zotero_data": zotero_item,
                            "zotero_links": cached_links,
                        },
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed to read cached item {key}: {e}")

        # Download full text or PDF
        content = None
        attachment = zotero_links.get("attachment", {})

        if attachment.get("attachmentType") == "application/pdf" and (pdf_href := attachment.get("href")):
            attachment_key = pdf_href.split("/")[-1]

            # Try full text from Zotero API first
            try:
                logger.debug(f"⬇️  Downloading full text for {key} '{title[:50]}'")
                fulltext = self.zot.fulltext_item(attachment_key)

                # Check if Zotero indexed enough pages (>90%)
                if fulltext and fulltext.get("indexedPages", 0) > 0 and fulltext["indexedPages"] >= (fulltext.get("totalPages", 0) * 0.9):
                    content = fulltext["content"]
                    logger.debug(f"Full text retrieved: {fulltext['indexedPages']}/{fulltext['totalPages']} pages")
            except zotero_errors.ResourceNotFoundError:
                logger.debug(f"Full text not available for {key}")
            except Exception as e:
                logger.warning(f"Error fetching full text for {key}: {e}")

            # Fall back to PDF download + extraction
            if not content:
                if not pdf_file.exists():
                    logger.debug(f"Downloading PDF for {key} to {pdf_file}")
                    # Zotero library is synchronous, don't try to async it
                    self.zot.dump(attachment_key, str(pdf_file))

                try:
                    content = get_pdf_text(pdf_file.as_posix())
                except Exception as e:
                    # PDF extraction failed - raise error
                    error_msg = f"Failed to extract text from PDF for {key}: {e}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

        else:
            # No PDF attachment found - skip this record (don't mark as failed)
            logger.debug(
                f"No PDF attachment found for {key} - skipping",
                key=key,
                title=title[:50] if title else "Unknown",
                doi_or_url=doi_or_url,
            )
            return  # Yield nothing - pipeline will mark as "skipped"

        # Save to cache
        cache_data = {
            "key": key,
            "data": zotero_item,
            "links": zotero_links,
            "content": content,
        }
        try:
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved to cache: {json_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")

        # Yield full Record
        logger.debug(f"✅ Download complete: {key} '{title[:50]}'")
        yield Record(
            record_id=key,
            content=content,
            file_path=pdf_file.as_posix(),
            metadata={
                "title": title,
                "doi_or_url": doi_or_url,
                "uri": json_file.as_posix(),
                "zotero_data": zotero_item,
                "zotero_links": zotero_links,
            },
        )
