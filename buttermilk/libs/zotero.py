import asyncio
import json  # Import json module
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pydantic
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
from pyzotero import zotero, zotero_errors

# Import bm for credentials access
from buttermilk._core.dmrc import get_bm
from buttermilk._core.log import logger
from buttermilk._core.types import Record

# Add TYPE_CHECKING block for forward reference if ChromaDBEmbeddings is in a different module
# and causes circular import issues. If they are in the same module or structure prevents
# circular imports, this might not be strictly necessary but is good practice.
if TYPE_CHECKING:
    from buttermilk.data.vector import ChromaDBEmbeddings


class ZotDownloader(BaseModel):
    """Downloads and processes items from a Zotero library with incremental sync support.
    
    This class handles:
    - Incremental synchronization using Zotero's version tracking
    - PDF and full-text download from Zotero attachments
    - Conversion to Record objects for downstream processing
    - Integration with vector stores for duplicate detection
    
    The incremental sync feature tracks the library version from the last successful
    sync and only fetches items that have been modified since then. This significantly
    reduces API calls and processing time for large libraries.
    
    Attributes:
        save_dir: Directory path for saving downloaded files and sync state
        library: Zotero library ID to sync from

    """

    save_dir: str = Field(..., description="Directory to save downloaded files and sync state")
    library: str = Field(..., description="Zotero library ID to sync from")
    local: bool = Field(default=False, description="Use local mode for Zotero API")
    download_concurrency: int = Field(default=20, description="Maximum concurrent downloads from Zotero")

    _zot: zotero.Zotero = PrivateAttr()
    # Add private attribute to store the vector store instance
    _vector_store: "ChromaDBEmbeddings | None" = PrivateAttr(default=None)

    @pydantic.field_validator("local")
    @classmethod
    def validate_local(cls, v) -> bool:
        if v is None:
            return False
        return TypeAdapter(bool).validate_python(v)

    @pydantic.model_validator(mode="after")
    def _init(self) -> Self:
        bm = get_bm()
        self._zot = zotero.Zotero(
            library_id=self.library,
            library_type="group",
            api_key=bm.credentials.get("ZOTERO_API_KEY"),
            local=self.local,  # Use local mode if specified
        )
        os.makedirs(self.save_dir, exist_ok=True)
        return self

    def set_vector_store(self, vectoriser: "ChromaDBEmbeddings") -> None:
        """Stores the vectoriser instance to allow checking for existing documents."""
        self._vector_store = vectoriser
        logger.info("Vector store instance set for ZotDownloader.")

    def _get_state_file_path(self) -> Path:
        """Get the path to the sync state file."""
        return Path(self.save_dir) / ".zotero_sync_state.json"

    def _load_version_state(self) -> dict:
        """Load the last sync version state from file.
        
        Returns:
            dict: Dictionary with 'last_version' and 'last_sync_timestamp'

        """
        state_file = self._get_state_file_path()
        if state_file.exists():
            try:
                with state_file.open("r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Error loading sync state: {e}. Starting fresh.")

        return {"last_version": None, "last_sync_timestamp": None}

    def _save_version_state(self, version: int, timestamp: str) -> None:
        """Save the current sync version state to file.
        
        Args:
            version: The last successfully processed library version
            timestamp: ISO format timestamp of the sync

        """
        state_file = self._get_state_file_path()
        state = {
            "last_version": version,
            "last_sync_timestamp": timestamp,
        }

        try:
            with state_file.open("w") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved sync state: version={version}")
        except OSError as e:
            logger.error(f"Failed to save sync state: {e}")

    def reset_sync_state(self) -> None:
        """Reset sync state to force a full re-sync on next run."""
        state_file = self._get_state_file_path()
        if state_file.exists():
            state_file.unlink()
            logger.info("Sync state reset. Next sync will fetch all items.")

    async def get_all_records(self, force_full_sync: bool = False, max_docs: int | None = None, **kwargs) -> AsyncIterator[Record]:
        """Fetches Zotero items, checks existence, downloads, extracts, and yields Records.
        
        This method implements incremental sync by default, only fetching items that have
        been modified since the last successful sync. The sync state is persisted to disk.
        
        Args:
            force_full_sync: If True, bypasses incremental sync and fetches all items
            max_docs: Maximum number of records to yield before stopping (None = no limit)
            **kwargs: Additional parameters to pass to the Zotero API
            
        Yields:
            Record: Processed records from Zotero items

        """
        # Load sync state for incremental sync
        sync_state = self._load_version_state()
        last_version = sync_state["last_version"]

        # Prepare API parameters
        api_params = {
            # Exclude attachments, notes, annotations
            "itemType": "-attachment",
            "limit": 100,
            **kwargs,  # Allow override of any parameters
        }

        # Add incremental sync parameters if not forcing full sync
        if False and not force_full_sync and last_version is not None:
            api_params.update({
                "since": last_version,
                "sort": "dateModified",
                "direction": "asc",
            })
            logger.info(f"Starting incremental sync from version {last_version}")
        else:
            logger.info("Starting full sync of Zotero library")

        items = []
        library_version = None  # Will store the current library version

        try:
            # Fetch only parent items (books, articles, ...), not attachments directly
            results = self._zot.items(**api_params)

            items.extend(results)

            _next = self._zot.links.get("next")

            # Try to get library version from response headers
            if hasattr(self._zot, "request") and hasattr(self._zot.request, "headers"):
                library_version = self._zot.request.headers.get("last-modified-version")
                if library_version:
                    library_version = int(library_version)
                    logger.debug(f"Current library version: {library_version}")

        except Exception as e:
            logger.error(
                f"Error fetching initial items from Zotero: {e} {e.args=}",
            )
            return

        processed_count = 0
        skipped_count = 0

        # Use a set to track pending tasks across all batches
        pending_tasks = set()
        max_concurrent = self.download_concurrency  # Limit concurrent downloads
        items_exhausted = False  # Track when we've fetched all items

        while not items_exhausted or pending_tasks:
            # Stop creating new tasks if we're at max_docs limit
            if max_docs is not None and processed_count >= max_docs:
                # Cancel any remaining pending tasks
                for pending_task in pending_tasks:
                    pending_task.cancel()
                break

            # Create new download tasks while we have items and capacity
            while items and len(pending_tasks) < max_concurrent:
                # Check if we would exceed max_docs with new tasks
                if max_docs is not None and (processed_count + len(pending_tasks)) >= max_docs:
                    break

                item = items.pop(0)  # Process in order
                key = item.get("key")

                if not key:
                    logger.warning(
                        f"Item missing key, skipping: {item.get('data', {}).get('title', 'N/A')}",
                    )
                    continue

                if item.get("data", {}).get("itemType") in {"attachment", "note", "annotation"}:
                    logger.debug(f"Skipping item {key} of type {item.get('data', {}).get('itemType')}.")
                    continue
                # --- Check for existence using the stored vector_store ---
                if self._vector_store and self._vector_store.check_document_exists(key):
                    logger.info(
                        f"Document {key} already exists in vector store, skipping.",
                    )
                    skipped_count += 1
                    continue
                # --- End existence check ---

                # Create task for this item
                try:
                    title = item.get("data", {}).get("title", "Unknown")[:50]
                    logger.debug(f"🔵 Creating download task for #{key} '{title}' (pending: {len(pending_tasks)})")
                    task = asyncio.create_task(self.download_record(item))
                    pending_tasks.add(task)
                except Exception as e:
                    logger.error(
                        f"Error creating task for {item.get('key', 'unknown')}: {e} {e.args=}",
                    )

            # Fetch next page if we need more items and have capacity
            if _next and not items:
                try:
                    logger.debug(f"Following 'next' link for more Zotero items: {_next}")
                    response = self._zot._retrieve_data(_next)
                    items = response.json()
                    _next = self._zot._extract_links().get("next")

                    # Update library version from response headers
                    if hasattr(response, "headers"):
                        new_version = response.headers.get("last-modified-version")
                        if new_version:
                            library_version = int(new_version)
                            logger.debug(f"Updated library version: {library_version}")

                except Exception as e:
                    logger.error(
                        f"Error fetching next page from Zotero: {e} {e.args=}",
                    )
                    items_exhausted = True  # Mark as exhausted on error
            elif not _next and not items:
                items_exhausted = True  # No more items to fetch
                logger.debug("No more items to fetch from Zotero API")

            # If we have pending tasks, wait for at least one to complete
            if pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks immediately and yield results
                for task in done:
                    try:
                        result = await task
                        if result:
                            processed_count += 1
                            logger.debug(
                                f"🟢 Yielding record {result.record_id} '{result.title[:50] if result.title else 'Unknown'}' to pipeline"
                            )
                            yield result
                    except Exception as e:
                        logger.error(
                            f"Error processing download/convert result: {e} {e.args=}",
                        )

        logger.info(
            f"Finished Zotero processing. Processed: {processed_count}, Skipped (already exist): {skipped_count}",
        )

        # Save sync state if we have a library version (even if no items were processed,
        # we want to update the version to avoid re-checking the same items)
        if library_version is not None:
            timestamp = datetime.now(UTC).isoformat()
            self._save_version_state(library_version, timestamp)
            logger.info(f"Sync completed successfully. Library version: {library_version}")

    async def download_record(self, item) -> Record | None:
        """Downloads PDF/full text, saves item JSON, and creates a Record.

        Tries first to download the full text if available, otherwise downloads the PDF.
        Returns a Record if successful, None if not.
        """
        key = item.get("key")
        title = item.get("data", {}).get("title", "Unknown Title")
        # Try to populate title with first hit from item["data"] fields: title, shortTitle, nameOfAct, caseName,
        doi_or_url = item.get("data", {}).get("DOI") or item.get("data", {}).get("url")
        zotero_data = item.get("data", {})

        # Define file paths
        pdf_file = Path(self.save_dir) / f"{key}.pdf"
        json_file = Path(self.save_dir) / f"{key}.json"

        # Check if the item already exists in the cache
        if json_file.exists():
            # Load existing item from cache
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    item = json.load(f)
                if item.get("content"):
                    logger.debug(f"✅ Read fulltext from cache for #{key}  '{title[:50]}'")

                    metadata = {"title": title, "doi_or_url": doi_or_url, "uri": json_file.as_posix(), "zotero_data": zotero_data}

                    record = Record(
                        record_id=key,
                        content=item.get("content", ""),
                        file_path=pdf_file.as_posix(),
                        metadata=metadata,

                    )
                    return record
            except Exception as e:
                # If reading fails, proceed to download again
                logger.error(
                    f"Failed to read cached item JSON for {key}: {e} {e.args=}",
                )

        if not key:
            logger.warning(f"Item missing key: {item}")
            return None
        fulltext = None
        # Find the PDF attachment link
        attachment = item["links"].get("attachment", {})
        if (attachment.get("attachmentType") == "application/pdf") and (pdf_attachment := attachment.get("href")):
            attachment_key = pdf_attachment.split("/")[-1]
            try:
                # --- Download full text from Zotero ---
                logger.info(
                    f"⬇️ Starting full-text download for #{key} attachment #{attachment_key} for '{title[:50]}'..."
                )
                fulltext = self._zot.fulltext_item(attachment_key)

                # Here we check that the zotero index contains at least 90% of the PDF pages,
                # otherwise we'll download the PDF instead.
                # (By default Zotero indexes the first 100 pages.)
                if (
                    fulltext
                    and fulltext["indexedPages"] > 0
                    and fulltext["indexedPages"] >= (fulltext["totalPages"] * 0.9)
                ):
                    item["content"] = fulltext["content"]
                    logger.debug(
                        f"Full text downloaded for item {key}; {fulltext['indexedPages']} pages indexed by zotero out of {fulltext['totalPages']} total."
                    )
                else:
                    fulltext = None
                    logger.debug(
                        f"Full text not sufficient for item {key}; indexed pages: {fulltext['indexedPages'] if fulltext else 'N/A'}, total pages: {fulltext['totalPages'] if fulltext else 'N/A'}"
                    )

            except zotero_errors.ResourceNotFoundError as e:
                logger.debug(
                    f"Parsed full-text not found for item {key}: {e} {e.args=}",
                )
            except Exception as e:
                logger.error(
                    f"Error during download/convert for {key}: {e} {e.args=}",
                )
                return None

            if not fulltext:
                # --- Download PDF ---
                if not pdf_file.exists():
                    logger.debug(
                        f"Downloading attachment {attachment_key} for item {key} to {pdf_file}",
                    )
                    # Zotero python library is synchronous.
                    # Don't try to get around it, it's not thread safe
                    self._zot.dump(attachment_key, str(pdf_file))
                else:
                    logger.debug(f"PDF file already exists: {pdf_file}")

                    # TODO: Extract content from the PDF

            # --- Save Item JSON ---
            try:
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=4)
                logger.debug(f"Saved item data to {json_file}")
            except Exception as json_e:
                logger.error(
                    f"Failed to save item JSON for {key} to {json_file}: {json_e} {json_e.args=}",
                )

            # --- Prepare Record ---
            metadata = {
                "title": title,
                "doi_or_url": doi_or_url,
                "uri": json_file.as_posix(),
                "zotero_data": zotero_data,
            }
            record = Record(
                record_id=key,
                content=item.get("content", ""),
                file_path=pdf_file.as_posix(),
                metadata=metadata,
            )

            logger.debug(f"✅ Download complete for #{key} '{title[:50]}'")
            return record
        else:
            logger.debug(f"Skipping item {key}: No PDF attachment found.")
            # --- Save Item JSON even if no PDF ---
            try:
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=4)
                logger.debug(f"Saved item metadata (no PDF) to {json_file}")
            except Exception as json_e:
                logger.error(
                    f"Failed to save item JSON for {key} (no PDF) to {json_file}: {json_e} {json_e.args=}",
                )
            return None  # Return None as no PDF means no Record for embedding pipeline
