import asyncio
import json  # Import json module
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Self  # Import TYPE_CHECKING

import pydantic
from pydantic import BaseModel, PrivateAttr  # Import Field
from pyzotero import zotero

from buttermilk.bm import bm, logger

# Import ChromaDBEmbeddings for type hinting
from buttermilk.data.vector import ChromaDBEmbeddings, InputDocument

# Add TYPE_CHECKING block for forward reference if ChromaDBEmbeddings is in a different module
# and causes circular import issues. If they are in the same module or structure prevents
# circular imports, this might not be strictly necessary but is good practice.
if TYPE_CHECKING:
    from buttermilk.data.vector import ChromaDBEmbeddings


class ZotDownloader(BaseModel):
    save_dir: str
    library: str

    _zot: zotero.Zotero = PrivateAttr()
    # Add private attribute to store the vector store instance
    _vector_store: "ChromaDBEmbeddings | None" = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _init(self) -> Self:
        self._zot = zotero.Zotero(
            library_id=self.library,
            library_type="group",
            api_key=bm.credentials.get("ZOTERO_API_KEY"),
        )
        os.makedirs(self.save_dir, exist_ok=True)
        return self

    def set_vector_store(self, vectoriser: "ChromaDBEmbeddings") -> None:
        """Stores the vectoriser instance to allow checking for existing documents."""
        self._vector_store = vectoriser
        logger.info("Vector store instance set for ZotDownloader.")

    async def get_all_records(self, **kwargs) -> AsyncIterator[InputDocument]:
        """Fetches Zotero items, checks existence, downloads, extracts, and yields InputDocuments."""
        items = []
        try:
            # Fetch only parent items (books, articles), not attachments directly
            items.extend(
                self._zot.items(itemType="book || journalArticle", limit=100, **kwargs),
            )
            _next = self._zot.links.get("next")
        except Exception as e:
            logger.error(
                f"Error fetching initial items from Zotero: {e} {e.args=}",
            )
            return

        processed_count = 0
        skipped_count = 0

        while items or _next:
            tasks = []
            items_to_process_this_batch = []

            while items:
                item = items.pop(0)  # Process in order
                key = item.get("key")

                if not key:
                    logger.warning(
                        f"Item missing key, skipping: {item.get('data', {}).get('title', 'N/A')}",
                    )
                    continue

                # --- Check for existence using the stored vector_store ---
                if self._vector_store and self._vector_store.check_document_exists(key):
                    logger.info(
                        f"Document {key} already exists in vector store, skipping.",
                    )
                    skipped_count += 1
                    continue
                # --- End existence check ---

                # If not skipped, add to list for task creation
                items_to_process_this_batch.append(item)

            # Create tasks only for items not skipped
            for item_to_process in items_to_process_this_batch:
                try:
                    # Run download_and_convert in a separate thread to avoid blocking
                    # the event loop with synchronous file I/O and API calls within the loop.
                    # Note: self._zot calls might still be synchronous internally.
                    # Consider a thread pool executor for true non-blocking I/O if needed.
                    tasks.append(
                        asyncio.create_task(self.download_record(item_to_process)),
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating task for {item_to_process.get('key', 'unknown')}: {e} {e.args=}",
                    )

            # Process completed tasks for this batch
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    if result:
                        processed_count += 1
                        yield result
                except Exception as e:
                    logger.error(
                        f"Error processing download/convert result: {e} {e.args=}",
                    )
            # Fetch next page if available
            if _next:
                try:
                    logger.debug("Following 'next' link for more Zotero items...")
                    items = self._zot._retrieve_data(_next).json()
                    _next = self._zot.links.get("next")
                except Exception as e:
                    logger.error(
                        f"Error fetching next page from Zotero: {e} {e.args=}",
                    )
                    break  # Stop if pagination fails
            elif (
                not items
            ):  # Break if no next link AND no items left from previous fetch
                logger.debug(
                    "No 'next' link and no more items, finishing Zotero item retrieval.",
                )
                break

        logger.info(
            f"Finished Zotero processing. Processed: {processed_count}, Skipped (already exist): {skipped_count}",
        )

    async def download_record(self, item) -> InputDocument | None:
        """Downloads PDF, saves item JSON, and creates an InputDocument."""
        key = item.get("key")
        title = item.get("data", {}).get("title", "Unknown Title")
        doi_or_url = item.get("data", {}).get("DOI") or item.get("data", {}).get("url")
        zotero_data = item.get("data", {})

        if not key:
            logger.warning(f"Item missing key: {item}")
            return None

        # Define file paths
        pdf_file = Path(self.save_dir) / f"{key}.pdf"
        json_file = Path(self.save_dir) / f"{key}.json"

        # Find the PDF attachment link
        attachment = item["links"].get("attachment", {})
        if (attachment.get("attachmentType") == "application/pdf") and (
            pdf_attachment := attachment.get("href")
        ):
            attachment_key = pdf_attachment.split("/")[-1]
            try:
                # --- Download PDF ---
                if not pdf_file.exists():
                    logger.debug(
                        f"Downloading attachment {attachment_key} for item {key} to {pdf_file}",
                    )
                    # Wrap synchronous Zotero call
                    success = await asyncio.to_thread(
                        self._zot.dump,
                        attachment_key,
                        str(pdf_file),
                    )
                    if not success:
                        logger.warning(
                            f"Download failed for attachment {attachment_key} (item {key}).",
                        )
                        return None
                else:
                    logger.debug(f"PDF file already exists: {pdf_file}")

                # --- Save Item JSON ---
                try:
                    with json_file.open("w", encoding="utf-8") as f:
                        json.dump(item, f, ensure_ascii=False, indent=4)
                    logger.debug(f"Saved item metadata to {json_file}")
                except Exception as json_e:
                    logger.error(
                        f"Failed to save item JSON for {key} to {json_file}: {json_e} {json_e.args=}",
                    )

                # --- Prepare InputDocument ---
                metadata = {"doi_or_url": doi_or_url, "zotero_data": zotero_data}
                record = InputDocument(
                    file_path=pdf_file.as_posix(),
                    record_id=key,
                    title=title,
                    metadata=metadata,
                )
                return record

            except Exception as e:
                logger.error(
                    f"Error during download/convert for {key}: {e} {e.args=}",
                )
                return None
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
            return None  # Return None as no PDF means no InputDocument for embedding pipeline
