import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Self

import pydantic
from pydantic import BaseModel, PrivateAttr
from pyzotero import zotero

from buttermilk.bm import bm, logger
from buttermilk.data.vector import InputDocument
from buttermilk.utils import download_limited_async, get_pdf_text


# Placeholder implementation (replace with your actual function)
async def default_generate_citation(text: str) -> str:
    logger.warning("Using placeholder citation generator.")
    # Example: return first 100 chars as placeholder citation
    return f"Placeholder Citation: {text[:100]}..."


class ZotDownloader(BaseModel):
    save_dir: str
    library: str

    _zot: zotero.Zotero = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def _init(self) -> Self:
        self._zot = zotero.Zotero(
            library_id=self.library,
            library_type="group",
            api_key=bm.credentials.get("ZOTERO_API_KEY"),
        )
        os.makedirs(self.save_dir, exist_ok=True)
        return self

    async def get_all_records(self, **kwargs) -> AsyncIterator[InputDocument]:
        items = []
        items.extend(self._zot.items(itemType="book || journalArticle", **kwargs))
        while items or self._zot.links.get("next"):
            tasks = []
            while items:
                item = items.pop()
                try:
                    # Ensure download returns InputDocument | None
                    tasks.append(asyncio.create_task(self.download_and_convert(item)))

                except Exception as e:
                    logger.error(
                        f"Error creating task for {item.get('key', 'unknown')}: {e} {e.args=}",
                    )

            # Process completed tasks as they finish
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    if result:
                        yield result
                except Exception as e:
                    # Log errors from awaited tasks if necessary, though download handles its own
                    logger.error(f"Error processing download result: {e}")

            break  # for debugging only

            # Fetch next page if available
            if self._zot.links.get("next"):
                try:
                    items.extend(self._zot.follow())
                except Exception as e:
                    logger.error(f"Error fetching next page from Zotero: {e}")
                    break  # Stop if pagination fails
            # Explicitly break if there are no more items and no next link
            # This prevents an infinite loop if the initial items list was empty
            elif not items:
                break

    async def download_and_convert(self, item) -> InputDocument | None:
        uri = item.get("links", {}).get("attachment", {}).get("href")
        mime = item.get("links", {}).get("attachment", {}).get("attachmentType")
        key = item.get("key")
        title = item.get("data", {}).get("title", "Unknown Title")
        doi = item.get("data", {}).get("DOI", item.get("data", {}).get("url"))

        if not key:
            logger.warning(f"Item missing key: {item}")
            return None

        if uri and mime == "application/pdf":
            file = Path(self.save_dir) / f"{key}.pdf"
            try:
                if not file.exists():
                    logger.debug(f"Downloading {key} to {file}")
                    pdf_content, _mimetype = await download_limited_async(uri)
                    with file.open("wb") as f:
                        f.write(pdf_content)
                else:
                    logger.debug(f"File already exists: {file}")

                full_text = get_pdf_text(file.as_posix())

                metadata = dict(doi=doi, zotero_data=item.get("data"))
                record = InputDocument(
                    file_path=file.as_posix(),
                    full_text=full_text,
                    record_id=key,
                    title=title,
                    metadata=metadata,
                )

                return record

            except Exception as e:
                logger.error(
                    f"Error downloading or processing {key}: {e}",
                    exc_info=True,
                )
                return None
        else:
            logger.debug(f"Skipping item {key}: No PDF attachment found.")
            return None
