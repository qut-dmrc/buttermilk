import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Self

import pydantic
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pydantic import BaseModel, Field, PrivateAttr
from pyzotero import zotero

from buttermilk.bm import bm, logger
from buttermilk.data.vector import InputDocument
from buttermilk.utils import download_limited_async

CITATION_TEXT_CHAR_LIMIT = 4000  # characters


# Placeholder implementation (replace with your actual function)
async def default_generate_citation(text: str) -> str:
    logger.warning("Using placeholder citation generator.")
    # Example: return first 100 chars as placeholder citation
    return f"Placeholder Citation: {text[:100]}..."


class ZotDownloader(BaseModel):
    save_dir: str

    citation_generator: Callable[[str], Awaitable[str]] = Field(
        default=default_generate_citation,
        exclude=True,
    )  # Exclude from model serialization
    _zot: zotero.Zotero = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def _init(self) -> Self:
        self._zot = zotero.Zotero(
            library_id=bm.credentials.get("ZOTERO_LIBRARY_ID"),
            library_type="group",
            api_key=bm.credentials.get("ZOTERO_API_KEY"),
        )
        return self

    async def generate_citation(self, item: InputDocument) -> InputDocument:
        try:
            full_text = extract_text(item.file_path, laparams=LAParams())
        except Exception as e:
            logger.error(
                f"Error extracting text from PDF {item.file_path}: {e} {e.args=}",
            )
            return item

        try:
            # Take the first N characters for citation generation
            citation_text = full_text[:CITATION_TEXT_CHAR_LIMIT]
            logger.debug(
                f"Generating citation for doc {item.record_id} using first {len(citation_text)} chars.",
            )

            generated_citation = await self.citation_generator(citation_text)

            # Store it in the metadata (overwrites if 'citation' key already exists)
            item.metadata["citation"] = generated_citation
            logger.debug(
                f"Generated citation for doc {item.record_id}: '{generated_citation[:100]}...'",
            )
            return item
        except Exception as e:
            logger.error(
                f"Error generating citation for doc {item.record_id}: {e}",
                exc_info=True,
            )
            return item

    async def get_all_records(self, **kwargs):
        items = []
        items.extend(self._zot.items(itemType="book || journalArticle"))
        outputs = []
        while items or self._zot.links.get("next"):
            tasks = []
            while items:
                item = items.pop()
                try:
                    tasks.append(asyncio.create_task(self.download(item)))
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error with {item['key']}: {e} {e.args=}")

            results = await asyncio.gather(*tasks)
            for r in results:
                if r:
                    outputs.append(r.payload)
            break

            items.extend(self._zot.follow())
        return outputs

    async def download(self, item) -> InputDocument | None:
        uri = item.get("links", {}).get("attachment", {}).get("href")
        mime = item.get("links", {}).get("attachment", {}).get("attachmentType")
        if uri and mime == "application/pdf":
            file = Path(self.save_dir) / f"{item['key']}.pdf"

            if not file.exists():
                with file.open("wb") as f:
                    f.write(
                        await download_limited_async(
                            item["links"]["attachment"]["href"],
                        ),
                    )
            record = InputDocument(
                file_path=file.as_posix(),
                record_id=item["key"],
                title=item["data"]["title"],
                metadata=dict(doi=item["data"].get("DOI", item["data"].get("url"))),
            )

            record = await self.generate_citation(record)

            return record
        return None
