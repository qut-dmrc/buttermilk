import json
import re
from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk._core.types import Record


class PdfTextExtractor(BaseModel):
    """Extracts text content from PDF files."""

    save_dir: str
    _laparams: LAParams = PrivateAttr(
        default_factory=LAParams,
    )  # Allow customization if needed

    async def process(self, item: Record, **kwargs) -> Record | None:
        if isinstance(item.content, str) and item.content.strip():
            # Got it already; return
            return item

        item.content = self.extract(item.file_path) or ""
        metadata_file = Path(item.uri or "")

        # --- Save Item JSON with text ---
        try:
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(item.model_dump(), f, ensure_ascii=False, indent=4)
            logger.debug(f"Saved item record to {metadata_file}")
        except Exception as json_e:
            logger.error(
                f"Failed to save item JSON for {item.record_id} to {metadata_file}: {json_e} {json_e.args=}",
            )
        return item

    @staticmethod
    def _is_garbage_pdf_text(text: str) -> bool:
        # Check for high proportion of (cid:...) patterns
        cid_matches = re.findall(r"\(cid:\d+\)", text)
        cid_ratio = len(cid_matches) / max(len(text.split()), 1)

        # Check for low proportion of alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

        # Heuristic thresholds (tune as needed)
        return cid_ratio > 0.2 or alpha_ratio < 0.2

    def extract(self, file_path: str | Path) -> str | None:
        """Extracts text from the given PDF file path.

        Args:
            file_path: Path to the PDF file.

        Returns:
            The extracted text as a string, or None if extraction fails.

        """
        try:
            logger.debug(f"Extracting text from PDF: {file_path}")
            full_text = extract_text(file_path, laparams=self._laparams)
            if _is_garbage_pdf_text(full_text):
                logger.warning(
                    f"Extracted text from {file_path} appears to be garbage. "
                    "Consider using a different extraction method or preprocessing.",
                )
                return None
            logger.debug(f"Successfully extracted text from {file_path} (length: {len(full_text)}).")
            return full_text
        except Exception as e:
            logger.error(
                f"Error extracting text from PDF {file_path}: {e} {e.args=}",
            )
            return None
