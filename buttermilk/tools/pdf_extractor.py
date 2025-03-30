import json
from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk.data.vector import InputDocument


class PdfTextExtractor(BaseModel):
    """Extracts text content from PDF files."""

    save_dir: str
    _laparams: LAParams = PrivateAttr(
        default_factory=LAParams
    )  # Allow customization if needed

    async def process(self, item: InputDocument, **kwargs) -> InputDocument | None:
        if item.full_text:
            # Got it already; return
            return item

        item.full_text = self.extract(item.file_path)
        metadata_file = Path(item.record_path)

        # --- Save Item JSON with text ---
        try:
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=4)
            logger.debug(f"Saved item record to {metadata_file}")
        except Exception as json_e:
            logger.error(
                f"Failed to save item JSON for {item.record_id} to {metadata_file}: {json_e} {e.args=}",
            )
        return item

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
            logger.debug(f"Successfully extracted text from {file_path} (length: {len(full_text)}).")
            return full_text
        except Exception as e:
            logger.error(
                f"Error extracting text from PDF {file_path}: {e}",
                exc_info=True,
            )
            return None
