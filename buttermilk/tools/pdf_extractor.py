from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pydantic import BaseModel

from buttermilk import logger


class PdfTextExtractor(BaseModel):
    """Extracts text content from PDF files."""

    laparams: LAParams = LAParams()  # Allow customization if needed

    def extract(self, file_path: str | Path) -> str | None:
        """Extracts text from the given PDF file path.

        Args:
            file_path: Path to the PDF file.

        Returns:
            The extracted text as a string, or None if extraction fails.

        """
        try:
            logger.debug(f"Extracting text from PDF: {file_path}")
            full_text = extract_text(file_path, laparams=self.laparams)
            logger.debug(f"Successfully extracted text from {file_path} (length: {len(full_text)}).")
            return full_text
        except Exception as e:
            logger.error(
                f"Error extracting text from PDF {file_path}: {e}",
                exc_info=True,
            )
            return None
