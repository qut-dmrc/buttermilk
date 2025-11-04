"""CSV Metadata Logger for tracking image generation metadata."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk._core.types import BaseRecord


class CSVMetadataLogger(BaseModel):
    """Accumulates generation metadata and writes CSV log on finalize."""

    output_path: str
    _accumulated_records: list[dict] = PrivateAttr(default_factory=list)

    async def process(self, record: BaseRecord, *, processor_stage: str, **kwargs: Any) -> AsyncGenerator[BaseRecord, None]:
        """Accumulate metadata and pass record through unchanged."""
        # Extract metadata into dict for CSV row
        storage_uri = record.metadata.get("storage_uri", "")
        filename = Path(storage_uri).name if storage_uri else ""

        row = {
            "prompt": record.content,
            "model": record.metadata.get("model_class", "unknown"),
            "timestamp": record.metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "filename": filename,
            "scenario": record.metadata.get("scenario", ""),
            "session_id": record.metadata.get("session_id", ""),
            "repetition": record.metadata.get("repetition", 0),
        }

        self._accumulated_records.append(row)

        yield record  # Pass through unchanged

    async def finalize_processing(self) -> None:
        """Write accumulated records to CSV."""
        logger.info("csv_metadata_logger_finalizing", output_path=self.output_path, record_count=len(self._accumulated_records))

        # Define column order
        columns = ["prompt", "model", "timestamp", "filename", "scenario", "session_id", "repetition"]

        # Create DataFrame with specified column order
        if self._accumulated_records:
            df = pd.DataFrame(self._accumulated_records)
            # Ensure columns are in the right order
            df = df[columns]
        else:
            # Empty DataFrame with correct columns
            df = pd.DataFrame(columns=columns)

        # Ensure output directory exists
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        df.to_csv(self.output_path, index=False)

        logger.info("csv_metadata_logger_complete", output_path=self.output_path, rows_written=len(df))
