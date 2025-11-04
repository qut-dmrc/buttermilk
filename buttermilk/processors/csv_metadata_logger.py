"""CSV Metadata Logger for tracking image generation metadata."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from cloudpathlib import GSPath
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk._core.types import BaseRecord


class CSVMetadataLogger(BaseModel):
    """Accumulates generation metadata and writes CSV log on finalize.

    Attributes:
        bucket: GCS bucket name (without gs:// prefix).
        base_path: Base path within the bucket (session_id will be appended).
    """

    bucket: str
    base_path: str = ""
    _accumulated_records: list[dict] = PrivateAttr(default_factory=list)
    _session_id: str | None = PrivateAttr(default=None)

    async def process(self, record: BaseRecord, *, processor_stage: str, **kwargs: Any) -> AsyncGenerator[BaseRecord, None]:
        """Accumulate metadata and pass record through unchanged."""
        # Extract metadata into dict for CSV row
        storage_uri = record.metadata.get("storage_uri", "")
        filename = Path(storage_uri).name if storage_uri else ""

        # Capture session_id from first record
        if self._session_id is None:
            self._session_id = record.metadata.get("session_id", "")

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
        """Write accumulated records to CSV at GCS path constructed from session_id."""
        # Construct output path: gs://{bucket}/{base_path}/{session_id}/generation_log.csv
        if not self._session_id:
            logger.warning("csv_metadata_logger_no_session_id", message="No records processed, cannot determine session_id")
            return

        path_parts = [part for part in [self.base_path, self._session_id] if part]
        gcs_path = GSPath(f"gs://{self.bucket}") / "/".join(path_parts) / "generation_log.csv"

        logger.info("csv_metadata_logger_finalizing", output_path=str(gcs_path), record_count=len(self._accumulated_records))

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

        # Write to GCS
        with gcs_path.open("w") as f:
            df.to_csv(f, index=False)

        logger.info("csv_metadata_logger_complete", output_path=str(gcs_path), rows_written=len(df))
