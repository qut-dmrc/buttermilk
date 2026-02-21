"""GCS image storage processor for pipeline integration.

This module provides GCSImageStorageProcessor which handles uploading
generated images to Google Cloud Storage with structured paths based
on scenario and session metadata.
"""

import re
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cloudpathlib import GSPath
from pydantic import BaseModel

from buttermilk import logger
from buttermilk._core.types import BaseRecord


class GCSImageStorageProcessor(BaseModel):
    """Processor that uploads images to GCS with structured paths.

    Constructs paths following the pattern:
    {base_path}/{sanitized_scenario}/{session_id}/{model_prefix}{record_id}.png

    Attributes:
        bucket: GCS bucket name (without gs:// prefix).
        base_path: Base path within the bucket.
        local_only: If True, save to local directory instead of GCS (for testing).
        local_dir: Local directory to use when local_only=True.
    """

    bucket: str
    base_path: str = ""
    local_only: bool = False
    local_dir: str | None = None

    def _sanitize_scenario(self, scenario: str) -> str:
        """Sanitize scenario string for use in file paths.

        Replaces spaces with underscores and removes special characters.

        Args:
            scenario: Raw scenario string (e.g., "working in an office!")

        Returns:
            Sanitized string (e.g., "working_in_an_office")
        """
        # Replace spaces with underscores
        sanitized = scenario.replace(" ", "_")
        # Remove special characters, keep only alphanumeric and underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", sanitized)
        # Convert to lowercase for consistency
        return sanitized.lower()

    def _construct_storage_path(self, record: BaseRecord) -> str:
        """Construct the full storage path for an image.

        Path structure: {base_path}/{scenario}/{session_id}/{model_prefix}{record_id}.png

        Args:
            record: BaseRecord containing image metadata

        Returns:
            Full storage path (GCS URI or local path)

        Raises:
            KeyError: If required metadata fields are missing
        """
        # Extract required metadata with fail-fast
        session_id = record.metadata["session_id"]
        scenario = record.metadata["scenario"]
        model_prefix = record.metadata.get("model_prefix", "")

        # Sanitize scenario for file path
        sanitized_scenario = self._sanitize_scenario(scenario)

        # Construct path components
        path_parts = []
        if self.base_path:
            path_parts.append(self.base_path)
        path_parts.extend([sanitized_scenario, session_id])

        # Filename: model_prefix + record_id + extension
        filename = f"{model_prefix}{record.record_id}.png"

        if self.local_only:
            # Local path
            if not self.local_dir:
                raise ValueError("local_dir must be set when local_only=True")
            base = Path(self.local_dir)
            for part in path_parts:
                base = base / part
            return str(base / filename)
        else:
            # GCS path
            path_str = "/".join(path_parts + [filename])
            return f"gs://{self.bucket}/{path_str}"

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by uploading its image to GCS.

        Args:
            record: BaseRecord with image_uri in metadata
            processor_stage: Pipeline stage identifier (e.g., "store")
            **kwargs: Additional arguments (unused)

        Yields:
            BaseRecord with original data plus storage_uri in metadata

        Raises:
            KeyError: If required metadata fields are missing
            ValueError: If image_uri is not set or file doesn't exist
        """
        # Fail-fast: Check for required metadata
        if "image_uri" not in record.metadata:
<<<<<<< HEAD
            raise ValueError(f"Record {record.record_id} missing 'image_uri' in metadata")
=======
            raise ValueError(
                f"Record {record.record_id} missing 'image_uri' in metadata"
            )
>>>>>>> origin/stable

        image_uri = record.metadata["image_uri"]

        # Parse source image URI
        parsed = urlparse(image_uri)
        if parsed.scheme == "file":
            source_path = Path(parsed.path)
        else:
            source_path = Path(image_uri)

        if not source_path.exists():
            raise ValueError(f"Source image does not exist: {source_path}")

        # Construct destination path
        dest_path_str = self._construct_storage_path(record)

        logger.info(
            f"Storing image for record {record.record_id}",
            extra={
                "record_id": record.record_id,
                "source": str(source_path),
                "destination": dest_path_str,
                "stage": processor_stage,
            },
        )

        # Perform upload/copy
        if self.local_only:
            # Copy to local destination
            dest_path = Path(dest_path_str)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            storage_uri = dest_path.as_uri()
        else:
            # Upload to GCS
            gs_path = GSPath(dest_path_str)
            gs_path.parent.mkdir(parents=True, exist_ok=True)
            gs_path.write_bytes(source_path.read_bytes())
            storage_uri = str(gs_path)

        # Return record with storage_uri added to metadata
        result = record.model_copy(
            update={
                "metadata": {
                    **record.metadata,
                    "storage_uri": storage_uri,
                }
            }
        )

        yield result
