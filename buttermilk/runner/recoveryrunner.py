"""Recovery runner for failed BigQuery uploads.

This module provides functionality to retry failed BigQuery uploads that were
saved to fallback storage (GCS or local disk) when the original upload failed.
"""

import json

from cloudpathlib import AnyPath
from pydantic import BaseModel, Field

from buttermilk._core.bm_init import get_bm
from buttermilk._core.config import SaveInfo
from buttermilk._core.log import logger
from buttermilk.utils.save import upload_rows


class RecoveryRunner(BaseModel):
    """Runner for recovering failed BigQuery uploads."""

    mode: str = Field(default="recovery", description="Runner mode")
    ui: str = Field(default="console", description="UI type")
    backup_dir: str | None = Field(default=None, description="Directory to scan for failed uploads")
    save: SaveInfo | None = Field(default=None, description="Save destination for retry uploads")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.backup_dir:
            # Use BM save_dir if no backup_dir specified
            bm = get_bm()
            self.backup_dir = bm.save_dir

    async def run(self) -> None:
        """Main recovery process."""
        logger.info(f"Starting recovery process, scanning: {self.backup_dir}")

        if not self.save:
            logger.error("No save destination configured for recovery")
            return

        backup_path = AnyPath(self.backup_dir)

        # Find potential failed upload files
        failed_files = []

        if backup_path.is_cloud:
            # For cloud storage, we'd need to list files differently
            # This is a simplified version - in practice you'd use the cloud client
            logger.warning("Cloud storage scanning not fully implemented")
            return
        else:
            # Local directory scanning
            for pattern in ["data_*.json", "data_*.pkl", "backup_*.json"]:
                failed_files.extend(backup_path.glob(pattern))

        if not failed_files:
            logger.info("No failed upload files found")
            return

        logger.info(f"Found {len(failed_files)} potential failed upload files")

        successful_recoveries = 0
        failed_recoveries = 0

        for file_path in failed_files:
            try:
                logger.info(f"Attempting to recover: {file_path}")

                # Load the failed data
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Attempt BigQuery upload
                result = upload_rows(data, save_dest=self.save)

                if result:
                    logger.info(f"Successfully recovered {file_path} to {result}")
                    successful_recoveries += 1

                    # Move or delete the recovered file
                    recovered_path = file_path.with_suffix(file_path.suffix + ".recovered")
                    file_path.rename(recovered_path)
                    logger.info(f"Marked as recovered: {recovered_path}")
                else:
                    logger.warning(f"Recovery upload returned no result for {file_path}")
                    failed_recoveries += 1

            except Exception as e:
                logger.error(f"Failed to recover {file_path}: {e}")
                failed_recoveries += 1

        logger.info(f"Recovery complete: {successful_recoveries} successful, {failed_recoveries} failed")


def create_recovery_runner(**kwargs) -> RecoveryRunner:
    """Factory function to create a RecoveryRunner instance."""
    return RecoveryRunner(**kwargs)
