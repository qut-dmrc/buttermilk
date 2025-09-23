"""File storage implementation for unified storage operations."""

import json
from typing import TYPE_CHECKING, Iterator

from cloudpathlib import AnyPath  # For handling local and cloud paths

from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord, Record

from .base import Storage, StorageError

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM

    from .._core.storage_config import StorageConfig


class FileStorage(Storage):
    """Unified file storage supporting both read and write operations.
    
    Supports local files and cloud storage paths (GCS, S3) for JSON/JSONL formats.
    """

    def __init__(self, config: "StorageConfig", bm: "BM | None" = None):
        """Initialize file storage.
        
        Args:
            config: Storage configuration with file path
            bm: Buttermilk instance (optional for file operations)
        """
        super().__init__(config, bm)

        if not config.path:
            raise ValueError("File storage requires a path")

        self.path = AnyPath(config.path)

    def __iter__(self) -> Iterator[BaseRecord]:
        """Iterate over records from file.

        Yields:
            BaseRecord objects from the file (Record, Title, or other subclasses)
        """
        try:
            if not self.exists():
                logger.warning(f"File does not exist: {self.path}")
                return

            # Handle both local and cloud paths (GCS, S3, etc.)
            if str(self.path).startswith(("gs://", "s3://", "azure://")):
                # Use cloudpathlib for cloud storage paths
                file_obj = self.path.open("r", encoding="utf-8")
            else:
                # Use regular open for local files
                file_obj = open(self.path, "r")

            try:
                # Check if the file is a JSON array or JSONL
                first_char = file_obj.read(1)
                file_obj.seek(0)  # Reset to beginning

                if first_char == "[":
                    # Handle JSON array format
                    data_array = json.load(file_obj)
                    for line_num, data in enumerate(data_array, 1):
                        try:
                            record = self._dict_to_record(data, line_num)
                            yield record
                        except Exception as e:
                            logger.warning(f"Error processing JSON array item {line_num}: {e}")
                else:
                    # Handle JSONL format (one JSON object per line)
                    for line_num, line in enumerate(file_obj, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            record = self._dict_to_record(data, line_num)
                            yield record
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")
                            continue
            finally:
                file_obj.close()

        except Exception as e:
            logger.error(f"Error reading from file {self.path}: {e}")
            raise StorageError(f"Failed to read file: {e}") from e

    def save(self, records: list[BaseRecord] | BaseRecord | list | dict) -> None:
        """Save records to file.

        Args:
            records: Single record/dict or list of records/dicts to save
        """
        if not isinstance(records, list):
            records = [records]

        if not records:
            logger.warning("No records to save")
            return

        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Convert records to dictionaries
            data = [self._record_to_dict(record) for record in records]

            with open(self.path, "w") as f:
                if self.path.suffix == ".jsonl":
                    # JSONL format - one JSON object per line
                    for record_dict in data:
                        json.dump(record_dict, f, ensure_ascii=False)
                        f.write("\n")
                else:
                    # JSON format - single JSON array
                    json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved {len(records)} records to {self.path}")

        except Exception as e:
            logger.error(f"Error saving records to file {self.path}: {e}")
            raise StorageError(f"Failed to save file: {e}") from e

    def count(self) -> int:
        """Count total records in file.
        
        Returns:
            Number of records in the file
        """
        try:
            count = 0
            for _ in self:
                count += 1
            return count
        except Exception as e:
            logger.warning(f"Error counting records in file {self.path}: {e}")
            return -1

    def exists(self) -> bool:
        """Check if the file exists.
        
        Returns:
            True if file exists, False otherwise
        """
        return self.path.exists() and self.path.is_file()

    def create(self) -> None:
        """Create an empty file if it doesn't exist."""
        if self.exists():
            return

        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty file with appropriate format
            with open(self.path, "w") as f:
                if self.path.suffix == ".jsonl":
                    # Empty JSONL file
                    pass
                else:
                    # Empty JSON array
                    json.dump([], f)

            logger.info(f"Created empty file: {self.path}")

        except Exception as e:
            logger.error(f"Error creating file {self.path}: {e}")
            raise StorageError(f"Failed to create file: {e}") from e

    def _dict_to_record(self, data: dict, index: int) -> BaseRecord:
        """Convert dictionary to BaseRecord object.

        Simple conversion that lets the consuming code handle type-specific logic.

        Args:
            data: Dictionary data from file
            index: Record index for error reporting

        Returns:
            BaseRecord object (typically a Record)
        """
        try:
            # Apply column mapping if configured
            if self.config.columns:
                mapped_data = {}
                consumed_source_fields = set()  # Track which source fields should be removed

                # Collect all source fields that will be mapped
                all_source_fields = set()
                for new_key, old_key in self.config.columns.items():
                    if new_key == "metadata" and isinstance(old_key, dict):
                        for meta_key, meta_source in old_key.items():
                            if meta_source in data:
                                all_source_fields.add(meta_source)
                    elif old_key in data:
                        all_source_fields.add(old_key)

                # Perform the actual mapping
                for new_key, old_key in self.config.columns.items():
                    # Handle nested metadata mapping
                    if new_key == "metadata" and isinstance(old_key, dict):
                        metadata = {}
                        for meta_key, meta_source in old_key.items():
                            if meta_source in data:
                                metadata[meta_key] = data[meta_source]
                        mapped_data["metadata"] = metadata
                    elif old_key in data:
                        mapped_data[new_key] = data[old_key]

                # Mark source fields for removal only if they were actually consumed
                for new_key, old_key in self.config.columns.items():
                    if new_key == "metadata" and isinstance(old_key, dict):
                        for meta_key, meta_source in old_key.items():
                            if meta_source in data:
                                consumed_source_fields.add(meta_source)
                    elif old_key in data:
                        consumed_source_fields.add(old_key)

                # Merge mapped data with original, but handle metadata specially
                if mapped_data:
                    # Start with original data
                    data = {**data}

                    # Apply non-metadata mappings
                    for key, value in mapped_data.items():
                        if key != "metadata":
                            data[key] = value

                    # Merge metadata: original metadata + mapped metadata
                    if "metadata" in mapped_data:
                        original_metadata = data.get("metadata", {})
                        if isinstance(original_metadata, str):
                            try:
                                original_metadata = json.loads(original_metadata)
                            except json.JSONDecodeError:
                                original_metadata = {"raw_metadata": original_metadata}

                        mapped_metadata = mapped_data["metadata"]
                        data["metadata"] = {**original_metadata, **mapped_metadata}

                # Remove only the original source fields to avoid duplication
                # But preserve unmapped fields and target fields

                # Get all direct mapping source fields (not nested metadata)
                direct_source_fields = [old_key for new_key, old_key in self.config.columns.items()
                                      if new_key != "metadata" and isinstance(old_key, str)]

                # Get all metadata source fields
                metadata_source_fields = []
                for new_key, old_key in self.config.columns.items():
                    if new_key == "metadata" and isinstance(old_key, dict):
                        metadata_source_fields.extend(old_key.values())

                for field in consumed_source_fields:
                    should_remove = False

                    # Remove if it's a direct mapping source field that's being renamed
                    if field in direct_source_fields and field not in self.config.columns.keys():
                        should_remove = True

                    # Remove if it's only used for metadata mapping and not a target field
                    if field in metadata_source_fields and field not in self.config.columns.keys():
                        should_remove = True

                    if should_remove:
                        data.pop(field, None)

            # Parse metadata field if it's a string
            if "metadata" in data and isinstance(data["metadata"], str):
                try:
                    data["metadata"] = json.loads(data["metadata"])
                except json.JSONDecodeError:
                    data["metadata"] = {"raw_metadata": data["metadata"]}
            elif "metadata" not in data:
                data["metadata"] = {}

            # Parse error field if it exists and is a string
            if "error" in data and isinstance(data["error"], str):
                try:
                    data["error"] = json.loads(data["error"])
                except json.JSONDecodeError:
                    data["error"] = []
            elif "error" not in data:
                data["error"] = []

            # Ensure required BaseRecord fields with sensible defaults
            if "record_id" not in data:
                data["record_id"] = data.get("id", f"record_{index}")
            if "dataset_name" not in data:
                data["dataset_name"] = self.config.dataset_name or "default"
            if "split_type" not in data:
                data["split_type"] = self.config.split_type or "default"

            # Map common alternative field names
            if "content" not in data and "text" in data:
                data["content"] = data["text"]

            # Create a basic Record - let consuming code handle type conversions
            try:
                return Record(**data)
            except Exception as e:
                # If Record creation fails, create minimal valid record
                logger.warning(f"Failed to create Record from data at index {index}: {e}")
                return Record(
                    record_id=str(data.get("record_id", f"error_{index}")),
                    dataset_name=str(data.get("dataset_name", "default")),
                    split_type=str(data.get("split_type", "default")),
                    metadata=data.get("metadata", {}),
                    content=str(data),  # Store full data as content for debugging
                    error=data.get("error", [])
                )

        except Exception as e:
            logger.warning(f"Error converting data to Record at index {index}: {e}")
            # Create a safer error record with string representation of data
            try:
                safe_data = str(data)[:1000]  # Limit length to avoid huge error messages
                safe_metadata = {"parse_error": str(e)}
                # Don't include original_data as it might not be serializable
                return Record(
                    record_id=f"error_{index}",
                    content=safe_data,
                    metadata=safe_metadata
                )
            except Exception as e2:
                # Ultimate fallback
                return Record(
                    record_id=f"error_{index}",
                    content=f"Failed to parse record: {str(e2)}",
                    metadata={"critical_error": True}
                )

    def _record_to_dict(self, record: BaseRecord) -> dict:
        """Convert Record object to dictionary for file storage.
        
        Args:
            record: Record object to convert
            
        Returns:
            Dictionary representation
        """
        result = {
            "record_id": record.record_id,
            "content": record.content,
            "metadata": record.metadata,
        }

        # Add optional fields if present
        if record.alt_text:
            result["alt_text"] = record.alt_text
        if record.ground_truth:
            result["ground_truth"] = record.ground_truth
        if record.metadata.get("uri"):
            result["uri"] = record.metadata.get("uri")
        if record.mime and record.mime != "text/plain":
            result["mime"] = record.mime

        return result
