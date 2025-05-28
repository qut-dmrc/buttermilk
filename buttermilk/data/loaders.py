"""
Data loading infrastructure for Buttermilk.

This module provides a clean, streaming-based alternative to the pandas-heavy
prepare_step_df() approach. Each flow orchestrator uses a single DataLoader
that yields Records in an iterator pattern.
"""

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import cloudpathlib

from buttermilk._core.config import DataSourceConfig
from buttermilk._core.log import logger
from buttermilk._core.types import Record


class DataLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, config: DataSourceConfig):
        """Initialize loader with configuration.

        Args:
            config: Data source configuration specifying type, path, etc.
        """
        self.config = config

    @abstractmethod
    def __iter__(self) -> Iterator[Record]:
        """Yield Record objects from the data source."""
        pass

    def __len__(self) -> int:
        """Return number of records if known, 0 if streaming/unknown."""
        return 0


class HuggingFaceDataLoader(DataLoader):
    """Loader for HuggingFace datasets with streaming support."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        try:
            from datasets import load_dataset
            self._load_dataset = load_dataset
        except ImportError:
            raise ImportError("datasets package required for HuggingFace loader")

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from HuggingFace dataset."""
        logger.debug(f"Loading HuggingFace dataset: {self.config.path}")

        # Get Record field names dynamically
        record_fields = set(Record.model_fields.keys())

        # Load with streaming for large datasets
        dataset = self._load_dataset(
            self.config.path,
            name=getattr(self.config, "name", None),
            split=getattr(self.config, "split", "train"),
            streaming=True
        )

        for idx, item in enumerate(dataset):
            # Apply column mapping if specified
            if self.config.columns:
                mapped_item = {}
                for new_name, old_name in self.config.columns.items():
                    if old_name in item:
                        mapped_item[new_name] = item[old_name]
                processed_item = mapped_item
            else:
                processed_item = item

            # Separate Record fields from metadata
            record_kwargs = {}
            metadata_items = {}

            for key, value in processed_item.items():
                if key in record_fields:
                    record_kwargs[key] = value
                else:
                    metadata_items[key] = value

            # Set defaults for required fields
            if "record_id" not in record_kwargs:
                record_kwargs["record_id"] = f"{self.config.path}:{idx}"
            if "content" not in record_kwargs:
                record_kwargs["content"] = processed_item.get("text", str(processed_item))

            # Add loader metadata
            base_metadata = {
                "source": self.config.path,
                "loader_type": "huggingface",
            }

            # Merge with existing metadata, prioritizing existing metadata over our additions
            if "metadata" in record_kwargs:
                existing_metadata = record_kwargs["metadata"] or {}
                record_kwargs["metadata"] = {**base_metadata, **existing_metadata, **metadata_items}
            else:
                record_kwargs["metadata"] = {**base_metadata, **metadata_items}

            record = Record(**record_kwargs)
            yield record


class JSONLDataLoader(DataLoader):
    """Loader for JSONL (JSON Lines) files."""

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from JSONL file."""
        logger.debug(f"Loading JSONL file: {self.config.path}")

        # Get Record field names dynamically
        record_fields = set(Record.model_fields.keys())

        # Support both local and cloud paths
        if self.config.path.startswith(("gs://", "s3://", "azure://")):
            path = cloudpathlib.CloudPath(self.config.path)
            file_obj = path.open("r", encoding="utf-8")
        else:
            file_obj = open(self.config.path, "r", encoding="utf-8")

        try:
            for line_num, line in enumerate(file_obj):
                line_content = line.strip()
                if not line_content:
                    continue

                try:
                    item = json.loads(line_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num + 1}: {e}")
                    continue

                # Apply column mapping if specified
                if self.config.columns:
                    mapped_item = {}
                    for new_name, old_name in self.config.columns.items():
                        if old_name in item:
                            mapped_item[new_name] = item[old_name]
                    processed_item = mapped_item
                else:
                    processed_item = item

                # Separate Record fields from metadata
                record_kwargs = {}
                metadata_items = {}

                for key, value in processed_item.items():
                    if key in record_fields:
                        record_kwargs[key] = value
                    else:
                        metadata_items[key] = value

                # Set defaults for required fields
                if "record_id" not in record_kwargs:
                    record_kwargs["record_id"] = f"{self.config.path}:{line_num}"
                if "content" not in record_kwargs:
                    record_kwargs["content"] = processed_item.get("text", str(processed_item))

                # Add loader metadata
                base_metadata = {
                    "source": self.config.path,
                    "loader_type": "jsonl",
                    "line_number": line_num,
                }

                # Merge with existing metadata, prioritizing existing metadata over our additions
                if "metadata" in record_kwargs:
                    existing_metadata = record_kwargs["metadata"] or {}
                    record_kwargs["metadata"] = {**base_metadata, **existing_metadata, **metadata_items}
                else:
                    record_kwargs["metadata"] = {**base_metadata, **metadata_items}

                record = Record(**record_kwargs)
                yield record
        finally:
            file_obj.close()


class CSVDataLoader(DataLoader):
    """Loader for CSV files."""

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from CSV file."""
        logger.debug(f"Loading CSV file: {self.config.path}")

        # Get Record field names dynamically
        record_fields = set(Record.model_fields.keys())

        # Support both local and cloud paths
        if self.config.path.startswith(("gs://", "s3://", "azure://")):
            path = cloudpathlib.CloudPath(self.config.path)
            file_obj = path.open("r", encoding="utf-8")
        else:
            file_obj = open(self.config.path, "r", encoding="utf-8")

        try:
            reader = csv.DictReader(file_obj)
            for row_num, row in enumerate(reader):
                # Apply column mapping if specified
                if self.config.columns:
                    mapped_row = {}
                    for new_name, old_name in self.config.columns.items():
                        if old_name in row:
                            mapped_row[new_name] = row[old_name]
                    processed_row = mapped_row
                else:
                    processed_row = row

                # Separate Record fields from metadata
                record_kwargs = {}
                metadata_items = {}

                for key, value in processed_row.items():
                    if key in record_fields:
                        record_kwargs[key] = value
                    else:
                        metadata_items[key] = value

                # Set defaults for required fields
                if "record_id" not in record_kwargs:
                    record_kwargs["record_id"] = f"{self.config.path}:{row_num}"
                if "content" not in record_kwargs:
                    record_kwargs["content"] = processed_row.get("text", str(processed_row))

                # Add loader metadata
                base_metadata = {
                    "source": self.config.path,
                    "loader_type": "csv",
                    "row_number": row_num,
                }

                # Merge with existing metadata, prioritizing existing metadata over our additions
                if "metadata" in record_kwargs:
                    existing_metadata = record_kwargs["metadata"] or {}
                    record_kwargs["metadata"] = {**base_metadata, **existing_metadata, **metadata_items}
                else:
                    record_kwargs["metadata"] = {**base_metadata, **metadata_items}

                record = Record(**record_kwargs)
                yield record
        finally:
            file_obj.close()


class PlaintextDataLoader(DataLoader):
    """Loader for plaintext files (reads all files in a directory)."""

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from plaintext files."""
        logger.debug(f"Loading plaintext files from: {self.config.path}")

        # Get glob pattern
        pattern = getattr(self.config, "glob", "*")

        if self.config.path.startswith(("gs://", "s3://", "azure://")):
            base_path = cloudpathlib.CloudPath(self.config.path)
            files = list(base_path.glob(pattern))
        else:
            base_path = Path(self.config.path)
            files = list(base_path.glob(pattern))

        for file_path in files:
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")

                    # For plaintext, we mainly have content and metadata
                    # but we prepare the structure to handle any future Record fields
                    record_kwargs = {
                        "record_id": str(file_path.name),
                        "content": content,
                        "uri": str(file_path),
                        "metadata": {
                            "source": str(file_path),
                            "loader_type": "plaintext",
                            "filename": file_path.name,
                            "file_size": len(content)
                        }
                    }

                    record = Record(**record_kwargs)
                    yield record
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")



def create_data_loader(config: DataSourceConfig) -> DataLoader:
    """Factory function to create appropriate DataLoader for given config.

    Args:
        config: Data source configuration

    Returns:
        Configured DataLoader instance

    Raises:
        ValueError: If data source type is not supported
    """
    if config.type == "huggingface":
        return HuggingFaceDataLoader(config)
    elif config.type == "file":
        # Determine file type from extension or explicit format
        path_lower = config.path.lower()
        if path_lower.endswith(".jsonl") or path_lower.endswith(".ndjson"):
            return JSONLDataLoader(config)
        elif path_lower.endswith(".csv"):
            return CSVDataLoader(config)
        else:
            # Default to JSONL for file type
            return JSONLDataLoader(config)
    elif config.type == "plaintext":
        return PlaintextDataLoader(config)
    else:
        raise ValueError(f"Unsupported data source type: {config.type}")
