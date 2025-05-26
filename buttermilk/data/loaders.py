"""
Data loading infrastructure for Buttermilk.

This module provides a clean, streaming-based alternative to the pandas-heavy
prepare_step_df() approach. Each flow orchestrator uses a single DataLoader
that yields Records in an iterator pattern.
"""

import asyncio
import csv
import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Iterator, Optional, Sequence

import cloudpathlib
from cloudpathlib import GSPath
from pydantic import BaseModel, PrivateAttr

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

    def __len__(self) -> Optional[int]:
        """Return number of records if known, None if streaming/unknown."""
        return None


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

            # Create Record with metadata
            record = Record(
                record_id=processed_item.get("record_id", f"{self.config.path}:{idx}"),
                content=processed_item.get("content", processed_item.get("text", str(processed_item))),
                metadata={
                    "source": self.config.path,
                    "loader_type": "huggingface",
                    **{k: v for k, v in processed_item.items() if k not in ["record_id", "content"]}
                }
            )
            yield record


class JSONLDataLoader(DataLoader):
    """Loader for JSONL (JSON Lines) files."""

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from JSONL file."""
        logger.debug(f"Loading JSONL file: {self.config.path}")

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

                record = Record(
                    record_id=processed_item.get("record_id", f"{self.config.path}:{line_num}"),
                    content=processed_item.get("content", processed_item.get("text", str(processed_item))),
                    metadata={
                        "source": self.config.path,
                        "loader_type": "jsonl",
                        "line_number": line_num,
                        **{k: v for k, v in processed_item.items() if k not in ["record_id", "content"]}
                    }
                )
                yield record
        finally:
            file_obj.close()


class CSVDataLoader(DataLoader):
    """Loader for CSV files."""

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from CSV file."""
        logger.debug(f"Loading CSV file: {self.config.path}")

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

                record = Record(
                    record_id=processed_row.get("record_id", f"{self.config.path}:{row_num}"),
                    content=processed_row.get("content", processed_row.get("text", str(processed_row))),
                    metadata={
                        "source": self.config.path,
                        "loader_type": "csv",
                        "row_number": row_num,
                        **{k: v for k, v in processed_row.items() if k not in ["record_id", "content"]}
                    }
                )
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

                    record = Record(
                        record_id=str(file_path.name),
                        content=content,
                        metadata={
                            "source": str(file_path),
                            "loader_type": "plaintext",
                            "filename": file_path.name,
                            "file_size": len(content)
                        }
                    )
                    yield record
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")


# Legacy GCS loader (keeping for compatibility)
class LoaderGCS(BaseModel):
    uri: str
    glob: str
    _filelist: list[GSPath] = PrivateAttr(default_factory=list)

    def get_filelist(self) -> Sequence[GSPath]:
        if not self._filelist:
            for file in GSPath(self.uri).glob(self.glob):
                self._filelist.append(file)

            random.shuffle(self._filelist)

        return self._filelist

    async def read_files_concurrently(self, *, list_files: Sequence, num_readers: int):
        # Read a set of files from GCS using several asyncio readers, and
        # yield one file at a time

        semaphore = asyncio.Semaphore(num_readers)

        async def sem_read_file(file):
            async with semaphore:
                file_content = await self._fs.cat(file)
                yield file, file_content

        tasks = [sem_read_file(file) for file in list_files]
        for task in asyncio.as_completed(tasks):
            yield await task

    async def read_files(self, num_readers=16) -> AsyncGenerator:
        # Not asynchronous
        for file in self.get_filelist():
            file_content = file.read_bytes()
            yield file.as_uri(), file_content
            await asyncio.sleep(0)


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
