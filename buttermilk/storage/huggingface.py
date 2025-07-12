"""HuggingFace dataset storage implementation for Buttermilk."""

from typing import Iterator

from buttermilk._core.log import logger
from buttermilk._core.storage_config import StorageConfig
from buttermilk._core.types import Record
from buttermilk.storage.base import StorageError


class HuggingFaceStorage:
    """Storage interface for HuggingFace datasets."""

    def __init__(self, config: StorageConfig, bm_instance=None):
        """Initialize HuggingFace dataset storage.
        
        Args:
            config: Storage configuration
            bm_instance: BM instance for context (optional)
        """
        self.config = config
        self.bm_instance = bm_instance

        # Import datasets with proper error handling
        try:
            from datasets import load_dataset
            self._load_dataset = load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required for HuggingFace storage. "
                "Install with: pip install datasets"
            )

    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from HuggingFace dataset."""
        logger.debug(f"Loading HuggingFace dataset: {self.config.path}")

        try:
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
                try:
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
                        "storage_type": "huggingface",
                        "dataset_name": self.config.name or "default",
                        "split": self.config.split or "train",
                        "index": idx,
                    }

                    # Merge with existing metadata
                    if "metadata" in record_kwargs:
                        existing_metadata = record_kwargs["metadata"] or {}
                        record_kwargs["metadata"] = {**base_metadata, **existing_metadata, **metadata_items}
                    else:
                        record_kwargs["metadata"] = {**base_metadata, **metadata_items}

                    record = Record(**record_kwargs)
                    yield record

                    # Respect limit if set
                    if self.config.limit and idx >= self.config.limit - 1:
                        break

                except Exception as e:
                    logger.warning(f"Error processing HuggingFace dataset item {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset {self.config.path}: {e}")
            raise StorageError(f"Failed to load HuggingFace dataset: {e}") from e

    def __len__(self) -> int:
        """Get dataset length (may be expensive for streaming datasets)."""
        try:
            # For streaming datasets, this might not be available
            dataset = self._load_dataset(
                self.config.path,
                name=getattr(self.config, "name", None),
                split=getattr(self.config, "split", "train"),
                streaming=False  # Need to load fully to get length
            )
            length = len(dataset)
            return min(length, self.config.limit) if self.config.limit else length
        except Exception as e:
            logger.warning(f"Could not determine HuggingFace dataset length: {e}")
            return -1

    def save(self, records: list[Record] | Record) -> None:
        """Save records to HuggingFace dataset (not typically supported)."""
        raise StorageError(
            "HuggingFace datasets are typically read-only. "
            "Use FileStorage or BigQueryStorage for saving records."
        )

    def count(self) -> int:
        """Count records in the dataset."""
        return len(self)

    def exists(self) -> bool:
        """Check if the dataset exists and is accessible."""
        try:
            # Try to load just the dataset info
            self._load_dataset(
                self.config.path,
                name=getattr(self.config, "name", None),
                split=getattr(self.config, "split", "train"),
                streaming=True
            )
            return True
        except Exception:
            return False

    def create(self) -> None:
        """Create dataset (not applicable for HuggingFace datasets)."""
        raise StorageError(
            "Cannot create HuggingFace datasets programmatically. "
            "Datasets must exist on the HuggingFace Hub."
        )
