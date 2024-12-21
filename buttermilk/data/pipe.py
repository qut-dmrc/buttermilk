"""DataPipe is a generic class for loading datasets, standardized for Hugging Face Datasets."""

from typing import Literal, Optional

import cloudpathlib as cp
import pandas as pd
import shortuuid
from cloudpathlib import CloudPath
from datasets import load_dataset
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk._core.runner_types import FlowResult, RecordInfo
from buttermilk.bm import BM, logger
from typing import Any, Dict, Iterator, Optional

from datasets import Dataset, load_dataset

from buttermilk._core.runner_types import RecordInfo

from typing import Any, Dict, Iterator, Optional

from datasets import Dataset, load_dataset
import pandas as pd
from cloudpathlib import CloudPath
from buttermilk._core.runner_types import RecordInfo


class DataPipe(BaseModel):
    """
    Generic base class for loading datasets, standardized for Hugging Face Datasets.
    """
    source: str = Field(..., description="A unique identifying name for this dataset.")

    dataset_kwargs: Dict[
        str, Any
    ] = {}  # Keyword arguments for Hugging Face Dataset creation.
    _dataset: Any = PrivateAttr(default=None)

    def __iter__(self) -> Iterator[RecordInfo]:
        """Yields RecordInfo objects from the Hugging Face Dataset."""
        if not self._dataset:
            raise ValueError("_dataset must be initialized in subclass")

        # Returns a record generator
        for record in self._dataset:
            yield RecordInfo(**record)


class CloudStorageDatasetPipe(DataPipe):
    """Loads datasets from Cloud Storage, converting to Hugging Face Datasets."""

    uri: str  # URI of the dataset on Cloud Storage
    format: str = (
        "csv"  # Format of the data (e.g., "csv", "json", "parquet"). Defaults to CSV.
    )
    read_kwargs: Dict[str, Any] = {}  # Keyword arguments for Pandas read functions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load data from Cloud Storage using pandas.
        try:
            if self.format == "csv":
                _data = pd.read_csv(self.uri, **self.read_kwargs)
            elif self.format == "json":
                _data = pd.read_json(self.uri, **self.read_kwargs)
            elif self.format == "parquet":
                _data = pd.read_parquet(self.uri, **self.read_kwargs)
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            self._dataset = Dataset.from_pandas(_data, **self.dataset_kwargs)

        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}") from e


class HuggingFaceDatasetPipe(DataPipe):
    """Loads datasets directly from the Hugging Face Hub."""

    path: str  # Path to the Hugging Face dataset
    name: Optional[str] = None  # Configuration name (optional)
    split: str = "train"  # Split to use
    streaming: bool = True  # Whether to stream the dataset
    shuffle: bool = True  # Whether to shuffle the dataset
    configuration: Dict[str, Any] = {}  # Keyword arguments to load_dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load dataset from Hugging Face Hub.
        try:
            self._dataset = load_dataset(
                path=self.path,
                name=self.name,
                split=self.split,
                streaming=self.streaming,
                **self.configuration,
            )
            if self.shuffle:
                self._dataset = self._dataset.shuffle()

        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}") from e
