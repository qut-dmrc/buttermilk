from abc import abstractmethod
from typing import Any, AsyncGenerator, Iterable, Iterator
from pydantic import BaseModel
from buttermilk._core.runner_types import RecordInfo
from .loaders import LoaderGCS

class RecordMaker(BaseModel):
    @abstractmethod 
    async def record_generator(self) -> AsyncGenerator[RecordInfo, None]:
        for record in self.data:
            yield RecordInfo(**record)

class RecordMakerDF(RecordMaker):
    async def record_generator(self) -> AsyncGenerator[RecordInfo, None]:
        # Generator to yield records from the dataset
        for _, record in self._dataset.sample(frac=1).iterrows():
            yield RecordInfo(**record.to_dict())

class RecordMakerCloudStorageFiles(RecordMaker, LoaderGCS):
    async def record_generator(self) -> AsyncGenerator[RecordInfo, None]:
        # Generator to yield records from the dataset
        for uri, content in self.read_files():
            yield RecordInfo(uri=uri, content=content)