import asyncio
from abc import abstractmethod
from typing import Any, AsyncGenerator

from pydantic import BaseModel

from buttermilk._core.types import Record

from .loaders import LoaderGCS


class RecordMaker(BaseModel):
    @abstractmethod
    async def record_generator(self) -> AsyncGenerator[Record, None]:
        for record in self.data:
            yield Record(**record)


class RecordMakerDF(RecordMaker):
    dataset: Any

    async def record_generator(self) -> AsyncGenerator[Record, None]:
        # Generator to yield records from the dataset
        for _, record in self.dataset.sample(frac=1).iterrows():
            yield Record(**record.to_dict())


class RecordMakerCloudStorageFiles(RecordMaker, LoaderGCS):
    async def record_generator(self) -> AsyncGenerator[Record, None]:
        # Generator to yield records from the dataset
        async for uri, content in self.read_files():
            yield Record(uri=uri, _text=content.decode())
            await asyncio.sleep(0)
