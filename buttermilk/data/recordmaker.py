from abc import abstractmethod
import asyncio
from typing import Any, AsyncGenerator, Iterable, Iterator
import pandas as pd
from pydantic import BaseModel
from buttermilk._core.runner_types import RecordInfo
from .loaders import LoaderGCS

class RecordMaker(BaseModel):
    
    @abstractmethod 
    async def record_generator(self) -> AsyncGenerator[RecordInfo, None]:
        for record in self.data:
            yield RecordInfo(**record)

class RecordMakerDF(RecordMaker):
    dataset: Any

    async def record_generator(self) -> AsyncGenerator[RecordInfo, None]:
        # Generator to yield records from the dataset
        for _, record in self.dataset.sample(frac=1).iterrows():
            yield RecordInfo(**record.to_dict())

class RecordMakerCloudStorageFiles(RecordMaker, LoaderGCS):
    async def record_generator(self) -> AsyncGenerator[RecordInfo, None]:
        # Generator to yield records from the dataset
        async for uri, content in self.read_files():
            yield RecordInfo(uri=uri, text=content.decode())
            await asyncio.sleep(0)