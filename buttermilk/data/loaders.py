
import random
from typing import Sequence
import gcsfs
from pydantic import BaseModel, Field
import asyncio
from aiohttp import ClientSession

from buttermilk.utils.utils import read_file


class LoaderGCS(BaseModel):
    glob: str

    fs: gcsfs.GCSFileSystem = Field(default_factory=lambda: gcsfs.GCSFileSystem(asynchronous=True), exclude=True)

    async def read_files_concurrently(self, *, list_files: Sequence, num_readers: int):
        # Read a set of files from GCS using several asyncio readers, and
        # yield one file at a time.

        semaphore = asyncio.Semaphore(num_readers)
        async def sem_read_file(file):
            async with semaphore:
                return await file, read_file(self.fs, file)
        
        tasks = [sem_read_file(file) for file in list_files]
        for task in asyncio.as_completed(tasks):
            yield await task

    async def read_files(self, num_readers=16):
        list_files = self.fs.glob(self.glob)
        random.shuffle(list_files)
        
        async with self.fs:
            async for uri, file_content in self.read_files_concurrently(list_files=list_files, num_readers=num_readers):
                yield uri, file_content