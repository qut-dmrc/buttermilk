
import random
from typing import Self, Sequence
import gcsfs
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import asyncio
from aiohttp import ClientSession

from buttermilk.utils.utils import read_file

class LoaderGCS(BaseModel):
    glob: str
    _fs: gcsfs.GCSFileSystem = PrivateAttr(default_factory=lambda: gcsfs.GCSFileSystem(asynchronous=True))
    _filelist: Sequence[str] = PrivateAttr(default_factory=list)
    _filelist_task: asyncio.Task | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def read_glob(self) -> Self:
        # Get or create event loop and start loading the list immediately
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._filelist_task = asyncio.create_task(
            asyncio.to_thread(self._fs.glob, self.glob)
        ) 
        return self
        
    async def get_filelist(self) -> Sequence[str]:
        if self._filelist is None:
            self._filelist = await self._filelist_task
        return self._filelist
    
    async def read_files_concurrently(self, *, list_files: Sequence, num_readers: int):
        # Read a set of files from GCS using several asyncio readers, and
        # yield one file at a time

        semaphore = asyncio.Semaphore(num_readers)
        async def sem_read_file(fs, file):
            async with semaphore:
                file_content = await fs.cat(file)
                yield file, file_content
        
        tasks = [sem_read_file(file) for file in list_files]
        for task in asyncio.as_completed(tasks):
            yield await task

    async def read_files(self, num_readers=16):
        list_files = await self.get_filelist()
        random.shuffle(list_files)
        
        async with self._fs:
            async for uri, file_content in self.read_files_concurrently(
                list_files=list_files, 
                num_readers=num_readers
            ):
                yield uri, file_content
                
    