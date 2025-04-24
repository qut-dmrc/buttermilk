import asyncio
import random
from typing import AsyncGenerator, Sequence

from cloudpathlib import GSPath
from pydantic import BaseModel, PrivateAttr


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
