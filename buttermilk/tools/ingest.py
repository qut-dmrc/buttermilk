

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.utils.media import download_and_convert

class RecordMaker:
    async def fetch(self, uri: str) -> RecordInfo:
        logger.info("Fetching %s", uri)
        return await download_and_convert(uri=uri)
    