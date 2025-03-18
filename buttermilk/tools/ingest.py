

from buttermilk import logger
from buttermilk._core.runner_types import Record
from buttermilk.utils.media import download_and_convert


class RecordMaker:
    async def fetch(self, uri: str) -> Record:
        logger.info("Fetching %s", uri)
        return await download_and_convert(uri=uri)
