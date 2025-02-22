import pytest

from buttermilk._core.runner_types import Job, RecordInfo, MediaObj
from buttermilk.agents.lc import LC
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS

@pytest.fixture
def ingester():
    return Ingester()

NEWS_RECORDS = [
    (   "abc news web",
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
        "text/html",
        250
    ),]


@pytest.mark.parametrize(argvalues=NEWS_RECORDS,
    argnames=["id", "uri", "expected_mimetype", "expected_size"], ids=lambda x: x[0]
)
async def test_ingest_news(ingester, id, uri, expected_mimetype, expected_size):
    media_obj = ingester.fetch(uri=uri)
    assert media_obj.mimetype == expected_mimetype
    assert len(media_obj.data) == expected_size
    pass
    
    
