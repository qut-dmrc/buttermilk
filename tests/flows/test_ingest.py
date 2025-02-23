import pytest

from buttermilk._core.runner_types import Job, RecordInfo, MediaObj
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS
from buttermilk.tools.ingest import RecordMaker   


@pytest.fixture
def ingester():
    return RecordMaker()

NEWS_RECORDS = [
    (   "abc news web",
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
        "text/html",
        5687
    ),
        ("semaphor web", "https://www.semafor.com/article/11/12/2024/after-a-stinging-loss-democrats-debate-over-where-to-draw-the-line-on-transgender-rights", "text/html", 5586)
        ]


@pytest.mark.anyio
@pytest.mark.parametrize(argvalues=NEWS_RECORDS,
    argnames=["id", "uri", "expected_mimetype", "expected_size"], ids=[x[0] for x in NEWS_RECORDS]
)
async def test_ingest_news(ingester, id, uri, expected_mimetype, expected_size):
    media_obj = await ingester.fetch(uri=uri)
    assert len(media_obj.fulltext) == expected_size
    assert media_obj.uri == uri

    pass
    
    
