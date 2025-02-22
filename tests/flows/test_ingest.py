import pytest

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS

NEWS_RECORDS = [
    (
        "web page",
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
        "text/html",
    ),]

@pytest.mark.parametrize(argvalues=NEWS_RECORDS,
    argnames=[x[0] for x in NEWS_RECORDS],
)
def test_ingest_news():
    # add logic to check
    pass

