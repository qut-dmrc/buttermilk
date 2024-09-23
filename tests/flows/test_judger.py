import buttermilk
import pytest
from buttermilk.flows.common.config import COL_PREDICTION
from buttermilk.flows.judge.judge import Judger
from buttermilk.llms import CHEAP_CHAT_MODELS
from buttermilk.utils.utils import read_text

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_judger_ordinary(bm, fight_no_more_forever, model):
    judger = Judger(standards_path="criteria_ordinary.jinja2", model=model, connections=bm._connections_azure)
    output = judger(content=fight_no_more_forever['text'])
    assert output
    assert output[COL_PREDICTION] == False
    pass


@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_judger_vaw(bm,  fight_no_more_forever, model):
    judger = Judger(standards_path="criteria_vaw.jinja2", model=model,  connections=bm._connections_azure)
    output = judger(content=fight_no_more_forever)
    assert output
    assert output[COL_PREDICTION] == False
    pass
