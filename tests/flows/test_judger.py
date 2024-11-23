import pytest

from buttermilk.llms import CHEAP_CHAT_MODELS


@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_judger_ordinary(bm, fight_no_more_forever, model):
    judger = Judger(standards_path="criteria_ordinary.jinja2", model=model, connections=bm._connections_azure)
    output = judger(content=fight_no_more_forever["text"])
    assert output
    assert not output[COL_PREDICTION]


@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_judger_vaw(bm, fight_no_more_forever, model):
    judger = Judger(
        standards_path="criteria_vaw.jinja2",
        model=model,
        connections=bm._connections_azure,
    )
    output = judger(content=fight_no_more_forever)
    assert output
    assert not output[COL_PREDICTION]
