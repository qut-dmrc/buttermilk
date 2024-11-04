import buttermilk
import pytest
from buttermilk.llms import CHEAP_CHAT_MODELS
from buttermilk.utils.utils import read_text

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_analyse_template(bm, fight_no_more_forever, model):
    from buttermilk.flows.extract import Analyst
    standards = read_text("buttermilk/templates/criteria_vaw.jinja2")
    flow = Analyst(template="generic.prompty", criteria=standards, model=model)
    output = flow(content=fight_no_more_forever)
    pass
    assert output
