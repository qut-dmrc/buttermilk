import buttermilk
import pytest
from buttermilk.flows.judge.judge import Judger
from buttermilk.llms import CHEAP_CHAT_MODELS
from buttermilk.utils.utils import read_text

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_judger_ordinary(bm, fight_no_more_forever, model):
    judger = Judger(standards_path="criteria_ordinary.jinja2", model=model)
    output = judger(content=fight_no_more_forever['text'])
    assert output
    assert output['predicted'] == False
    pass


@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_judger_vaw(bm,  fight_no_more_forever, model):
    judger = Judger(standards_path="criteria_vaw.jinja2", model=model)
    output = judger(content=fight_no_more_forever)
    assert output
    assert output['predicted'] == False
    pass

@pytest.fixture
def fight_no_more_forever() -> dict:
    record = dict(text= """Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever.""",id="ChiefJoseph1877",
                  expected=dict(violates=False))
    return record