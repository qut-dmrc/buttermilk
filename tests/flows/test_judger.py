import buttermilk
import pytest
from buttermilk.flows.judge.judge import Judger


@pytest.fixture(scope="session")
def bm():
    return buttermilk.BM()

def test_judger_ordinary(bm, fight_no_more_forever):
    judger = Judger(standards_path="criteria_ordinary.jinja2", system_prompt_path="instructions.jinja2", process_path="process.jinja2", template_path="apply_rules.jinja2")
    output = judger(content="Hello, world!", model="gpt4o")
    assert output
    pass

def test_judger_vaw(bm,  fight_no_more_forever):
    judger = Judger(standards_path="buttermilk/examples/automod/prompts/criteria_vaw.jinja2", system_prompt_path="instructions.jinja2", process_path="process.jinja2", template_path="apply_rules.jinja2")
    output = judger(content=fight_no_more_forever, model="haiku")
    assert output
    pass

@pytest.fixture
def fight_no_more_forever() -> str:
    return """Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever."""