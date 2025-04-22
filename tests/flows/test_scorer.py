import json
import pytest

from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.llms import CHATMODELS
from buttermilk._core.types import Record
from buttermilk.agents.llm import LLMAgent


@pytest.fixture(params=CHATMODELS)
def scorer(request):
    return LLMAgent(
        role="judge",
        name="judge",
        description="judger test agent",
        parameters={
            "template": "score",
            "model": request.param,
            "criteria": "criteria_ordinary",
            "formatting": "json_rules",
        },
        inputs={},
    )


@pytest.fixture
def judge_output():
    json_str = """{"role":"judge","error":[],"metadata":{"finish_reason":"stop","usage":{"prompt_tokens":1905,"completion_tokens":244},"cached":false,"logprobs":null,"thought":null},"inputs":{"role":"judge","error":[],"metadata":{},"inputs":{},"parameters":{"template":"judge","model":"gemini2flashlite","criteria":"criteria_ordinary","formatting":"json_rules"},"context":[],"records":[{"record_id":"8YxHsqsrdKQG5VweBp7hYY","metadata":{"title":"fight_no_more_forever"},"alt_text":null,"ground_truth":null,"uri":null,"content":"Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever.","mime":"text/plain", "ground_truth":"This is a classic example of counterspeech, where Chief Joseph laments the atrocities committed by British-Americans against Native American peoples."}],"prompt":"","type":"InputRequest","is_error":false},"content":null,"outputs":{"reasons":["The content appears to be a transcription of a speech given by Chief Joseph of the Nez Perce tribe. He is recounting the loss of his people and the dire conditions they are facing, including death and starvation. The speech is a plea for surrender.","RULE 1: The content is not directed at a marginalized group. While Native Americans have historically suffered from systemic discrimination, the speech does not target them on the basis of their group membership or characteristics. Instead, the speech is a reflection on the circumstances faced by the group.","RULE 2: The content does not originate from a position of power or privilege. The speaker, Chief Joseph, is clearly speaking from a position of weakness and desperation, seeking peace.","RULE 3: The content does not subordinate or treat the group as inferior. The speech is a lament for the loss of life and suffering of his people, and therefore it does not reinforce systemic discrimination."],"prediction":false,"confidence":"high","severity":"N/A","labels":[],"error":null},"type":"Agent","is_error":false}"""
    return json.loads(json_str)


@pytest.mark.anyio
async def test_run_flow_scorer(scorer, judge_output):
    input_data = AgentInput(
        role="scorer",
        inputs={"answer": judge_output["outputs"], "expected": judge_output["inputs"]["records"][-1]["ground_truth"]},
        records=judge_output["inputs"]["records"][-1:],
    )
    result = await scorer._run_fn(
        message=input_data,
    )
    assert result
    assert isinstance(result, AgentOutput)
    assert not result.error
    assert result.outputs and isinstance(result.outputs, dict)
    assert result.outputs["prediction"] is False
    assert len(result.outputs["reasons"]) > 0
    assert "joseph" in " ".join(result.outputs["reasons"]).lower()
