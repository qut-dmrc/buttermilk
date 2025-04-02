from buttermilk.utils.templating import load_template
from buttermilk.utils.utils import read_json


def test_template_synth():
    flow_data = read_json("tests/data/template_synth_01.json")
    params = {
        "template": "synthesise",
        "instructions": "Carefully apply EACH of the CRITERIA in order and provide a COMPLETE and SPECIFIC explanation about whether the particular rule has been violated and how. Use quotes from the content where necessary to support your analysis.",
        "criteria": "criteria_ordinary",
        "formatting": "json_rules",
    }
    rendered, unfilled = load_template(
        parameters=params,
        untrusted_inputs=flow_data,
    )
    assert not unfilled
    assert "RULE 1, TARGETS A MARGINALIZED GROUP" in rendered
    assert "Prompt is a jinja2 template that generates prompt for LLM" not in rendered
    assert "This phrase is highly ambiguous" in rendered
