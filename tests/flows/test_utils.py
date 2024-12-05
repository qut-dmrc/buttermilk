

import pytest
from pydantic import Base64Str

from buttermilk._core.runner_types import Job
from buttermilk.runner.helpers import parse_flow_vars
from buttermilk.utils.templating import get_templates
from buttermilk.utils.utils import read_json


@pytest.mark.parametrize(
    "pattern, expected_name, min_length",
    [
        ("criteria", "criteria_ordinary.jinja2", 5240),
        ("criteria", "criteria_hatefb_factorised.jinja2", 9000),
        ("synth", "synthesise.jinja2", 1000),
    ],
)
def test_get_templates_default_pattern(pattern, expected_name, min_length):
    # Act
    result = get_templates(pattern)

    success = False

    for filename, content in result:
        if filename == expected_name and len(content) >= min_length:
            success = True
            break
    # Assert
    assert success


def test_b64_str_validator():
    assert pytest.raises(ValueError, match="Invalid base64 string") == Base64Str(
        "invalid"
    )
    assert Base64Str("dGVzdA==").decode() == "test"

def test_input_map():
    TEST_FLOW_ID = 'test_flow_id'
    results = read_json('tests/data/result.json')
    input_map = {"answers": ["judger.answers", "synth.answers"], "flow_id": "flow_id"}
    job = Job(flow_id=TEST_FLOW_ID,source="testing")
    
    vars = parse_flow_vars(input_map, job=job, additional_data=results)

    assert vars['answers'][0]['flow_id'] != TEST_FLOW_ID
    assert len(vars['answers'][0]['reasons']) == 4
    assert len(vars['answers'][1]['reasons']) == 3
    assert vars['answers'][1]['flow_id'] != TEST_FLOW_ID
    assert vars['flow_id'] != TEST_FLOW_ID
    pass

def test_output_map():
    TEST_FLOW_ID = 'test_flow_id'
    results = read_json('tests/data/result.json')
    job = Job(flow_id=TEST_FLOW_ID,source="testing")
    output_map = {"reasons": "judger.answers.reasons", "flow_id": "flow_id"}

    outputs = parse_flow_vars(output_map, job=job, additional_data=results)
    assert outputs['flow_id'] == TEST_FLOW_ID
    assert len(outputs['reasons']) == 4
    