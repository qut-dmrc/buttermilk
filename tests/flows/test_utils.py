

import pytest
from pydantic import Base64Str

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.runner.helpers import parse_flow_vars
from buttermilk.utils.templating import get_templates
from buttermilk.utils.utils import read_json


@pytest.mark.parametrize(
    "pattern, expected_name, min_length",
    [
        ("criteria", "criteria_ordinary", 5240),
        ("criteria", "criteria_hatefb_factorised", 9000),
        ("synth", "synthesise", 1000),
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
        "invalid",
    )
    assert Base64Str("dGVzdA==").decode() == "test"


def test_input_map():
    TEST_FLOW_ID = "test_flow_id"
    # Result dict here has not yet been processed, so it has a 'result' field
    # but not an 'outputs' field.
    results = read_json("tests/data/result.json")  # two results with 4 & 3 reasons
    input_map = {
        "answers": ["judger.answers", "synth.answers"],
        "object": "record",
        "flow_id": "flow_id",
    }
    job = Job(flow_id=TEST_FLOW_ID, source="testing")

    vars = parse_flow_vars(input_map, job=job, additional_data=results)

    assert len(vars["answers"]) == 2
    assert len(vars["answers"][1]["reasons"]) == 3
    assert vars["answers"][1]["flow_id"] != TEST_FLOW_ID
    assert isinstance(vars["object"], RecordInfo)

    assert vars["flow_id"] != TEST_FLOW_ID


def test_output_map():
    TEST_FLOW_ID = "test_flow_id"
    # Result dict here has not yet been processed, so it has a 'result' field
    # but not an 'outputs' field.
    results = read_json("tests/data/result.json")  # three results with 1, 4 & 3 reasons
    job = Job(flow_id=TEST_FLOW_ID, source="testing")
    output_map = {
        "answers": ["judger.outputs.reasons", "synth.outputs.reasons"],
        "synth_job_id": "job_id",
        "object": "record",
    }

    outputs = parse_flow_vars(output_map, job=job, additional_data=results)

    assert "flow_id" not in outputs
    assert len(outputs["reasons"]) == 9
    assert outputs["synth_job_id"] == job.job_id
