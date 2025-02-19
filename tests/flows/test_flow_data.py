from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.runner.helpers import parse_flow_vars
from buttermilk.utils.utils import read_json


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

    vars = parse_flow_vars(
        input_map,
        flow_data=job.model_dump(),
        additional_data=results,
    )

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
        "identifier": "identifier",
        "answers": ["judger.outputs.reasons", "synth.outputs.reasons"],
        "synth_job_id": "job_id",
        "object": "record",
    }

    outputs = parse_flow_vars(
        output_map,
        flow_data=job.model_dump(),
        additional_data=results,
    )

    assert "flow_id" not in outputs
    assert len(outputs["reasons"]) == 9
    assert outputs["synth_job_id"] == job.job_id
    assert outputs["identifier"]


def test_output_judger():
    flow_data = {"job_id": "test_output_judger", "identifier": "test_id"}
    flow_data["outputs"] = read_json("tests/data/result_judge_tja.json")
    output_map = {
        "job_id": "job_id",
        "identifier": "identifier",
        "result": {
            "reasons": "outputs.reasons",
            "prediction": "outputs.prediction",
            "confidence": "outputs.confidence",
            "severity": "outputs.severity",
            "labels": "outputs.labels",
        },
    }
    outputs = parse_flow_vars(
        output_map,
        flow_data=flow_data,
        additional_data={},
    )
    assert len(outputs["result"]["reasons"]) == 7
    assert outputs["identifier"] == "test_id"


def test_input_synth():
    flow_data = read_json("tests/data/input_synth_tja.json")
    input_map = {
        "answers": ["judger"],
    }
    inputs = parse_flow_vars(
        input_map,
        flow_data=flow_data,
        additional_data={},
    )
    assert len(inputs["answers"]) == 1
    assert inputs["answers"][0]["identifier"] == "made up human readable"
    assert len(inputs["answers"][0]["result"]["reasons"]) == 6
