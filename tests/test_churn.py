# We think of predictions not as individual runs of links on chains,
# but instead as inherently stochastic insights into problems that
# are not always well-defined, by agents with varying capabilities.
##
# Accordingly, the Churn is a metaphor and methodology for repeatedly
# querying agents about a dataset from multiple perspectives over
# multiple time periods. We sample from prior answers to inform
# current predictions, adjusting for randomness and specificity.
##
# The basic unit is a Prediction, which has identifiers about the
# question, source data, agent; an answer; and a unique identifier.

from datetime import UTC, datetime

import pytest
from omegaconf import OmegaConf

from buttermilk._core.runner_types import Job
from buttermilk._core.types import SessionInfo


def test_session_info():
    agent_info = SessionInfo(
        agent="test_agent", agent_version="1.0", parameters={"param1": "value1"}
    )
    assert agent_info.agent == "test_agent"
    assert agent_info.agent_version == "1.0"
    assert run_info.parameters == {"param1": "value1"}


def test_prediction_result():
    result = PredictionResult(predicted_class="class1", predicted_result=0.75, labels=["label1", "label2"], confidence=0.9)
    assert result.predicted_class == "class1"
    assert result.predicted_result == 0.75
    assert result.labels == ["label1", "label2"]
    assert result.confidence == 0.9


def test_prediction_inputs():
    inputs = PredictionInputs(record_id="record1", parameters={"param3": "value3"})
    assert inputs.record_id == "record1"
    assert inputs.parameters == {"param3": "value3"}


def test_prediction():
    agent_info = AgentInfo(agent="test_agent", agent_version="1.0", parameters={"param1": "value1"}, step="testing")
    run_info = AgentInfo(run_id="test_run", experiment_name="exp1", parameters={"param2": "value2"})
    outputs = PredictionResult(predicted_class="class1", predicted_result=0.75, labels=["label1", "label2"], confidence=0.9)
    inputs = PredictionInputs(record_id="record1", parameters={"param3": "value3"})

    prediction = Prediction(agent_info=agent_info, run_info=run_info, outputs=outputs, inputs=inputs)

    assert isinstance(prediction.prediction_id, str)
    assert len(prediction.prediction_id) == 22  # ShortUUID length
    assert isinstance(prediction.timestamp, datetime)
    assert prediction.timestamp.tzinfo == UTC
    assert prediction.agent_info == agent_info
    assert prediction.run_info == run_info
    assert prediction.outputs == outputs
    assert prediction.inputs == inputs


def test_prediction_default_values():
    agent_info = AgentInfo(agent="test_agent", agent_version="1.0", parameters={})
    run_info = AgentInfo(run_id="test_run", experiment_name="exp1", parameters={})
    outputs = PredictionResult()
    inputs = PredictionInputs(record_id="record1", parameters={})

    prediction = Prediction(
        agent_info=agent_info, run_info=run_info, outputs=outputs, inputs=inputs
    )

    assert isinstance(prediction.prediction_id, str)
    assert len(prediction.prediction_id) == 22  # ShortUUID length
    assert isinstance(prediction.timestamp, datetime)
    assert prediction.timestamp.tzinfo == UTC


def test_prediction_result_optional_fields():
    result = PredictionResult()
    assert result.predicted_class is None
    assert result.predicted_result is None
    assert result.labels is None
    assert result.confidence is None


def test_single_flow(sample_job):
    prediction = run_flow(sample_job)
    assert isinstance(prediction, Job)
    assert prediction.agent_info.agent == "test_agent"
    assert prediction.run_info.run_id == "test_run"
    assert prediction.outputs.predicted_class == "class1"
    assert prediction.inputs.record_id == "record1"


class TestSynthesis:
    def test_get_examples(self, pail, sample_record, identifiers):
        examples = pail._get_examples(sample_record, identifiers)
        assert isinstance(examples, list)
        assert len(examples) == 1
        example = examples[0]
        assert example["record_id"] == "record1"
        assert example["parameters"] == {"param3": "value3"}


@pytest.fixture
def flow():
    return Judger(
        model="fake", template_path="judge.jinja2", criteria="criteria_ordinary.jinja2"
    )
    # lc = LangChainMulti(models=["haiku"], template_path="judge.jinja2", other_templates={"criteria": "criteria_ordinary.jinja2"})
    # return lc


DATA_CONFIGS = [
    {
        "type": "file",
        "name": "drag",
        "uri": "gs://dmrc-platforms/data/drag_train.jsonl",
        "columns": {
            "record_id": "record_id",
            "content": "alt_text",
            "groundtruth": "expected",
        },
    },
    {
        "type": "job",
        "name": "mod",
        "dataset": "dmrc-analysis.toxicity.step",
        "filter": {"max": 32},
        "group": ["record.record_id"],
        "columns": {"draft": "outputs"},
    },
]


@pytest.fixture
def data_cfg():
    return OmegaConf.create(DATA_CONFIGS)


@pytest.fixture
def expected_rows():
    return 20


async def test_load_data(data_cfg, expected_rows):
    from buttermilk.runner import load_data

    data = await load_data(data_cfg)
    assert data.shape[0] == expected_rows
