## We think of predictions not as individual runs of links on chains,
## but instead as inherently stochastic insights into problems that
## are not always well-defined, by agents with varying capabilities.
##
## Accordingly, the Churn is a metaphor and methodology for repeatedly
## querying agents about a dataset from multiple perspectives over 
## multiple time periods. We sample from prior answers to inform 
## current predictions, adjusting for randomness and specificity.
##
## The basic unit is a Prediction, which has identifiers about the 
## question, source data, agent; an answer; and a unique identifier.

from datetime import datetime, timezone

import pytest
import shortuuid

from buttermilk.churn.types import (
    AgentInfo,
    Prediction,
    PredictionInputs,
    PredictionResult,
    RunInfo,
)
from buttermilk.flows.lc.lc import LangChainMulti


def test_agent_info():
    agent_info = AgentInfo(agent_id="test_agent", agent_version="1.0", parameters={"param1": "value1"})
    assert agent_info.agent_id == "test_agent"
    assert agent_info.agent_version == "1.0"


def test_run_info():
    run_info = RunInfo(run_id="test_run", experiment_name="exp1", parameters={"param2": "value2"})
    assert run_info.run_id == "test_run"
    assert run_info.experiment_name == "exp1"
    assert run_info.parameters == {"param2": "value2"}

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
    agent_info = AgentInfo(agent_id="test_agent", agent_version="1.0", parameters={"param1": "value1"})
    run_info = RunInfo(run_id="test_run", experiment_name="exp1", parameters={"param2": "value2"})
    outputs = PredictionResult(predicted_class="class1", predicted_result=0.75, labels=["label1", "label2"], confidence=0.9)
    inputs = PredictionInputs(record_id="record1", parameters={"param3": "value3"})

    prediction = Prediction(agent_info=agent_info, run_info=run_info, outputs=outputs, inputs=inputs)

    assert isinstance(prediction.prediction_id, str)
    assert len(prediction.prediction_id) == 22  # ShortUUID length
    assert isinstance(prediction.timestamp, datetime)
    assert prediction.timestamp.tzinfo == timezone.utc
    assert prediction.agent_info == agent_info
    assert prediction.run_info == run_info
    assert prediction.outputs == outputs
    assert prediction.inputs == inputs

def test_prediction_default_values():
    agent_info = AgentInfo(agent_id="test_agent", agent_version="1.0", parameters={})
    run_info = RunInfo(run_id="test_run", experiment_name="exp1", parameters={})
    outputs = PredictionResult()
    inputs = PredictionInputs(record_id="record1", parameters={})

    prediction = Prediction(agent_info=agent_info, run_info=run_info, outputs=outputs, inputs=inputs)

    assert isinstance(prediction.prediction_id, str)
    assert len(prediction.prediction_id) == 22  # ShortUUID length
    assert isinstance(prediction.timestamp, datetime)
    assert prediction.timestamp.tzinfo == timezone.utc

def test_prediction_result_optional_fields():
    result = PredictionResult()
    assert result.predicted_class is None
    assert result.predicted_result is None
    assert result.labels is None
    assert result.confidence is None


def test_single_flow(sample_record, flow):
    prediction = flow.predict(sample_record)
    assert isinstance(prediction, Prediction)
    assert prediction.agent_info.agent_id == "test_agent"
    assert prediction.run_info.run_id == "test_run"
    assert prediction.outputs.predicted_class == "class1"
    assert prediction.inputs.record_id == "record1"

def test_batch_flow(sample_batch, flow):
    predictions = flow.batch_predict(sample_batch)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    for prediction in predictions:
        assert isinstance(prediction, Prediction)
        assert prediction.agent_info.agent_id == "test_agent"
        assert prediction.run_info.run_id == "test_run"
        assert prediction.outputs.predicted_class == "class1"

class TestSynthesis:
    def test_get_examples(self, pail, sample_record, identifiers):
        examples = pail._get_examples(sample_record, identifiers)
        assert isinstance(examples, list)
        assert len(examples) == 1
        example = examples[0]
        assert example["record_id"] == "record1"
        assert example["parameters"] == {"param3": "value3"}


@pytest.fixture
def pail():
    return Pail()

@pytest.fixture
def flow():
    lc = LangChainMulti(models=["haiku"], template_path="judge.jinja2", other_templates={"criteria": "criteria_ordinary.jinja2"})
    return lc
    